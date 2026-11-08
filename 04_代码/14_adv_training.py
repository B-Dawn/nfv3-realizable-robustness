"""
14_adv_training.py
贡献 C5：约束一致 vs 普通对抗训练（防御对比）。

用法（项目根目录）：python 04_代码/14_adv_training.py <源数据集名> --seed 0
输入：01_数据/processed/<源>.pkl；05_模型/<源>_RAW_<模型>_s<seed>.joblib（12 脚本，作 D0 与 D3 的攻击对象）
输出：05_模型/<源>_<D1|D2|D3>_<模型>_s<seed>.joblib
      03_结果/N_at_<源>_s<seed>.csv

四种防御（同一训练集；D1–D3 均只追加 N_AUG 条增强样本，标签为攻击，差别只在样本怎么造）：
  D0 不防御：12 脚本的 RAW 模型。
  D1 无约束增强：对训练集攻击流在 log1p 空间对全部特征加高斯噪声（σ=0.5），截零——现实中不可能出现的流。
  D2 约束内随机增强：对训练集攻击流随机施加 LOOSE 动作（代价 ≤10×）——物理可实现，但不针对模型弱点。
  D1′ 无约束对抗训练：以 D0 为对象，在 log1p 空间对全部特征做不受约束的分数型搜索，取最难变体（查询预算同 D3）——
     对齐 Dyrmishi 2023 / Alhussien 2024 的"无约束对抗样本加固"。（2026-11-07 精读后补，缺口 G4）
  D3 约束内对抗训练：以 D0 模型为对象，对训练集攻击流做分数型动作空间搜索（LOOSE，≤10×），
     取每条流分数最低的变体加入训练——标准约束对抗训练（一轮）。
评估（全部在同一测试集）：
  检测：AUROC；阈值在验证集上对每个防御模型重新标定（FPR 0.1%/1%/5%），另加默认 0.5。
  鲁棒：自适应攻击——攻击者直接针对该防御模型重新搜索（11 脚本 run()，判决型 + 分数型，STRICT / LOOSE）。
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nfv3_constraints import apply_split, check  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


atk = _load("atk11", "11_adaptive_search.py")
trn = _load("trn12", "12_formal_train.py")

ROOT = HERE.parent
PROC, RES, MOD = ROOT / "01_数据" / "processed", ROOT / "03_结果", ROOT / "05_模型"
DATASETS = ["NF-UNSW-NB15-v3", "NF-CSE-CIC-IDS2018-v3", "NF-ToN-IoT-v3"]
NON_FEAT = {"Label", "Attack", "split", "FLOW_START_MILLISECONDS"}
N_AUG, N_EVAL = 3000, 200
FPRS = [0.001, 0.01, 0.05]


def aug_noise(X, rng):
    Z = np.log1p(X.clip(lower=0).to_numpy(np.float64)) + rng.normal(0, 0.5, X.shape)
    return pd.DataFrame(np.expm1(np.clip(Z, 0, None)), columns=X.columns)


def aug_random(X, rng):
    act = atk.sample_actions(X.reset_index(drop=True), rng, "LOOSE", 10.0)
    return atk.build(X.reset_index(drop=True), act, "LOOSE")


def aug_adversarial(X, model, feats, rng):
    """分数型动作空间搜索，返回每条流分数最低的变体（不论是否已绕过）。"""
    X = X.reset_index(drop=True)
    n, per = len(X), 100
    rep = X.loc[X.index.repeat(per)].reset_index(drop=True)
    act = atk.sample_actions(rep, rng, "LOOSE", 10.0)
    Y = atk.build(rep, act, "LOOSE")
    s = model.predict_proba(Y[feats])[:, 1].reshape(n, per)
    j = s.argmin(axis=1)
    pick = np.arange(n) * per + j
    cur = {k: v[pick] for k, v in act.items()}
    cur_s = s[np.arange(n), j]
    for _ in range(10):
        rep2 = X.loc[X.index.repeat(atk.LAM)].reset_index(drop=True)
        par = {k: np.repeat(v, atk.LAM) for k, v in cur.items()}
        m = atk.mutate(par, rep2, rng, "LOOSE")
        Y2 = atk.build(rep2, m, "LOOSE")
        s2 = model.predict_proba(Y2[feats])[:, 1].reshape(n, atk.LAM)
        j2 = s2.argmin(axis=1)
        better = s2[np.arange(n), j2] < cur_s
        sel = np.arange(n) * atk.LAM + j2
        for k in cur:
            cur[k] = np.where(better, m[k][sel], cur[k])
        cur_s = np.where(better, s2[np.arange(n), j2], cur_s)
    return atk.build(X, cur, "LOOSE"), float((cur_s <= 0.5).mean())


def aug_unconstrained_adversarial(X, model, feats, rng):
    """
    D1′ 无约束对抗训练样本：在 log1p 空间对全部特征做分数型搜索（不受任何约束），取分数最低的变体。
    与 D3 查询预算相同（100 个随机起点 + 10 轮 × LAM 个变异），只差"是否受约束"，用于与
    Dyrmishi 2023、Alhussien 2024 的"无约束对抗样本加固"对齐比较。
    """
    X = X.reset_index(drop=True)
    n, per = len(X), 100
    Z0 = np.log1p(X.clip(lower=0).to_numpy(np.float64))
    sig = rng.choice([0.1, 0.3, 0.5, 1.0], size=(n, per, 1))
    Z = Z0[:, None, :] + rng.normal(0, 1, (n, per, Z0.shape[1])) * sig
    Y = pd.DataFrame(np.expm1(np.clip(Z, 0, None)).reshape(n * per, -1), columns=X.columns)
    s = model.predict_proba(Y[feats])[:, 1].reshape(n, per)
    j = s.argmin(axis=1)
    cur = Z[np.arange(n), j]
    cur_s = s[np.arange(n), j]
    for _ in range(10):
        cand = cur[:, None, :] + rng.normal(0, 0.2, (n, atk.LAM, Z0.shape[1]))
        Yc = pd.DataFrame(np.expm1(np.clip(cand, 0, None)).reshape(n * atk.LAM, -1), columns=X.columns)
        sc = model.predict_proba(Yc[feats])[:, 1].reshape(n, atk.LAM)
        jj = sc.argmin(axis=1)
        better = sc[np.arange(n), jj] < cur_s
        cur = np.where(better[:, None], cand[np.arange(n), jj], cur)
        cur_s = np.where(better, sc[np.arange(n), jj], cur_s)
    return pd.DataFrame(np.expm1(np.clip(cur, 0, None)), columns=X.columns)


def thresholds(model, feats, va):
    ben = model.predict_proba(va[feats])[:, 1][va.Label.to_numpy() == 0]
    t = {f"FPR {f:.1%}": float(np.quantile(ben, 1 - f)) for f in FPRS}
    t["默认 0.5"] = 0.5
    return t


def main(src, seed):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    d = apply_split(pd.read_pickle(PROC / f"{src}.pkl"), "random", seed=seed)
    allf = [c for c in d.columns if c not in NON_FEAT]
    b04 = trn.b04
    tr = b04.cap(d[d.split == "train"], b04.N_TRAIN, seed)
    va = d[d.split == "val"]
    te = b04.cap(d[d.split == "test"], b04.N_TEST, seed)
    te_att = d[(d.split == "test") & (d.Label == 1)][allf]
    tr_att = tr[tr.Label == 1][allf]
    base_X = tr[allf].reset_index(drop=True)
    base_y = tr.Label.to_numpy()
    rows = []
    for mname in ("MLP", "RF", "XGB"):
        b0 = joblib.load(MOD / f"{src}_RAW_{mname}_s{seed}.joblib")
        m0, feats = b0["model"], b0["features"]
        src_rows = tr_att.iloc[rng.choice(len(tr_att), size=min(N_AUG, len(tr_att)), replace=False)]
        defenses = {"D0 不防御": (m0, None)}
        t1 = time.time()
        Xa1 = aug_noise(src_rows, rng)
        Xa2 = aug_random(src_rows, rng)
        Xa3, pre_evade = aug_adversarial(src_rows, m0, feats, rng)
        Xa1b = aug_unconstrained_adversarial(src_rows, m0, feats, np.random.default_rng(seed + 2000))
        viol = {k: float(np.mean(np.any(list(check(v).values()), axis=0))) for k, v in
                (("D1", Xa1), ("D2", Xa2), ("D3", Xa3), ("D1b", Xa1b))}
        print(f"  {src} s{seed} {mname}: 增强样本违反率 D1 {viol['D1']:.1%} / D2 {viol['D2']:.1%} / D3 {viol['D3']:.1%}；"
              f"D3 样本在 0.5 下已绕过 D0 的比例 {pre_evade:.1%}（{time.time() - t1:.0f}s）", flush=True)
        for key, Xa, name in (("D1", Xa1, "D1 无约束增强"), ("D1b", Xa1b, "D1′ 无约束对抗训练"),
                              ("D2", Xa2, "D2 约束内随机增强"), ("D3", Xa3, "D3 约束内对抗训练")):
            path = MOD / f"{src}_{key}_{mname}_s{seed}.joblib"
            if path.exists():
                m = joblib.load(path)["model"]
            else:
                m = trn.models(seed)[mname]
                Xt = pd.concat([base_X[feats], Xa[feats]], ignore_index=True)
                yt = np.concatenate([base_y, np.ones(len(Xa), dtype=int)])
                m.fit(Xt, yt)
                joblib.dump({"model": m, "features": feats, "seed": seed, "defense": key}, path)
            defenses[name] = (m, viol[key])
        for name, (m, aug_viol) in defenses.items():
            t2 = time.time()
            thrs = thresholds(m, feats, va)
            sc = m.predict_proba(te[feats])[:, 1]
            y = te.Label.to_numpy()
            det = {"AUROC": roc_auc_score(y, sc)}
            for k, thr in thrs.items():
                det[f"TPR|{k}"] = float((sc[y == 1] > thr).mean())
                det[f"FPR|{k}"] = float((sc[y == 0] > thr).mean())
            s_att = m.predict_proba(te_att[feats])[:, 1]
            pool = te_att[s_att > max(thrs.values())]
            arng = np.random.default_rng(seed + 1000)
            X0 = pool.iloc[arng.choice(len(pool), size=min(N_EVAL, len(pool)), replace=False)] if len(pool) else pool
            for setting in ("STRICT", "LOOSE"):
                if not len(X0):
                    continue
                hard, sb, v = atk.run(m, feats, thrs, X0, setting, arng)
                for t in thrs:
                    for cap in atk.CAPS_REPORT:
                        rows.append({"源": src, "种子": seed, "模型": mname, "防御": name, "增强样本违反率": aug_viol,
                                     **det, "设定": setting, "工作点": t, "阈值": thrs[t], "代价上限": f"≤{cap:g}×",
                                     "攻击样本数": len(X0), "判决型成功率": hard[t][cap], "分数型成功率": sb[t][cap],
                                     "构造流违反率": v})
            print(f"    {name}: AUROC {det['AUROC']:.4f}，TPR@FPR1% {det['TPR|FPR 1.0%']:.4f}（{time.time() - t2:.0f}s）", flush=True)
    rep = pd.DataFrame(rows)
    out = RES / f"N_at_{src}_s{seed}.csv"
    rep.to_csv(out, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    v = rep[(rep["代价上限"] == "≤1×") & (rep["工作点"] == "FPR 1.0%")].pivot_table(
        index=["模型", "防御"], columns="设定", values="分数型成功率", sort=False)
    a = rep.drop_duplicates(["模型", "防御"]).set_index(["模型", "防御"])[["AUROC", "TPR|FPR 1.0%"]]
    print(v.join(a).round(4).to_string())
    print(f"\n构造流最大违反率：{rep['构造流违反率'].max()}（必须为 0）")
    print(f"已写入 {out}，用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", choices=DATASETS)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.src, a.seed)
