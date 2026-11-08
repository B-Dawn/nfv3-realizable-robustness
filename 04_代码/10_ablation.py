"""
10_ablation.py
检验 D：STRICT → LOOSE 成功率跳升的归因（单一种子）。

用法（项目根目录）：
  python 04_代码/10_ablation.py actions <源数据集名>    # 动作消融（使用 04 脚本保存的 RAW 模型）
  python 04_代码/10_ablation.py nottl                    # UNSW 去 TTL 重训对照
输出：03_结果/I_ablation_<源>.csv（可行性模式）或 I_ablation_<源>_s<seed>.csv（正式模式：actions <源> --seed s，读取 12 脚本 RAW 模型）；03_结果/I_nottl_UNSW.csv

(1) 动作消融：在 STRICT（追加包 + 放慢）基础上每次只多开放一类动作：
    +TTL、+窗口、+标志；另加"只改 TTL（不加包不放慢）"与 LOOSE 全开作对照。
    代价上限 ≤1×，每样本 500 次查询，n=200（与 09 脚本一致，可直接比较）。
(2) UNSW 去 TTL 重训：去掉 MIN_TTL、MAX_TTL，按 04 脚本相同设置训练 RF/XGB/MLP（RAW 其余特征），
    报告同数据集检测性能（AUROC、源阈值下 TPR/FPR）与 STRICT / LOOSE 动作空间搜索成功率，
    与含 TTL 模型对照。若去 TTL 后检测几乎不降而 LOOSE 成功率明显回落，则支持"TTL 是可被攻击者利用的泄漏捷径"。
"""
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
from nfv3_projection import apply_actions, random_actions  # noqa: E402

ROOT = HERE.parent
PROC, RES, MOD = ROOT / "01_数据" / "processed", ROOT / "03_结果", ROOT / "05_模型"
DATASETS = ["NF-UNSW-NB15-v3", "NF-CSE-CIC-IDS2018-v3", "NF-ToN-IoT-v3"]
NON_FEAT = {"Label", "Attack", "split", "FLOW_START_MILLISECONDS"}
SEED, N, Q, CAP = 0, 200, 500, 1.0
TTL = ["MIN_TTL", "MAX_TTL"]
# 动作组合：A1/A2 键 = dp、db、dt；A3 = ttl_min/ttl_max；A4 = win_in；A5 = flags_add
GROUPS = {
    "STRICT（加包+放慢）": {"dp", "db", "dt"},
    "STRICT+TTL": {"dp", "db", "dt", "ttl_min", "ttl_max"},
    "STRICT+窗口": {"dp", "db", "dt", "win_in"},
    "STRICT+标志": {"dp", "db", "dt", "flags_add"},
    "只改TTL": {"ttl_min", "ttl_max"},
    "LOOSE（全开）": {"dp", "db", "dt", "ttl_min", "ttl_max", "win_in", "flags_add"},
}


def any_violation(X):
    v = check(X)
    return float(np.mean(np.any(list(v.values()), axis=0))) if len(X) else float("nan")


def search(model, feats, thr, X0, keys, rng):
    """动作空间随机搜索，只保留 keys 中的动作。X0 为全部 46 个特征（用于重建），预测时只取模型特征。"""
    n = len(X0)
    rep = X0.loc[X0.index.repeat(Q)].reset_index(drop=True)
    act = random_actions(rep, rng, "LOOSE", max_pkt_mult=CAP, max_dt_mult=CAP)
    act = {k: v for k, v in act.items() if k in keys}
    Y = apply_actions(rep, act, "LOOSE")
    ok = (model.predict_proba(Y[feats])[:, 1] <= thr).reshape(n, Q)
    succ = ok.any(axis=1)
    first = ok.argmax(axis=1)
    idx = (np.arange(n) * Q + first)[succ]
    return float(succ.mean()), any_violation(Y.iloc[idx])


def pick(model, feats, thr, te_all, rng):
    det = te_all[model.predict_proba(te_all[feats])[:, 1] > thr]
    return det.iloc[rng.choice(len(det), size=min(N, len(det)), replace=False)] if len(det) else det


def actions(src, seed=None):
    t0 = time.time()
    d = apply_split(pd.read_pickle(PROC / f"{src}.pkl"), "random", seed=0 if seed is None else seed)
    allf = [c for c in d.columns if c not in NON_FEAT]
    te = d[(d.split == "test") & (d.Label == 1)][allf]
    rows = []
    for mname in ("MLP", "RF", "XGB"):
        rng = np.random.default_rng(SEED if seed is None else seed)
        b = joblib.load(MOD / (f"{src}_random_{mname}.joblib" if seed is None else f"{src}_RAW_{mname}_s{seed}.joblib"))
        model, feats, thr = b["model"], b["features"], b["threshold"]
        X0 = pick(model, feats, thr, te, rng)
        for g, keys in GROUPS.items():
            sr, vr = search(model, feats, thr, X0, keys, rng)
            rows.append({"源": src, "种子": seed, "模型": mname, "动作组合": g, "样本数": len(X0), "成功率": sr, "成功样本违反率": vr})
            print(f"  {src} {mname} {g}: {sr:.1%}", flush=True)
    rep = pd.DataFrame(rows)
    out = RES / (f"I_ablation_{src}.csv" if seed is None else f"I_ablation_{src}_s{seed}.csv")
    rep.to_csv(out, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(rep.pivot_table(index="模型", columns="动作组合", values="成功率", sort=False).round(3).to_string())
    print(f"\n成功样本最大违反率：{rep['成功样本违反率'].max()}（必须为 0）")
    print(f"已写入 {out}，用时 {time.time() - t0:.0f}s")


def nottl():
    import importlib.util
    spec = importlib.util.spec_from_file_location("b04", HERE / "04_baseline_artifact.py")
    b04 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b04)
    t0 = time.time()
    src = "NF-UNSW-NB15-v3"
    d = apply_split(pd.read_pickle(PROC / f"{src}.pkl"), "random")
    allf = [c for c in d.columns if c not in NON_FEAT]
    tr = b04.cap(d[d.split == "train"], b04.N_TRAIN, SEED)
    va, te = d[d.split == "val"], b04.cap(d[d.split == "test"], b04.N_TEST, SEED)
    te_att = d[(d.split == "test") & (d.Label == 1)][allf]
    rows = []
    for variant in ("含TTL", "去TTL"):
        feats = allf if variant == "含TTL" else [c for c in allf if c not in TTL]
        for mname, model in b04.models().items():
            rng = np.random.default_rng(SEED)
            if variant == "含TTL":
                bnd = joblib.load(MOD / f"{src}_random_{mname}.joblib")
                model, feats, thr = bnd["model"], bnd["features"], bnd["threshold"]
            else:
                model.fit(tr[feats], tr.Label)
                ben = model.predict_proba(va[feats])[:, 1][va.Label.to_numpy() == 0]
                thr = float(np.quantile(ben, 0.99))
            sc = model.predict_proba(te[feats])[:, 1]
            y = te.Label.to_numpy()
            pred = sc > thr
            X0 = pick(model, feats, thr, te_att, rng)
            res = {"变体": variant, "模型": mname, "特征数": len(feats), "阈值": thr,
                   "AUROC": roc_auc_score(y, sc), "TPR@源阈值": float(pred[y == 1].mean()),
                   "FPR@源阈值": float(pred[y == 0].mean()), "攻击样本数": len(X0)}
            for g in ("STRICT（加包+放慢）", "LOOSE（全开）"):
                sr, vr = search(model, feats, thr, X0, GROUPS[g], rng) if len(X0) else (np.nan, np.nan)
                res[f"成功率|{g}"], res[f"违反率|{g}"] = sr, vr
            rows.append(res)
            print(f"  {variant} {mname}: AUROC {res['AUROC']:.4f}，STRICT {res['成功率|STRICT（加包+放慢）']:.1%}，"
                  f"LOOSE {res['成功率|LOOSE（全开）']:.1%}（{time.time() - t0:.0f}s）", flush=True)
    rep = pd.DataFrame(rows)
    out = RES / "I_nottl_UNSW.csv"
    rep.to_csv(out, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(rep.round(4).to_string(index=False))
    print(f"\n已写入 {out}，用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    args = sys.argv[1:]
    seed = None
    if "--seed" in args:
        i = args.index("--seed")
        seed = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    if len(args) == 2 and args[0] == "actions" and args[1] in DATASETS:
        actions(args[1], seed)
    elif len(args) == 1 and args[0] == "nottl":
        nottl()
    else:
        sys.exit("用法: python 04_代码/10_ablation.py actions <源数据集名> [--seed s] | nottl")
