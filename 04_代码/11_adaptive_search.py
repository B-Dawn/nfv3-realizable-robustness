"""
11_adaptive_search.py
检验 E：更强、更规范的动作空间攻击（单一种子先跑通，之后用于多种子正式实验）。

用法（项目根目录）：python 04_代码/11_adaptive_search.py <源数据集名> [--n 200] [--seed 0]
输入：05_模型/<源>_random_{MLP,RF,XGB}.joblib，01_数据/processed/<源>.pkl
输出：03_结果/J_adaptive_<源>_s<seed>.csv

相对 09 脚本的三处改进：
1. 多工作点：在源验证集良性误报率 0.1% / 1% / 5% 三个阈值及默认 0.5 下分别报告。
   （UNSW RF 在 FPR 1% 下阈值为 0：超过 99% 的良性流得分恰为 0，属于 RF 分数粗粒度导致的真实工作点，
     不是程序错误；但它说明"稳健"结论依赖工作点，需并列报告。）
2. 两类攻击者：
   - 判决型（hard-label）：只看到"拦/不拦"。分层随机搜索：代价档位 {0.1, 0.3, 1, 3, 10}× 平均分配查询，
     每档在 [0, 档位] 内对数均匀采样，查询总数 Q_RAND。
   - 分数型（score-based）：能看到模型分数。在随机搜索最好的若干动作上做爬山（每轮 LAM 个变异，共 ROUNDS 轮），
     目标是最小化攻击分数；代价上限 10×。查询总数 = Q_RAND + ROUNDS × LAM。
3. 累积代价曲线：样本在代价上限 c 下"成功" = 存在一次查询，其分数 ≤ 阈值且代价 ≤ c。
   代价 = max(追加包数 / 原正向包数, 追加时长 / (原时长 + 1 s))，因此成功率随 c 单调不减。
所有构造的流都用 check() 复核，违反率必须为 0。
"""
import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nfv3_constraints import apply_split, check  # noqa: E402
from nfv3_projection import HDR_OTHER, HDR_TCP, MTU, apply_actions  # noqa: E402

ROOT = HERE.parent
PROC, RES, MOD = ROOT / "01_数据" / "processed", ROOT / "03_结果", ROOT / "05_模型"
DATASETS = ["NF-UNSW-NB15-v3", "NF-CSE-CIC-IDS2018-v3", "NF-ToN-IoT-v3"]
NON_FEAT = {"Label", "Attack", "split", "FLOW_START_MILLISECONDS"}
LEVELS = [0.1, 0.3, 1.0, 3.0, 10.0]
CAPS_REPORT = [0.1, 1.0, 10.0]
Q_RAND, ROUNDS, LAM = 500, 20, 20
FPRS = [0.001, 0.01, 0.05]
LOOSE_KEYS = ("ttl_min", "ttl_max", "win_in", "flags_add")


def sample_actions(X, rng, setting, cap):
    """在 [0, cap] 内对数均匀采样动作；cap 可为逐行数组。返回动作字典与参数（用于爬山）。"""
    n = len(X)
    cap = np.broadcast_to(np.asarray(cap, float), (n,))
    p0 = np.maximum(X.IN_PKTS.to_numpy(float), 1)
    d0 = X.FLOW_DURATION_MILLISECONDS.to_numpy(float) + 1000
    tcp = np.round(X.PROTOCOL.to_numpy()) == 6
    hdr = np.where(tcp, HDR_TCP, HDR_OTHER)
    dp = np.where(rng.random(n) < 0.8, np.floor(np.expm1(rng.uniform(0, np.log1p(cap * p0), n))), 0)
    size = np.exp(rng.uniform(np.log(hdr), np.log(MTU), n))
    dt = np.where(rng.random(n) < 0.8, np.expm1(rng.uniform(0, np.log1p(cap * d0), n)), 0)
    act = {"dp": dp, "size": size, "dt": dt}
    if setting == "LOOSE":
        keep = rng.random(n) < 0.5
        t1 = rng.integers(1, 256, n).astype(float)
        act["ttl_min"] = np.where(keep, X.MIN_TTL, t1)
        act["ttl_max"] = np.where(keep, X.MAX_TTL, np.where(rng.random(n) < 0.7, t1, rng.integers(1, 256, n)))
        act["win_in"] = np.where(rng.random(n) < 0.5, X.TCP_WIN_MAX_IN, rng.integers(0, 65536, n)).astype(float)
        act["flags_add"] = np.where(rng.random(n) < 0.7, 0, rng.integers(0, 256, n)).astype(float)
    return act


def mutate(act, X, rng, setting, cap=10.0):
    n = len(X)
    p0 = np.maximum(X.IN_PKTS.to_numpy(float), 1)
    d0 = X.FLOW_DURATION_MILLISECONDS.to_numpy(float) + 1000
    tcp = np.round(X.PROTOCOL.to_numpy()) == 6
    hdr = np.where(tcp, HDR_TCP, HDR_OTHER)
    m = {}
    m["dp"] = np.clip(np.round(np.expm1(np.log1p(act["dp"]) + rng.normal(0, 0.5, n))), 0, cap * p0)
    m["size"] = np.clip(act["size"] * np.exp(rng.normal(0, 0.3, n)), hdr, MTU)
    m["dt"] = np.clip(np.expm1(np.log1p(act["dt"]) + rng.normal(0, 0.5, n)), 0, cap * d0)
    if setting == "LOOSE":
        for k in ("ttl_min", "ttl_max"):
            m[k] = np.where(rng.random(n) < 0.3, rng.integers(1, 256, n),
                            np.clip(np.round(act[k] + rng.normal(0, 10, n)), 0, 255)).astype(float)
        m["win_in"] = np.where(rng.random(n) < 0.3, rng.integers(0, 65536, n),
                               np.clip(np.round(act["win_in"] * np.exp(rng.normal(0, 0.3, n))), 0, None)).astype(float)
        flip = np.where(rng.random(n) < 0.2, 1 << rng.integers(0, 8, n), 0)
        m["flags_add"] = (act["flags_add"].astype(np.int64) ^ flip).astype(float)
    return m


def build(X, act, setting):
    a = {k: v for k, v in act.items() if k != "size"}
    a["db"] = act["dp"] * act["size"]
    if setting == "STRICT":
        a = {k: v for k, v in a.items() if k not in LOOSE_KEYS}
    return apply_actions(X, a, setting)


def cost(X, act):
    p0 = np.maximum(X.IN_PKTS.to_numpy(float), 1)
    d0 = X.FLOW_DURATION_MILLISECONDS.to_numpy(float) + 1000
    return np.maximum(act["dp"] / p0, act["dt"] / d0)


class Tracker:
    """记录每个样本在各阈值、各代价上限下是否已成功（累积）。"""

    def __init__(self, n, thrs):
        self.thrs = thrs
        self.best = {t: np.full(n, np.inf) for t in thrs}  # 成功时的最小代价
        self.viol_checked, self.viol_bad = 0, 0

    def update(self, idx, score, c, Y):
        for t, thr in self.thrs.items():
            ok = score <= thr
            if ok.any():
                np.minimum.at(self.best[t], idx[ok], c[ok])
        v = check(Y)
        self.viol_checked += len(Y)
        self.viol_bad += int(np.any(list(v.values()), axis=0).sum())

    def rate(self, t, cap):
        return float((self.best[t] <= cap + 1e-12).mean())


def run(model, feats, thrs, X0, setting, rng):
    n = len(X0)
    trk = Tracker(n, thrs)
    # ---- 判决型：分层随机搜索 ----
    per = Q_RAND // len(LEVELS)
    rep = X0.loc[X0.index.repeat(per * len(LEVELS))].reset_index(drop=True)
    idx = np.repeat(np.arange(n), per * len(LEVELS))
    caps = np.tile(np.repeat(LEVELS, per), n)
    act = sample_actions(rep, rng, setting, caps)
    Y = build(rep, act, setting)
    score = model.predict_proba(Y[feats])[:, 1]
    c = cost(rep, act)
    trk.update(idx, score, c, Y)
    hard = {t: {cap: trk.rate(t, cap) for cap in CAPS_REPORT} for t in thrs}
    # ---- 分数型：从每个样本分数最低的动作出发爬山 ----
    sc = score.reshape(n, -1)
    best_j = sc.argmin(axis=1)
    pick = np.arange(n) * sc.shape[1] + best_j
    cur = {k: v[pick] for k, v in act.items()}
    cur_s = sc[np.arange(n), best_j]
    for _ in range(ROUNDS):
        rep2 = X0.loc[X0.index.repeat(LAM)].reset_index(drop=True)
        par = {k: np.repeat(v, LAM) for k, v in cur.items()}
        m = mutate(par, rep2, rng, setting)
        Y2 = build(rep2, m, setting)
        s2 = model.predict_proba(Y2[feats])[:, 1]
        c2 = cost(rep2, m)
        idx2 = np.repeat(np.arange(n), LAM)
        trk.update(idx2, s2, c2, Y2)
        s2m = s2.reshape(n, LAM)
        j = s2m.argmin(axis=1)
        better = s2m[np.arange(n), j] < cur_s
        sel = np.arange(n) * LAM + j
        for k in cur:
            cur[k] = np.where(better, m[k][sel], cur[k])
        cur_s = np.where(better, s2m[np.arange(n), j], cur_s)
    score_based = {t: {cap: trk.rate(t, cap) for cap in CAPS_REPORT} for t in thrs}
    return hard, score_based, trk.viol_bad / max(trk.viol_checked, 1)


def main(src, n, seed, variant=None):
    """variant=None：可行性检验模式（04 脚本模型，切分种子 0）；
    variant=RAW/noTTL：正式模式（12 脚本按种子训练的模型，切分种子 = seed，与训练时一致）。"""
    t0 = time.time()
    d = apply_split(pd.read_pickle(PROC / f"{src}.pkl"), "random", seed=0 if variant is None else seed)
    allf = [c for c in d.columns if c not in NON_FEAT]
    te = d[(d.split == "test") & (d.Label == 1)][allf]
    va = d[d.split == "val"]
    rows = []
    for mname in ("MLP", "RF", "XGB"):
        path = MOD / (f"{src}_random_{mname}.joblib" if variant is None else f"{src}_{variant}_{mname}_s{seed}.joblib")
        b = joblib.load(path)
        model, feats = b["model"], b["features"]
        ben = model.predict_proba(va[feats])[:, 1][va.Label.to_numpy() == 0]
        thrs = {f"FPR {f:.1%}": float(np.quantile(ben, 1 - f)) for f in FPRS}
        thrs["默认 0.5"] = 0.5
        for setting in ("STRICT", "LOOSE"):
            rng = np.random.default_rng(seed)
            # 每个工作点各自选"被检出"的攻击样本：用阈值最高的工作点下仍被检出的样本，保证对所有工作点都是被检出的
            s_te = model.predict_proba(te[feats])[:, 1]
            det = te[s_te > max(thrs.values())]
            X0 = det.iloc[rng.choice(len(det), size=min(n, len(det)), replace=False)] if len(det) else det
            if not len(X0):
                print(f"  {src} {mname} {setting}: 无在全部工作点均被检出的攻击，跳过", flush=True)
                continue
            t1 = time.time()
            hard, sb, viol = run(model, feats, thrs, X0, setting, rng)
            for t, thr in thrs.items():
                for cap in CAPS_REPORT:
                    rows.append({"源": src, "变体": variant or "可行性", "种子": seed, "模型": mname, "设定": setting,
                                 "工作点": t, "阈值": thr,
                                 "代价上限": f"≤{cap:g}×", "样本数": len(X0),
                                 "判决型成功率": hard[t][cap], "分数型成功率": sb[t][cap], "构造流违反率": viol})
            print(f"  {src} {mname} {setting}: FPR1% ≤1× 判决型 {hard['FPR 1.0%'][1.0]:.1%} / 分数型 {sb['FPR 1.0%'][1.0]:.1%}，"
                  f"违反率 {viol:.4%}（{time.time() - t1:.0f}s）", flush=True)
    rep = pd.DataFrame(rows)
    out = RES / (f"J_adaptive_{src}_s{seed}.csv" if variant is None else f"L_formal_{src}_{variant}_s{seed}.csv")
    rep.to_csv(out, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    if len(rep):
        v = rep[rep["代价上限"] == "≤1×"].pivot_table(index=["模型", "设定"], columns="工作点",
                                                     values=["判决型成功率", "分数型成功率"], sort=False)
        print(v.round(3).to_string())
        print(f"\n构造流最大违反率：{rep['构造流违反率'].max()}（必须为 0）")
    print(f"已写入 {out}，用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", choices=DATASETS)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default=None, choices=["RAW", "noTTL"],
                    help="不指定 = 可行性检验模式；RAW/noTTL = 正式模式（读取 12 脚本的模型，输出 L_formal_*.csv）")
    a = ap.parse_args()
    main(a.src, a.n, a.seed, a.variant)
