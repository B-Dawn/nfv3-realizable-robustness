"""
09_constrained_attack.py
检验 C：只允许攻击者真实可做的动作时，模型还剩多少脆弱性？（单一种子，先跑通）

用法（项目根目录）：python 04_代码/09_constrained_attack.py <源数据集名> [--n 200] [--queries 500]
输入：05_模型/<源>_random_{MLP,RF,XGB}.joblib，01_数据/processed/<源>.pkl
输出：03_结果/H_constrained_<源>.csv（可行性模式）；03_结果/P_methods_<源>_s<seed>.csv（正式模式：--seed，读取 12 脚本 RAW 模型，切分种子 = seed）

两种方式（STRICT = 追加包 + 放慢；LOOSE = 另可改 TTL、客户端窗口、追加 TCP 标志）：
 (A) 事后投影：常规攻击（MLP：PGD ε=1.0；RF/XGB：HopSkipJump，前 50 个样本）→ 投影 P → 是否仍绕过检测。
 (B) 动作空间随机搜索：每个样本随机采样 Q 组动作，任一组绕过检测即成功。
     代价上限两档：低（追加包 ≤ 1× 原正向包数，追加时长 ≤ 1×(原时长+1 s)）、高（均为 10×）。
     记录成功样本中"总代价最小"那一组动作的包数倍数与时长倍数（中位数）。
所有成功样本都用 check() 复核约束违反率，必须为 0%。
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nfv3_constraints import apply_split, check  # noqa: E402
from nfv3_projection import apply_actions, project, random_actions  # noqa: E402

ROOT = HERE.parent
PROC, RES, MOD = ROOT / "01_数据" / "processed", ROOT / "03_结果", ROOT / "05_模型"
DATASETS = ["NF-UNSW-NB15-v3", "NF-CSE-CIC-IDS2018-v3", "NF-ToN-IoT-v3"]
SEED = 0
CAPS = {"低代价(≤1×)": 1.0, "高代价(≤10×)": 10.0}

_spec = importlib.util.spec_from_file_location("atk05", HERE / "05_unconstrained_attack.py")
atk05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(atk05)


def any_violation(X):
    v = check(X)
    return float(np.mean(np.any(list(v.values()), axis=0))) if len(X) else float("nan")


def search(model, thr, X0, setting, cap, Q, rng):
    """动作空间随机搜索。返回 (成功标记, 最小代价成功流, 包数倍数, 时长倍数)。"""
    n = len(X0)
    rep = X0.loc[X0.index.repeat(Q)].reset_index(drop=True)
    act = random_actions(rep, rng, setting, max_pkt_mult=cap, max_dt_mult=cap)
    Y = apply_actions(rep, act, setting)
    ok = model.predict_proba(Y)[:, 1] <= thr
    p0 = np.maximum(rep.IN_PKTS.to_numpy(float), 1)
    d0 = rep.FLOW_DURATION_MILLISECONDS.to_numpy(float) + 1000
    pr, tr = act["dp"] / p0, act["dt"] / d0
    cost = np.where(ok, pr + tr, np.inf).reshape(n, Q)
    best = cost.argmin(axis=1)
    succ = np.isfinite(cost[np.arange(n), best])
    idx = np.arange(n) * Q + best
    return succ, Y.iloc[idx[succ]], pr[idx[succ]], tr[idx[succ]]


def main(src, n, Q, seed=None):
    t0 = time.time()
    rng = np.random.default_rng(SEED if seed is None else seed)
    d = apply_split(pd.read_pickle(PROC / f"{src}.pkl"), "random", seed=0 if seed is None else seed)
    te = d[(d.split == "test") & (d.Label == 1)]
    tr = d[d.split == "train"]
    rows = []
    for mname in ("MLP", "RF", "XGB"):
        b = joblib.load(MOD / (f"{src}_random_{mname}.joblib" if seed is None else f"{src}_RAW_{mname}_s{seed}.joblib"))
        model, feats, thr = b["model"], b["features"], b["threshold"]
        det = te[feats][model.predict_proba(te[feats])[:, 1] > thr]
        if not len(det):
            print(f"  {src} {mname}: 无被检出攻击，跳过", flush=True)
            continue
        X0 = det.iloc[rng.choice(len(det), size=min(n, len(det)), replace=False)]
        base = {"源": src, "种子": seed, "模型": mname, "阈值": thr}

        # (A) 事后投影
        t1 = time.time()
        if mname == "MLP":
            XA0 = X0
            adv = atk05.pgd(model, X0.to_numpy(np.float64), 1.0)
            label = "PGD ε=1.0"
        else:
            XA0 = X0.iloc[:50]
            trb = tr[tr.Label == 0][feats]
            trb = trb.iloc[rng.choice(len(trb), size=min(20_000, len(trb)), replace=False)]
            ben = trb[model.predict_proba(trb)[:, 1] <= thr].to_numpy(np.float32)
            adv, _ = atk05.hsj(model, XA0.to_numpy(np.float32), tr[feats].to_numpy(np.float32), thr, feats, ben)
            label = "HopSkipJump"
        A = pd.DataFrame(adv, columns=feats, index=XA0.index)
        s_un = float((model.predict_proba(A)[:, 1] <= thr).mean())
        for st in ("STRICT", "LOOSE"):
            P = project(XA0, A, st)
            ok = model.predict_proba(P)[:, 1] <= thr
            rows.append({**base, "方式": f"(A) 事后投影｜{label}", "设定": st, "代价上限": "—", "样本数": len(XA0),
                         "无约束成功率": s_un, "约束内成功率": float(ok.mean()),
                         "成功样本违反率": any_violation(P[ok]), "包数倍数中位数": np.nan, "时长倍数中位数": np.nan})
        print(f"  {src} {mname} (A) 完成：无约束 {s_un:.0%}（{time.time() - t1:.0f}s）", flush=True)

        # (B) 动作空间随机搜索
        for st in ("STRICT", "LOOSE"):
            for cname, cap in CAPS.items():
                t1 = time.time()
                succ, Y, pr, trt = search(model, thr, X0, st, cap, Q, rng)
                rows.append({**base, "方式": f"(B) 动作空间搜索｜{Q} 次查询", "设定": st, "代价上限": cname,
                             "样本数": len(X0), "无约束成功率": np.nan, "约束内成功率": float(succ.mean()),
                             "成功样本违反率": any_violation(Y),
                             "包数倍数中位数": float(np.median(pr)) if len(pr) else np.nan,
                             "时长倍数中位数": float(np.median(trt)) if len(trt) else np.nan})
                print(f"  {src} {mname} (B) {st} {cname}: 成功率 {succ.mean():.1%}（{time.time() - t1:.0f}s）", flush=True)
    rep = pd.DataFrame(rows)
    out = RES / (f"H_constrained_{src}.csv" if seed is None else f"P_methods_{src}_s{seed}.csv")
    rep.to_csv(out, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    show = ["模型", "方式", "设定", "代价上限", "样本数", "无约束成功率", "约束内成功率", "成功样本违反率", "包数倍数中位数", "时长倍数中位数"]
    print(rep[show].round(4).to_string(index=False))
    bad = rep["成功样本违反率"].fillna(0) > 0
    if bad.any():
        print("\n警告：存在违反约束的成功样本，投影算子有误，结果不可用")
    print(f"\n已写入 {out}，用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", choices=DATASETS)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--seed", type=int, default=None, help="指定则为正式模式")
    a = ap.parse_args()
    main(a.src, a.n, a.queries, a.seed)
