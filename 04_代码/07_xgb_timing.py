"""
07_xgb_timing.py
排查：为什么 XGB 的 HopSkipJump 攻击在 CIC 上极慢（UNSW 约 5 秒，CIC 跑 50 分钟未完成）。

用法（项目根目录）：python 04_代码/07_xgb_timing.py
输入：05_模型/<源>_random_XGB.joblib，01_数据/processed/<源>.pkl
输出：03_结果/F_xgb_timing.csv（预测延迟）、03_结果/F_xgb_timing.txt（模型规模 + 微型 HSJ 用时）

检查三件事：
1. 模型规模：树数、叶子总数（CIC 是否比 UNSW 大很多）。
2. 单次预测延迟：批大小 1/10/100/1000 × 线程数 n_jobs ∈ {原设置 -1, 1, 4}。
   HopSkipJump 的二分搜索会大量调用单样本预测，所以批大小 1 的延迟最关键。
3. 微型 HSJ：每个数据集 2 个攻击样本，分别在 n_jobs=1 与 n_jobs=-1 下计时。
保护：整个脚本超过 600 秒自动写出已有结果并退出，避免卡死占满 CPU。
"""
import importlib.util
import os
import sys
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nfv3_constraints import apply_split  # noqa: E402

ROOT = HERE.parent
PROC, RES, MOD = ROOT / "01_数据" / "processed", ROOT / "03_结果", ROOT / "05_模型"
SRCS = ["NF-UNSW-NB15-v3", "NF-CSE-CIC-IDS2018-v3"]
LIMIT_S = 600
rows, lines = [], [f"CPU 逻辑核数：{os.cpu_count()}"]


def flush(note=""):
    if rows:
        pd.DataFrame(rows).to_csv(RES / "F_xgb_timing.csv", index=False, encoding="utf-8-sig")
    (RES / "F_xgb_timing.txt").write_text("\n".join(lines + ([note] if note else [])), encoding="utf-8")


def watchdog():
    flush(f"【超时保护】脚本运行超过 {LIMIT_S} 秒，已强制退出；以上为已完成部分。")
    print(f"超过 {LIMIT_S} 秒，强制退出", flush=True)
    os._exit(2)


threading.Timer(LIMIT_S, watchdog).start()

spec = importlib.util.spec_from_file_location("atk", HERE / "05_unconstrained_attack.py")
atk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(atk)

for src in SRCS:
    b = joblib.load(MOD / f"{src}_random_XGB.joblib")
    m, feats, thr = b["model"], b["features"], b["threshold"]
    tdf = m.get_booster().trees_to_dataframe()
    n_trees, n_leaves = tdf.Tree.nunique(), int((tdf.Feature == "Leaf").sum())
    lines.append(f"{src}：树 {n_trees} 棵，叶子 {n_leaves:,} 个，平均每棵 {n_leaves / n_trees:.0f} 个，阈值 {thr:.6f}")
    print(lines[-1], flush=True)

    d = apply_split(pd.read_pickle(PROC / f"{src}.pkl"), "random")
    X = d[feats].sample(1000, random_state=0)
    for nj in (-1, 1, 4):
        m.set_params(n_jobs=nj)
        for bs in (1, 10, 100, 1000):
            Xb = X.iloc[:bs]
            m.predict_proba(Xb)  # 预热
            reps = max(3, 200 // bs)
            t = time.perf_counter()
            for _ in range(reps):
                m.predict_proba(Xb)
            ms = (time.perf_counter() - t) / reps * 1000
            rows.append({"源": src, "n_jobs": nj, "批大小": bs, "每次调用毫秒": round(ms, 3)})
            print(f"  n_jobs={nj} 批={bs}: {ms:.2f} ms/次", flush=True)
    flush()

    # 微型 HSJ：2 个被检出的攻击样本
    te = d[(d.split == "test") & (d.Label == 1)][feats]
    m.set_params(n_jobs=-1)
    det = te[m.predict_proba(te)[:, 1] > thr]
    X0 = det.sample(min(2, len(det)), random_state=0).to_numpy(np.float32)
    tr = d[d.split == "train"]
    trb = tr[tr.Label == 0][feats].sample(20_000, random_state=0)
    ben = trb[m.predict_proba(trb)[:, 1] <= thr].to_numpy(np.float32)
    Xtr = tr[feats].to_numpy(np.float32)
    for nj in (1, -1):
        m.set_params(n_jobs=nj)
        t = time.perf_counter()
        Xadv = atk.hsj(m, X0, Xtr, thr, feats, ben)
        s = time.perf_counter() - t
        succ = float((m.predict_proba(pd.DataFrame(Xadv, columns=feats))[:, 1] <= thr).mean())
        lines.append(f"  微型 HSJ（{len(X0)} 个样本）n_jobs={nj}：{s:.1f} 秒，成功率 {succ:.0%}")
        print(lines[-1], flush=True)
        flush()

flush("全部完成。")
print("全部完成，已写入 03_结果/F_xgb_timing.csv 与 F_xgb_timing.txt", flush=True)
os._exit(0)
