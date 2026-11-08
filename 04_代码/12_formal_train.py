"""
12_formal_train.py
正式实验：按随机种子训练 RF / XGB / MLP（与 04 脚本相同的模型与超参数），供 11 脚本多种子攻击使用。

用法（项目根目录）：python 04_代码/12_formal_train.py <源数据集名> --seed 0 [--variant RAW|noTTL]
  种子同时决定：数据切分（apply_split 的 seed）、训练集抽样、模型初始化。
  RAW   = 全部 46 个特征（与 04 脚本 RAW 相同）
  noTTL = 去掉 MIN_TTL、MAX_TTL（检验 TTL 泄漏）
输出：05_模型/<源>_<variant>_<模型>_s<seed>.joblib
      03_结果/K_train_<源>_<variant>_s<seed>.csv（同数据集测试集：AUROC、各工作点阈值下的 TPR/FPR）
已有同名模型文件时跳过训练（便于中断后续跑），但仍重新评估。
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
from nfv3_constraints import apply_split  # noqa: E402

_spec = importlib.util.spec_from_file_location("b04", HERE / "04_baseline_artifact.py")
b04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b04)

ROOT = HERE.parent
PROC, RES, MOD = ROOT / "01_数据" / "processed", ROOT / "03_结果", ROOT / "05_模型"
DATASETS = ["NF-UNSW-NB15-v3", "NF-CSE-CIC-IDS2018-v3", "NF-ToN-IoT-v3"]
NON_FEAT = {"Label", "Attack", "split", "FLOW_START_MILLISECONDS"}
TTL = ["MIN_TTL", "MAX_TTL"]
FPRS = [0.001, 0.01, 0.05]


def models(seed):
    ms = b04.models()
    for m in ms.values():
        est = m.steps[-1][1] if hasattr(m, "steps") else m
        est.set_params(random_state=seed)
    return ms


def main(src, seed, variant):
    t0 = time.time()
    MOD.mkdir(exist_ok=True)
    d = apply_split(pd.read_pickle(PROC / f"{src}.pkl"), "random", seed=seed)
    feats = [c for c in d.columns if c not in NON_FEAT and not (variant == "noTTL" and c in TTL)]
    tr = b04.cap(d[d.split == "train"], b04.N_TRAIN, seed)
    va = d[d.split == "val"]
    te = b04.cap(d[d.split == "test"], b04.N_TEST, seed)
    yv, yt = va.Label.to_numpy(), te.Label.to_numpy()
    rows = []
    for mname, model in models(seed).items():
        path = MOD / f"{src}_{variant}_{mname}_s{seed}.joblib"
        t1 = time.time()
        if path.exists():
            b = joblib.load(path)
            model = b["model"]
            status = "已存在，跳过训练"
        else:
            model.fit(tr[feats], tr.Label)
            status = "新训练"
        sv = model.predict_proba(va[feats])[:, 1]
        ben = sv[yv == 0]
        thrs = {f"FPR {f:.1%}": float(np.quantile(ben, 1 - f)) for f in FPRS}
        thrs["默认 0.5"] = 0.5
        if status == "新训练":
            joblib.dump({"model": model, "features": feats, "threshold": thrs["FPR 1.0%"], "thresholds": thrs,
                         "seed": seed, "variant": variant}, path)
        st = model.predict_proba(te[feats])[:, 1]
        row = {"源": src, "变体": variant, "种子": seed, "模型": mname, "特征数": len(feats),
               "AUROC": roc_auc_score(yt, st), "状态": status, "秒": round(time.time() - t1, 1)}
        for k, thr in thrs.items():
            pred = st > thr
            row[f"阈值|{k}"] = thr
            row[f"TPR|{k}"] = float(pred[yt == 1].mean())
            row[f"FPR|{k}"] = float(pred[yt == 0].mean())
        rows.append(row)
        print(f"  {src} {variant} s{seed} {mname}: AUROC {row['AUROC']:.4f}，TPR@FPR1% {row['TPR|FPR 1.0%']:.4f}（{status}，{row['秒']}s）", flush=True)
    rep = pd.DataFrame(rows)
    out = RES / f"K_train_{src}_{variant}_s{seed}.csv"
    rep.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"已写入 {out}，用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", choices=DATASETS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="RAW", choices=["RAW", "noTTL"])
    a = ap.parse_args()
    main(a.src, a.seed, a.variant)
