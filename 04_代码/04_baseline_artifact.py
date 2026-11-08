"""
04_baseline_artifact.py
可行性检验 B-1：基线模型 + 伪迹对照（RAW / DROP / CLEAN），同数据集与跨数据集评估。

用法（项目根目录）：python 04_代码/04_baseline_artifact.py <源数据集名> [time|random]
  例：python 04_代码/04_baseline_artifact.py NF-UNSW-NB15-v3 random
  切分：random=按攻击类型分层随机（主实验）；time=时间顺序（补充：含训练未见攻击类型）。默认 random。
输入：01_数据/processed/*.pkl（03 脚本生成）
输出：03_结果/C_baseline_<源数据集>_<切分>.csv；05_模型/<源数据集>_<切分>_<模型>.joblib（仅 RAW 变体，供 05 攻击脚本使用）

变体：
  RAW   全部特征
  DROP  去掉无法修复的伪迹特征：5 个包长分段计数、2 个 SECOND_BYTES
  CLEAN DROP + 修复：MIN_IP_PKT_LEN=0 → SHORTEST_FLOW_PKT；IAT_MIN → min(IAT_MIN, IAT_AVG)
阈值：在源数据集验证集良性样本的 99% 分位（源 FPR=1%），原样用于各目标测试集。
单一随机种子（0），属可行性检验；正式实验需多种子。
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nfv3_constraints import apply_split, log1p_clip  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "01_数据" / "processed"
RES = ROOT / "03_结果"
MOD = ROOT / "05_模型"
DATASETS = ["NF-UNSW-NB15-v3", "NF-CSE-CIC-IDS2018-v3", "NF-ToN-IoT-v3"]
NON_FEAT = {"Label", "Attack", "split", "FLOW_START_MILLISECONDS"}
BINS = ["NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES",
        "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES"]
SECOND = ["SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES"]
N_TRAIN, N_TEST, SEED = 300_000, 300_000, 0


def variant(d: pd.DataFrame, v: str) -> pd.DataFrame:
    X = d[[c for c in d.columns if c not in NON_FEAT]].copy()
    if v in ("DROP", "CLEAN"):
        X = X.drop(columns=[c for c in BINS + SECOND if c in X])
    if v == "CLEAN":
        m = X.MIN_IP_PKT_LEN == 0
        X.loc[m, "MIN_IP_PKT_LEN"] = X.loc[m, "SHORTEST_FLOW_PKT"]
        for s in ("SRC_TO_DST", "DST_TO_SRC"):
            X[f"{s}_IAT_MIN"] = np.minimum(X[f"{s}_IAT_MIN"], X[f"{s}_IAT_AVG"])
    return X


def models():
    log = FunctionTransformer(log1p_clip)
    return {
        "RF": RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=SEED),
        "XGB": XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, tree_method="hist",
                             n_jobs=-1, random_state=SEED, eval_metric="logloss"),
        "MLP": make_pipeline(log, StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=30,
                                           random_state=SEED)),
    }


def cap(d, n, seed):
    """按类别比例分层抽样至 n 行（不依赖 groupby.apply，兼容 pandas 3）。"""
    if len(d) <= n:
        return d
    parts = []
    for lab in sorted(d.Label.unique()):
        g = d[d.Label == lab]
        k = max(1, int(round(n * len(g) / len(d))))
        parts.append(g.sample(n=min(k, len(g)), random_state=seed))
    return pd.concat(parts).sort_index()


def main(src: str, mode: str):
    RES.mkdir(exist_ok=True)
    MOD.mkdir(exist_ok=True)
    t0 = time.time()
    data = {n: apply_split(pd.read_pickle(PROC / f"{n}.pkl"), mode) for n in DATASETS}
    tests = {n: cap(d[d.split == "test"], N_TEST, SEED) for n, d in data.items()}
    s = data[src]
    tr = cap(s[s.split == "train"], N_TRAIN, SEED)
    va = s[s.split == "val"]
    rows = []
    for v in ("RAW", "DROP", "CLEAN"):
        Xtr, Xva = variant(tr, v), variant(va, v)
        for mname, model in models().items():
            t1 = time.time()
            model.fit(Xtr, tr.Label)
            sva = model.predict_proba(Xva)[:, 1]
            ben = sva[va.Label.to_numpy() == 0]
            thr = float(np.quantile(ben, 0.99)) if len(ben) >= 100 else 0.5
            if v == "RAW":
                joblib.dump({"model": model, "features": list(Xtr.columns), "threshold": thr}, MOD / f"{src}_{mode}_{mname}.joblib")
            for tgt, te in tests.items():
                y = te.Label.to_numpy()
                sc = model.predict_proba(variant(te, v))[:, 1]
                pred = sc > thr
                b = sc[y == 0]
                thr_o = np.quantile(b, 0.99) if len(b) else np.nan
                rows.append({
                    "源": src, "切分": mode, "目标": tgt, "同数据集": src == tgt, "变体": v, "模型": mname,
                    "AUROC": roc_auc_score(y, sc) if 0 < y.mean() < 1 else np.nan,
                    "TPR@源阈值": float(pred[y == 1].mean()) if (y == 1).any() else np.nan,
                    "FPR@源阈值": float(pred[y == 0].mean()) if (y == 0).any() else np.nan,
                    "TPR@目标FPR1%(oracle)": float((sc[y == 1] > thr_o).mean()) if (y == 1).any() else np.nan,
                    "F1@源阈值": f1_score(y, pred, zero_division=0),
                    "源阈值": thr, "验证集良性数": int(len(ben)), "测试数": len(te), "测试攻击占比": float(y.mean()),
                    "特征数": Xtr.shape[1], "训练秒": round(time.time() - t1, 1),
                })
            print(f"  {src} {v} {mname} 完成（{time.time() - t0:.0f}s）", flush=True)
    rep = pd.DataFrame(rows)
    out = RES / f"C_baseline_{src}_{mode}.csv"
    rep.to_csv(out, index=False, encoding="utf-8-sig")
    (RES / f"C_baseline_{src}_{mode}.json").write_text(json.dumps(
        {"src": src, "split": mode, "n_train": len(tr), "n_val": len(va), "seed": SEED, "seconds": round(time.time() - t0, 1)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(rep[["目标", "变体", "模型", "AUROC", "TPR@源阈值", "FPR@源阈值", "TPR@目标FPR1%(oracle)"]].round(4).to_string(index=False))
    print(f"\n已写入 {out}，用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3) or sys.argv[1] not in DATASETS or (len(sys.argv) == 3 and sys.argv[2] not in ("time", "random")):
        sys.exit(f"用法: python 04_代码/04_baseline_artifact.py <{'|'.join(DATASETS)}> [time|random]")
    main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else "random")
