"""
06_feature_flip_audit.py
单特征方向审计：对每个特征，计算它单独作为“攻击分数”时在三个数据集上的 AUROC。
  AUROC > 0.5：该特征越大越像攻击；< 0.5：越大越像良性；≈ 0.5：无区分力。
方向反转特征 = 在一个数据集中 > 0.5+δ、在另一个数据集中 < 0.5−δ（δ=0.1）。
这类特征会让跨数据集模型的排序系统性颠倒（检验 B-1 中出现了 AUROC < 0.5 的跨数据集结果）。

用法（项目根目录）：python 04_代码/06_feature_flip_audit.py
输入：01_数据/processed/*.pkl（只用训练部分：random 切分的 train，避免触碰测试集）
输出：03_结果/E_feature_flip.csv（每特征 × 数据集的单特征 AUROC、方向、是否反转）
另输出缺失为 0 的占比（MIN_IP_PKT_LEN==0 等）与 TTL 取值的类别分布摘要，用于检验 UNSW 的 TTL 泄漏假设。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nfv3_constraints import apply_split  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "01_数据" / "processed"
RES = ROOT / "03_结果"
DATASETS = {"NF-UNSW-NB15-v3": "UNSW", "NF-CSE-CIC-IDS2018-v3": "CIC", "NF-ToN-IoT-v3": "ToN"}
NON_FEAT = {"Label", "Attack", "split", "FLOW_START_MILLISECONDS"}
DELTA = 0.1


def main():
    au, ttl_lines = {}, []
    for name, short in DATASETS.items():
        d = apply_split(pd.read_pickle(PROC / f"{name}.pkl"), "random")
        d = d[d.split == "train"]
        y = d.Label.to_numpy()
        feats = [c for c in d.columns if c not in NON_FEAT]
        au[short] = {f: (roc_auc_score(y, d[f].to_numpy()) if d[f].nunique() > 1 else 0.5) for f in feats}
        for f in ("MIN_TTL", "MAX_TTL"):
            for lab, g in d.groupby("Label"):
                top = g[f].value_counts(normalize=True).head(5)
                ttl_lines.append(f"{short} {f} [{'攻击' if lab else '良性'}] 前 5 取值：" +
                                 "，".join(f"{int(k)}={v:.1%}" for k, v in top.items()))
        print(f"  {short}: {len(feats)} 个特征，训练样本 {len(d):,}", flush=True)
    A = pd.DataFrame(au)
    hi, lo = A > 0.5 + DELTA, A < 0.5 - DELTA
    A["方向反转"] = (hi.any(axis=1) & lo.any(axis=1))
    A["最大偏离"] = (A[list(DATASETS.values())] - 0.5).abs().max(axis=1)
    A = A.sort_values(["方向反转", "最大偏离"], ascending=False)
    A.index.name = "特征"
    out = RES / "E_feature_flip.csv"
    A.round(4).to_csv(out, encoding="utf-8-sig")
    (RES / "E_ttl_summary.txt").write_text("\n".join(ttl_lines), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(A.round(3).to_string())
    print(f"\n方向反转特征数：{int(A['方向反转'].sum())} / {len(A)}")
    print("\n".join(ttl_lines))
    print(f"\n已写入 {out} 与 {RES / 'E_ttl_summary.txt'}")


if __name__ == "__main__":
    main()
