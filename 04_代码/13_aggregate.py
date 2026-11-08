"""
13_aggregate.py
汇总多种子正式实验结果：均值 ± 标准差（论文表格的数据来源）。

用法（项目根目录）：python 04_代码/13_aggregate.py
输入：03_结果/K_train_*_s*.csv（检测性能）、03_结果/L_formal_*_s*.csv（攻击成功率）
输出：03_结果/M_summary_detection.csv、03_结果/M_summary_attack.csv
同时检查：构造流违反率必须全部为 0；每个组合的种子数（不足 3 个时标注）。
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "03_结果"


def fmt(g, col):
    return f"{g[col].mean():.3f} ± {g[col].std(ddof=1):.3f}" if len(g) > 1 else f"{g[col].mean():.3f}"


def main():
    K = pd.concat([pd.read_csv(f) for f in sorted(RES.glob("K_train_*_s*.csv"))], ignore_index=True)
    L = pd.concat([pd.read_csv(f) for f in sorted(RES.glob("L_formal_*_s*.csv"))], ignore_index=True)
    bad = L["构造流违反率"].max()
    print(f"构造流最大违反率：{bad}（必须为 0）")

    det_rows = []
    for (src, var, m), g in K.groupby(["源", "变体", "模型"], sort=False):
        det_rows.append({"源": src, "变体": var, "模型": m, "种子数": g["种子"].nunique(),
                         "AUROC": fmt(g, "AUROC"), "TPR@FPR0.1%": fmt(g, "TPR|FPR 0.1%"),
                         "TPR@FPR1%": fmt(g, "TPR|FPR 1.0%"), "TPR@0.5": fmt(g, "TPR|默认 0.5"),
                         "FPR@0.5": fmt(g, "FPR|默认 0.5")})
    D = pd.DataFrame(det_rows)
    D.to_csv(RES / "M_summary_detection.csv", index=False, encoding="utf-8-sig")

    att_rows = []
    keys = ["源", "变体", "模型", "设定", "工作点", "代价上限"]
    for k, g in L.groupby(keys, sort=False):
        att_rows.append({**dict(zip(keys, k)), "种子数": g["种子"].nunique(),
                         "判决型成功率": fmt(g, "判决型成功率"), "分数型成功率": fmt(g, "分数型成功率"),
                         "判决型均值": g["判决型成功率"].mean(), "分数型均值": g["分数型成功率"].mean(),
                         "分数型标准差": g["分数型成功率"].std(ddof=1) if len(g) > 1 else float("nan")})
    A = pd.DataFrame(att_rows)
    A.to_csv(RES / "M_summary_attack.csv", index=False, encoding="utf-8-sig")

    few = sorted(set(D.loc[D["种子数"] < 3, "源"]) | set(A.loc[A["种子数"] < 3, "源"]))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 40)
    print("\n检测性能（同数据集测试集）：")
    print(D.to_string(index=False))
    print("\n攻击成功率（分数型，代价 ≤1×）：")
    v = A[A["代价上限"] == "≤1×"].pivot_table(index=["源", "变体", "模型", "设定"], columns="工作点",
                                               values="分数型成功率", aggfunc="first", sort=False)
    print(v.to_string())
    if few:
        print(f"\n注意：以下数据集存在不足 3 个种子的组合：{few}")
    print(f"\n已写入 {RES / 'M_summary_detection.csv'} 与 {RES / 'M_summary_attack.csv'}")


if __name__ == "__main__":
    main()
