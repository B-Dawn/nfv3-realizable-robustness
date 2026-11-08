"""
15_aggregate_at.py
汇总对抗训练对比（14 脚本）的多种子结果：均值 ± 标准差。

用法（项目根目录）：python 04_代码/15_aggregate_at.py
输入：03_结果/N_at_*_s*.csv
输出：03_结果/O_summary_at.csv
"""
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parent.parent / "03_结果"


def ms(x):
    return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}" if len(x) > 1 else f"{x.mean():.3f}"


def main():
    N = pd.concat([pd.read_csv(f) for f in sorted(RES.glob("N_at_*_s*.csv"))], ignore_index=True)
    print(f"构造流最大违反率：{N['构造流违反率'].max()}（必须为 0）")
    keys = ["源", "模型", "防御", "设定", "工作点", "代价上限"]
    rows = []
    for k, g in N.groupby(keys, sort=False):
        rows.append({**dict(zip(keys, k)), "种子数": g["种子"].nunique(), "AUROC": ms(g["AUROC"]),
                     "TPR@该工作点": ms(g[f"TPR|{k[4]}"]), "FPR@该工作点": ms(g[f"FPR|{k[4]}"]),
                     "判决型成功率": ms(g["判决型成功率"]), "分数型成功率": ms(g["分数型成功率"]),
                     "分数型均值": g["分数型成功率"].mean()})
    O = pd.DataFrame(rows)
    O.to_csv(RES / "O_summary_at.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 30)
    for wp in ("FPR 1.0%", "FPR 0.1%"):
        v = O[(O["代价上限"] == "≤1×") & (O["工作点"] == wp)]
        print(f"\n分数型攻击成功率（{wp}，代价 ≤1×）与该工作点 TPR：")
        print(v.pivot_table(index=["源", "模型", "防御"], columns="设定", values="分数型成功率",
                            aggfunc="first", sort=False).join(
            v.drop_duplicates(["源", "模型", "防御"]).set_index(["源", "模型", "防御"])[["TPR@该工作点", "AUROC"]]).to_string())
    few = O[O["种子数"] < 3]
    if len(few):
        print(f"\n注意：{len(few)} 个组合不足 3 个种子")
    print(f"\n已写入 {RES / 'O_summary_at.csv'}")


if __name__ == "__main__":
    main()
