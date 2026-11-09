"""
19_aggregate_ack.py
汇总 ACK 敏感性分析（18 脚本）：与主实验（11 脚本 L_formal_*_RAW_s*.csv）逐配置对比。

用法（项目根目录）：python 04_代码/19_aggregate_ack.py
输出：03_结果/R_summary_ack.csv（各配置均值 ± 标准差与差值）
      并在屏幕打印 1% 误报率、代价 ≤1× 下的组合计数（差值阈值 0.01）与最大变化。
"""
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "03_结果"
KEY = ["源", "模型", "设定", "工作点", "代价上限"]


def load(pattern, ratio):
    fs = sorted(glob.glob(str(RES / pattern)))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f) for f in fs])
    d["ACK比例"] = ratio
    return d


def main():
    base = load("L_formal_*_RAW_s*.csv", 0.0)
    parts = [base]
    for r in ("0.5", "1"):
        q = load(f"Q_ack_*_r{r}_s*.csv", float(r))
        if q is not None:
            parts.append(q)
    d = pd.concat(parts, ignore_index=True)
    viol = d.loc[d["ACK比例"] > 0, "构造流违反率"].max() if (d["ACK比例"] > 0).any() else float("nan")
    g = d.groupby(KEY + ["ACK比例"]).agg(种子数=("种子", "nunique"), 分数型均值=("分数型成功率", "mean"),
                                       分数型标准差=("分数型成功率", "std"), 判决型均值=("判决型成功率", "mean"))
    w = g.reset_index().pivot_table(index=KEY, columns="ACK比例", values=["分数型均值", "分数型标准差", "种子数", "判决型均值"])
    w.columns = [f"{a}|r{b:g}" for a, b in w.columns]
    w = w.reset_index()
    for r in ("0.5", "1"):
        if f"分数型均值|r{r}" in w:
            w[f"分数型差值|r{r}"] = w[f"分数型均值|r{r}"] - w["分数型均值|r0"]
    out = RES / "R_summary_ack.csv"
    w.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"ACK 变体构造流最大违反率：{viol}（必须为 0）")
    sub = w[(w["工作点"] == "FPR 1.0%") & (w["代价上限"] == "≤1×")]
    for r in ("0.5", "1"):
        c = f"分数型差值|r{r}"
        if c not in sub:
            continue
        s = sub.dropna(subset=[c])
        dd = s[c]
        print(f"\nratio={r}（1% 误报率，≤1×，分数型；{len(s)} 个组合，种子数 {s[f'种子数|r{r}'].min():g}–{s[f'种子数|r{r}'].max():g}）")
        print(f"  成功率下降 ≥0.01：{int((dd <= -0.01).sum())}；持平：{int((dd.abs() < 0.01).sum())}；上升 ≥0.01：{int((dd >= 0.01).sum())}")
        print(f"  最大变化：{dd.abs().max():.3f}")
        print(s[["源", "模型", "设定", "分数型均值|r0", f"分数型均值|r{r}", c]].round(3).to_string(index=False))
        allops = w.dropna(subset=[c])
        print(f"  全部工作点与代价上限（{len(allops)} 格）：最大 |差值| {allops[c].abs().max():.3f}，"
              f"|差值| ≥0.05 的格数 {int((allops[c].abs() >= 0.05).sum())}")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
