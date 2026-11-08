"""
03_prepare_data.py
可行性检验 B 的数据准备：每个数据集分块随机抽样约 150 万条，去掉标识符，按时间顺序切分训练/测试。

用法（项目根目录）：python 04_代码/03_prepare_data.py
输出：01_数据/processed/<数据集>.pkl（pandas pickle），03_结果/C0_split_summary.csv
- 抽样：每个数据块按固定随机种子抽取，抽样比例 = 150 万 / 总行数（总行数取自检验 A）。
- 去掉：IPV4_SRC_ADDR、IPV4_DST_ADDR、L4_SRC_PORT、L4_DST_PORT、DNS_QUERY_ID（标识符，防泄漏）；
  FLOW_START/END_MILLISECONDS 只用于排序切分，不作为特征。
- 无穷值（如 SECOND_BYTES 在原始数据中出现 inf）与无法解析的值：先按列计数写入报告，再置为 0。
- 切分：样本按 FLOW_START_MILLISECONDS 排序，前 70% 为训练（其中最后 15% 为验证，用于选阈值），后 30% 为测试。
"""
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "01_数据" / "raw"
OUT = ROOT / "01_数据" / "processed"
RES = ROOT / "03_结果"
TARGET = 1_500_000
TOTAL_ROWS = {  # 来自检验 A（03_结果/A_constraint_check_*.csv 样本数合计）
    "NF-UNSW-NB15-v3": 2_365_424,
    "NF-CSE-CIC-IDS2018-v3": 20_115_529,
    "NF-ToN-IoT-v3": 27_520_260,
}
DROP_ID = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT", "DNS_QUERY_ID"]
TIME = ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"]


def load(name: str) -> pd.DataFrame:
    zp = RAW / f"{name}.zip"
    frac = min(1.0, TARGET / TOTAL_ROWS[name])
    parts, t0 = [], time.time()
    inf_count = {}
    with zipfile.ZipFile(zp) as z:
        cands = [i for i in z.infolist() if i.filename.endswith(".csv") and "NetFlow_v3_Features" not in i.filename]
        csv_name = max(cands, key=lambda i: i.file_size).filename
        with z.open(csv_name) as f:
            for i, d in enumerate(pd.read_csv(f, chunksize=500_000, low_memory=False)):
                d.columns = [c.strip() for c in d.columns]
                d = d.drop(columns=[c for c in DROP_ID if c in d])
                if frac < 1.0:
                    d = d.sample(frac=frac, random_state=1000 + i)
                num = [c for c in d.columns if c not in ("Attack",)]
                d[num] = d[num].apply(pd.to_numeric, errors="coerce")
                for c in num:
                    k = int(np.isinf(d[c]).sum())
                    if k:
                        inf_count[c] = inf_count.get(c, 0) + k
                nan_before = d[num].isna().sum()
                for c, k in nan_before[nan_before > 0].items():
                    inf_count[f"{c}（NaN）"] = inf_count.get(f"{c}（NaN）", 0) + int(k)
                d[num] = d[num].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
                d["FLOW_START_MILLISECONDS"] = d["FLOW_START_MILLISECONDS"].astype("float64")
                parts.append(d)
                print(f"  {name}: 已读 {(i + 1) * 500_000:,} 行，保留 {sum(len(p) for p in parts):,}（{time.time() - t0:.0f}s）", flush=True)
    d = pd.concat(parts, ignore_index=True)
    d["Label"] = d["Label"].astype("int8")
    return d, frac, inf_count


def split(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values("FLOW_START_MILLISECONDS", kind="mergesort").reset_index(drop=True)
    n = len(d)
    n_tr = int(n * 0.70)
    n_val = int(n_tr * 0.15)
    s = np.full(n, "test", dtype=object)
    s[:n_tr - n_val] = "train"
    s[n_tr - n_val:n_tr] = "val"
    d["split"] = s
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(exist_ok=True)
    rows = []
    for name in TOTAL_ROWS:
        d, frac, inf_count = load(name)
        rows.append({"数据集": name, "切分": "抽样中被置 0 的无穷值/缺失值（按列）", "行数": sum(inf_count.values()),
                     "起始时间": "; ".join(f"{k}={v}" for k, v in inf_count.items()) or "无"})
        print(f"  {name}: 抽样中无穷值/缺失值计数 {inf_count or '无'}", flush=True)
        d = split(d)
        d.drop(columns=["FLOW_END_MILLISECONDS"]).to_pickle(OUT / f"{name}.pkl")
        for sp in ("train", "val", "test"):
            m = d.split == sp
            rows.append({"数据集": name, "抽样比例": round(frac, 5), "切分": sp, "行数": int(m.sum()),
                         "攻击数": int(d.Label[m].sum()), "攻击占比": float(d.Label[m].mean()),
                         "攻击类型数": int(d.Attack[m & (d.Label == 1)].nunique()),
                         "起始时间": pd.to_datetime(d.FLOW_START_MILLISECONDS[m].min(), unit="ms").isoformat(),
                         "结束时间": pd.to_datetime(d.FLOW_START_MILLISECONDS[m].max(), unit="ms").isoformat()})
        tr_types = set(d.Attack[(d.split == "train") & (d.Label == 1)])
        te_types = set(d.Attack[(d.split == "test") & (d.Label == 1)])
        print(f"  {name}: 测试集中训练集未出现的攻击类型：{sorted(te_types - tr_types)}", flush=True)
        rows.append({"数据集": name, "切分": "仅测试集出现的攻击类型", "行数": len(te_types - tr_types),
                     "起始时间": "; ".join(sorted(te_types - tr_types))})
        del d
    rep = pd.DataFrame(rows)
    rep.to_csv(RES / "C0_split_summary.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(rep.to_string(index=False))


if __name__ == "__main__":
    main()
