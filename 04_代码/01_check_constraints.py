"""
01_check_constraints.py
最小可行性检验 A：NetFlow v3 协议约束在真实数据上的满足率。

用法（在 VSCode 终端，位于 新论文_A2_NIDS对抗鲁棒性 目录下）：
    python 04_代码/01_check_constraints.py 01_数据/raw/NF-UNSW-NB15-v3.zip

输出：
    03_结果/A_constraint_check_<数据集名>.csv   每条约束 × 类别 的违反数与违反率
    03_结果/A_constraint_check_<数据集名>.json  同上 + 运行元数据
说明：
- 直接从 zip 内分块读取 CSV，不解压，内存占用约 1 GB 以内。
- 每条约束都是“假设”，本脚本的目的就是检验它们在真实数据上是否成立；
  违反率高说明约束写错或数据本身有瑕疵，两种情况都需要人工复核，不能直接删除约束或数据。
- 派生特征（速率、吞吐量）用相对误差容差判断，容差写在 TOL 中。
"""
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

CHUNK = 500_000
TOL = {"rel": 0.05, "dur_ms": 2}  # 派生速率允许 5% 相对误差；时长允许 2 ms 差

BINS = ["NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES",
        "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES"]
BIN_LO = np.array([0, 129, 257, 513, 1025])
BIN_HI = np.array([128, 256, 512, 1024, 1514])


def rel_close(a, b, rel):
    a = a.astype(float)
    b = b.astype(float)
    return np.abs(a - b) <= rel * np.maximum(np.abs(a), np.abs(b)) + 1.0  # +1 吸收整数取整


def constraints(d: pd.DataFrame) -> dict:
    """返回 {约束名: 布尔 Series（True=违反）}。约束编号对应论文中的 C 编号草案。"""
    pkts = d.IN_PKTS + d.OUT_PKTS
    byts = d.IN_BYTES + d.OUT_BYTES
    bins = d[BINS].to_numpy()
    tcp = d.PROTOCOL == 6
    span = d.FLOW_END_MILLISECONDS - d.FLOW_START_MILLISECONDS
    v = {}
    # --- 包长分布与计数 ---
    v["C01 包长分段计数之和 == 总包数"] = bins.sum(1) != pkts
    v["C02 包长分段计数之和 <= 总包数（允许 >1514 的超长包）"] = bins.sum(1) > pkts
    v["C03 MIN_IP_PKT_LEN <= MAX_IP_PKT_LEN"] = (d.MIN_IP_PKT_LEN > d.MAX_IP_PKT_LEN) & (pkts > 0)
    v["C04 LONGEST_FLOW_PKT == MAX_IP_PKT_LEN"] = d.LONGEST_FLOW_PKT != d.MAX_IP_PKT_LEN
    v["C05 SHORTEST_FLOW_PKT == MIN_IP_PKT_LEN"] = d.SHORTEST_FLOW_PKT != d.MIN_IP_PKT_LEN
    v["C06 总字节 ∈ [总包数×最短包, 总包数×最长包]"] = (byts < pkts * d.MIN_IP_PKT_LEN) | (byts > pkts * d.MAX_IP_PKT_LEN)
    v["C07 总字节 ∈ 分段计数给出的上下界"] = (byts < bins @ BIN_LO) | (byts > bins @ BIN_HI)
    # --- 重传 ---
    v["C08 重传包数 <= 对应方向包数"] = (d.RETRANSMITTED_IN_PKTS > d.IN_PKTS) | (d.RETRANSMITTED_OUT_PKTS > d.OUT_PKTS)
    v["C09 重传字节 <= 对应方向字节"] = (d.RETRANSMITTED_IN_BYTES > d.IN_BYTES) | (d.RETRANSMITTED_OUT_BYTES > d.OUT_BYTES)
    # --- TCP 语义 ---
    v["C10 TCP_FLAGS == CLIENT | SERVER"] = d.TCP_FLAGS != (d.CLIENT_TCP_FLAGS | d.SERVER_TCP_FLAGS)
    v["C11 非 TCP 流的 TCP 标志全为 0"] = (~tcp) & ((d.TCP_FLAGS != 0) | (d.CLIENT_TCP_FLAGS != 0) | (d.SERVER_TCP_FLAGS != 0))
    v["C12 非 TCP 流的 TCP 窗口为 0"] = (~tcp) & ((d.TCP_WIN_MAX_IN != 0) | (d.TCP_WIN_MAX_OUT != 0))
    v["C13 非 TCP 流的重传为 0"] = (~tcp) & ((d.RETRANSMITTED_IN_PKTS != 0) | (d.RETRANSMITTED_OUT_PKTS != 0))
    # --- 其他字段 ---
    v["C14 MIN_TTL <= MAX_TTL"] = d.MIN_TTL > d.MAX_TTL
    v["C15 ICMP_TYPE // 256 == ICMP_IPV4_TYPE"] = (d.ICMP_TYPE // 256) != d.ICMP_IPV4_TYPE
    # --- 时间 ---
    v["C16 DURATION_IN/OUT <= FLOW_DURATION"] = (d.DURATION_IN > d.FLOW_DURATION_MILLISECONDS + TOL["dur_ms"]) | (d.DURATION_OUT > d.FLOW_DURATION_MILLISECONDS + TOL["dur_ms"])
    v["C17 END-START ≈ FLOW_DURATION"] = np.abs(span - d.FLOW_DURATION_MILLISECONDS) > TOL["dur_ms"]
    for side in ("SRC_TO_DST", "DST_TO_SRC"):
        mn, mx, av, sd = (d[f"{side}_IAT_{k}"] for k in ("MIN", "MAX", "AVG", "STDDEV"))
        v[f"C18 {side} IAT: MIN <= AVG <= MAX, STD >= 0"] = (mn > av) | (av > mx) | (sd < 0)
    # --- 派生速率（检验命名方向：两种假设都测，哪一种成立由数据决定）---
    dur = d.FLOW_DURATION_MILLISECONDS.where(d.FLOW_DURATION_MILLISECONDS > 0)
    ok = dur.notna()
    for name, num in (("IN_BYTES", d.IN_BYTES), ("OUT_BYTES", d.OUT_BYTES)):
        v[f"C19 SRC_TO_DST_SECOND_BYTES ≈ {name}/FLOW_DURATION(ms)"] = ok & ~rel_close(d.SRC_TO_DST_SECOND_BYTES, num / dur, TOL["rel"])
        v[f"C20 DST_TO_SRC_SECOND_BYTES ≈ {name}/FLOW_DURATION(ms)"] = ok & ~rel_close(d.DST_TO_SRC_SECOND_BYTES, num / dur, TOL["rel"])
    spn = span.where(span > 0)
    ok2 = spn.notna()
    for name, num in (("IN_BYTES", d.IN_BYTES), ("OUT_BYTES", d.OUT_BYTES)):
        v[f"C21 SRC_TO_DST_AVG_THROUGHPUT ≈ {name}×8/(END-START)s"] = ok2 & ~rel_close(d.SRC_TO_DST_AVG_THROUGHPUT, num * 8000 / spn, TOL["rel"])
        v[f"C22 DST_TO_SRC_AVG_THROUGHPUT ≈ {name}×8/(END-START)s"] = ok2 & ~rel_close(d.DST_TO_SRC_AVG_THROUGHPUT, num * 8000 / spn, TOL["rel"])
    # --- 基本取值 ---
    count_cols = ["IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS"] + BINS
    v["C23 计数类特征非负"] = (d[count_cols] < 0).any(axis=1)
    return {k: pd.Series(np.asarray(m), index=d.index) for k, m in v.items()}


def main(zip_path: str):
    zp = Path(zip_path)
    name = zp.stem
    out_dir = Path("03_结果")
    out_dir.mkdir(exist_ok=True)
    t0 = time.time()
    with zipfile.ZipFile(zp) as z:
        # 各数据集压缩包内 CSV 命名不统一（如 NF-CSE-CIC-IDS2018-v3.zip 内为 NF-CICIDS2018-v3.csv），
        # 因此取除特征说明表外最大的 CSV。
        cands = [i for i in z.infolist() if i.filename.endswith(".csv") and "NetFlow_v3_Features" not in i.filename]
        csv_name = max(cands, key=lambda i: i.file_size).filename
        print(f"读取 {csv_name}（未压缩 {max(c.file_size for c in cands) / 1e9:.2f} GB）", flush=True)
        with z.open(csv_name) as f:
            total = {}
            n_rows = 0
            for i, d in enumerate(pd.read_csv(f, chunksize=CHUNK, low_memory=False)):
                d.columns = [c.strip() for c in d.columns]
                grp = np.where(d.Label == 0, "良性", "攻击")
                for k, viol in constraints(d).items():
                    for g in ("良性", "攻击"):
                        sel = grp == g
                        key = (k, g)
                        a, b = total.get(key, (0, 0))
                        total[key] = (a + int(viol[sel].sum()), b + int(sel.sum()))
                n_rows += len(d)
                print(f"  已处理 {n_rows:,} 行（第 {i + 1} 块，{time.time() - t0:.0f}s）", flush=True)
    rows = []
    for (k, g), (nv, n) in sorted(total.items()):
        rows.append({"约束": k, "类别": g, "违反数": nv, "样本数": n, "违反率": nv / n if n else np.nan})
    rep = pd.DataFrame(rows)
    pivot = rep.pivot_table(index="约束", columns="类别", values="违反率").reindex(columns=["良性", "攻击"])
    csv_out = out_dir / f"A_constraint_check_{name}.csv"
    rep.to_csv(csv_out, index=False, encoding="utf-8-sig")
    meta = {"dataset": name, "zip": str(zp), "rows": n_rows, "seconds": round(time.time() - t0, 1),
            "tolerance": TOL, "results": rows}
    (out_dir / f"A_constraint_check_{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 70)
    print("\n违反率（0.0000 表示约束在该类别上完全成立）：")
    print(pivot.apply(lambda col: col.map(lambda x: f"{x:.4%}")).to_string())
    print(f"\n共 {n_rows:,} 行，用时 {time.time() - t0:.0f}s。结果已写入 {csv_out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("用法: python 04_代码/01_check_constraints.py <数据集 zip 路径>")
    main(sys.argv[1])
