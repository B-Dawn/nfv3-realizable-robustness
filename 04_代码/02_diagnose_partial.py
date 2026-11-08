"""
02_diagnose_partial.py
诊断 01 中“部分成立”的约束：C01（分段计数 vs 总包数）、C05（最短包 vs 最小 IP 包长）、C21/C22（吞吐量公式）。

用法：python 02_diagnose_partial.py <数据集 zip 路径> [抽样比例，默认 1.0]
  大数据集（CIC、ToN 约 2000 万行）建议 0.1：每个数据块按固定随机种子抽 10%，可复现。
输出：项目根目录/03_结果/B_diagnose_<数据集名>.txt
只读数据，不修改任何文件。
"""
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BINS = ["NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES",
        "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES"]
USE = ["PROTOCOL", "L7_PROTO", "IN_PKTS", "OUT_PKTS", "IN_BYTES", "OUT_BYTES", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
       "SHORTEST_FLOW_PKT", "LONGEST_FLOW_PKT", "FLOW_DURATION_MILLISECONDS", "FLOW_START_MILLISECONDS",
       "FLOW_END_MILLISECONDS", "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT", "Label", "Attack",
       "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES"] + BINS + \
      [f"{s}_IAT_{k}" for s in ("SRC_TO_DST", "DST_TO_SRC") for k in ("MIN", "MAX", "AVG", "STDDEV")]


def vc(s, n=12):
    t = s.value_counts().head(n)
    return "\n".join(f"    {k!s:>14}: {v:>10,}  ({v / len(s):.2%})" for k, v in t.items()) if len(s) else "    （无）"


def main(zip_arg, frac=1.0):
    zp = Path(zip_arg)
    if not zp.is_absolute():
        zp = ROOT / zp
    name = zp.stem
    t0 = time.time()
    with zipfile.ZipFile(zp) as z:
        cands = [i for i in z.infolist() if i.filename.endswith(".csv") and "NetFlow_v3_Features" not in i.filename]
        csv_name = max(cands, key=lambda i: i.file_size).filename
        parts = []
        with z.open(csv_name) as f:
            for d in pd.read_csv(f, chunksize=500_000, usecols=lambda c: c.strip() in USE, low_memory=False):
                d.columns = [c.strip() for c in d.columns]
                if frac < 1.0:
                    d = d.sample(frac=frac, random_state=len(parts))
                parts.append(d)
                print(f"  读取 {sum(len(p) for p in parts):,} 行", flush=True)
    d = pd.concat(parts, ignore_index=True)
    d["类别"] = np.where(d.Label == 0, "良性", "攻击")
    pkts = d.IN_PKTS + d.OUT_PKTS
    bsum = d[BINS].sum(axis=1)
    span = d.FLOW_END_MILLISECONDS - d.FLOW_START_MILLISECONDS
    out = [f"数据集：{name}   分析行数：{len(d):,}（抽样比例 {frac}）", ""]

    # ---- C01 ----
    diff = bsum - pkts
    out += ["== C01 分段计数之和 − 总包数 ==",
            f"  不相等比例：{(diff != 0).mean():.2%}；其中分段计数更多的比例：{(diff[diff != 0] > 0).mean():.2%}"]
    for g in ("良性", "攻击"):
        m = (d.类别 == g) & (diff != 0)
        out += [f"  [{g}] 差值分布（前 12）：", vc(diff[m]),
                f"  [{g}] 差值/总包数 的分位数：{np.round(np.quantile((diff[m] / pkts[m].clip(lower=1)), [.1, .5, .9]), 3).tolist() if m.any() else '无'}",
                f"  [{g}] 不相等流的协议分布：", vc(d.PROTOCOL[m], 6)]
    eq_ratio = (bsum[diff != 0] == 2 * pkts[diff != 0]).mean() if (diff != 0).any() else float("nan")
    out += [f"  不相等的流中，分段计数之和恰好 = 2×总包数 的比例：{eq_ratio:.2%}",
            f"  不相等的流中，分段计数之和 = 总包数 + 重叠? 用 IN_PKTS 或 OUT_PKTS 单向解释：= IN_PKTS 占 {(bsum[diff != 0] == d.IN_PKTS[diff != 0]).mean():.2%}，= OUT_PKTS 占 {(bsum[diff != 0] == d.OUT_PKTS[diff != 0]).mean():.2%}", ""]

    # ---- C05 ----
    m5 = d.SHORTEST_FLOW_PKT != d.MIN_IP_PKT_LEN
    rel = np.sign(d.SHORTEST_FLOW_PKT - d.MIN_IP_PKT_LEN)
    out += ["== C05 SHORTEST_FLOW_PKT vs MIN_IP_PKT_LEN ==",
            f"  不相等比例：{m5.mean():.2%}；SHORTEST > MIN 的占比（不相等中）：{(rel[m5] > 0).mean():.2%}"]
    for g in ("良性", "攻击"):
        m = (d.类别 == g) & m5
        out += [f"  [{g}] MIN_IP_PKT_LEN 取值（前 8）：", vc(d.MIN_IP_PKT_LEN[m], 8),
                f"  [{g}] SHORTEST_FLOW_PKT 取值（前 8）：", vc(d.SHORTEST_FLOW_PKT[m], 8),
                f"  [{g}] 攻击类型分布：" if g == "攻击" else f"  [{g}] 协议分布：",
                vc(d.Attack[m] if g == "攻击" else d.PROTOCOL[m], 10)]
    out.append("")

    # ---- C21/C22 ----
    ok = (span > 0) & (d.SRC_TO_DST_AVG_THROUGHPUT > 0)
    implied = (d.IN_BYTES * 8000 / d.SRC_TO_DST_AVG_THROUGHPUT).where(ok)
    r_span = (implied / span.where(span > 0))
    r_dur = (implied / d.FLOW_DURATION_MILLISECONDS.where(d.FLOW_DURATION_MILLISECONDS > 0))
    out += ["== C21 吞吐量隐含分母：IN_BYTES×8000 / SRC_TO_DST_AVG_THROUGHPUT（单位 ms）=="]
    for g in ("良性", "攻击"):
        m = (d.类别 == g) & ok
        bad = m & ((r_span - 1).abs() > 0.05)
        out += [f"  [{g}] 可计算 {m.sum():,} 条，偏离 END-START 超过 5% 的 {bad.sum():,} 条",
                f"  [{g}] 偏离流：隐含分母 / (END-START) 分位数：{np.round(np.nanquantile(r_span[bad], [.1, .25, .5, .75, .9]), 3).tolist() if bad.any() else '无'}",
                f"  [{g}] 偏离流：隐含分母 / FLOW_DURATION 分位数：{np.round(np.nanquantile(r_dur[bad], [.1, .25, .5, .75, .9]), 3).tolist() if bad.any() else '无'}",
                f"  [{g}] 偏离流：隐含分母 − (END-START)（ms）分位数：{np.round(np.nanquantile((implied - span)[bad], [.1, .5, .9]), 1).tolist() if bad.any() else '无'}",
                f"  [{g}] 偏离流的 END-START（ms）分位数：{np.round(np.nanquantile(span[bad], [.1, .5, .9]), 0).tolist() if bad.any() else '无'}",
                f"  [{g}] 偏离流协议分布：", vc(d.PROTOCOL[bad], 6)]
    # ---- C19/C20：SECOND_BYTES 的分子方向与分母单位 ----
    out += ["", "== C19/C20 SECOND_BYTES：各候选公式的吻合比例（相对误差 ≤5% 或绝对误差 ≤1）=="]
    dur = d.FLOW_DURATION_MILLISECONDS.astype(float)
    dens = {"FLOW_DURATION(ms)": dur, "END-START(ms)": span.astype(float), "END-START+1(ms)": span.astype(float) + 1,
            "FLOW_DURATION(s)": dur / 1000, "DURATION_IN/OUT 未取": None}
    for field in ("SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES"):
        y = d[field].astype(float)
        for g in ("良性", "攻击"):
            gm = d.类别 == g
            line = [f"  [{g}] {field}："]
            for num_name in ("IN_BYTES", "OUT_BYTES"):
                for den_name, den in dens.items():
                    if den is None:
                        continue
                    valid = gm & (den > 0)
                    pred = d[num_name] / den.where(den > 0)
                    hit = (np.abs(y - pred) <= 0.05 * np.maximum(np.abs(y), np.abs(pred)) + 1.0) & valid
                    line.append(f"      {num_name}/{den_name}: {hit.sum() / max(valid.sum(), 1):.2%}")
            zero = gm & (dur == 0)
            line.append(f"      FLOW_DURATION=0 的流占该类 {zero.sum() / max(gm.sum(), 1):.2%}，其中字段值为 0 的比例 {(y[zero] == 0).mean() if zero.any() else float('nan'):.2%}")
            out += line
    # ---- C18：包间隔违反的具体形式 ----
    out += ["", "== C18 包间隔：违反形式与攻击类型 =="]
    for side in ("SRC_TO_DST", "DST_TO_SRC"):
        mn, mx, av, sd = (d[f"{side}_IAT_{k}"] for k in ("MIN", "MAX", "AVG", "STDDEV"))
        forms = {"MIN>AVG": mn > av, "AVG>MAX": av > mx, "STD<0": sd < 0, "MIN>MAX": mn > mx}
        anyv = forms["MIN>AVG"] | forms["AVG>MAX"] | forms["STD<0"]
        for g in ("良性", "攻击"):
            gm = d.类别 == g
            out.append(f"  [{side}][{g}] 违反率 {(anyv & gm).sum() / max(gm.sum(), 1):.2%}；形式：" +
                       "，".join(f"{k} {(v & gm).sum() / max((anyv & gm).sum(), 1):.1%}" for k, v in forms.items()))
            if g == "攻击" and (anyv & gm).any():
                out += [f"  [{side}] 违反流的攻击类型：", vc(d.Attack[anyv & gm], 8)]
            pk = d.IN_PKTS if side == "SRC_TO_DST" else d.OUT_PKTS
            if (anyv & gm).any():
                out.append(f"  [{side}][{g}] 违反流的该方向包数分位数：{np.quantile(pk[anyv & gm], [.1, .5, .9]).tolist()}")
    out.append(f"\n用时 {time.time() - t0:.0f}s")
    res = ROOT / "03_结果"
    res.mkdir(exist_ok=True)
    p = res / f"B_diagnose_{name}.txt"
    p.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"\n已写入 {p}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit("用法: python 02_diagnose_partial.py <数据集 zip 路径> [抽样比例]")
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) == 3 else 1.0)
