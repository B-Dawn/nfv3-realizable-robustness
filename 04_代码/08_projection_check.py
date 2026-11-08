"""
08_projection_check.py
约束投影算子的正确性检验（单元测试，用真实流）。

用法（项目根目录）：python 04_代码/08_projection_check.py [每数据集抽样条数，默认 50000]
输入：01_数据/processed/*.pkl
输出：03_结果/G_projection_check.csv；全部通过时打印"全部通过"，否则列出失败项并以退出码 1 结束。

检验项（每个数据集、良性与攻击混合抽样）：
  T1 恒等：P(x, x) == x（逐元素完全相等）
  T2 原流有效：check(x) 违反率 0%（复核检验 A）
  T3 随机动作（STRICT / LOOSE）重建的流：C 类与取值域约束违反率 0%
  T4 随机对抗扰动（log 空间加噪声）投影后：违反率 0%
  T5 不可改特征保持不变（STRICT 下 TTL、窗口、标志也不变）
  T6 只增特征不减少：IN_PKTS、IN_BYTES、FLOW_DURATION、DURATION_IN
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nfv3_constraints import check  # noqa: E402
from nfv3_projection import apply_actions, project, random_actions  # noqa: E402

ROOT = HERE.parent
PROC, RES = ROOT / "01_数据" / "processed", ROOT / "03_结果"
DATASETS = ["NF-UNSW-NB15-v3", "NF-CSE-CIC-IDS2018-v3", "NF-ToN-IoT-v3"]
NON_FEAT = {"Label", "Attack", "split", "FLOW_START_MILLISECONDS"}

IMMUTABLE = ["PROTOCOL", "L7_PROTO", "OUT_BYTES", "OUT_PKTS", "SERVER_TCP_FLAGS", "TCP_WIN_MAX_OUT", "DURATION_OUT",
             "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
             "DNS_QUERY_TYPE", "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE", "ICMP_TYPE", "ICMP_IPV4_TYPE",
             "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX", "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV"]
STRICT_ALSO = ["MIN_TTL", "MAX_TTL", "TCP_WIN_MAX_IN", "CLIENT_TCP_FLAGS", "TCP_FLAGS"]
MONOTONE = ["IN_PKTS", "IN_BYTES", "FLOW_DURATION_MILLISECONDS", "DURATION_IN"]


def viol(x):
    v = check(x)
    return float(np.mean(np.any(list(v.values()), axis=0))), {k: float(m.mean()) for k, m in v.items() if m.any()}


def run(name, X, rng):
    out = []
    add = lambda t, ok, detail="": out.append({"数据集": name, "检验": t, "通过": bool(ok), "说明": detail})
    P = project(X, X, "LOOSE")
    diff = int((P.to_numpy() != X.astype(np.float64).to_numpy()).sum())
    add("T1 恒等 P(x,x)=x（LOOSE）", diff == 0, f"不相等元素 {diff}")
    P = project(X, X, "STRICT")
    diff = int((P.to_numpy() != X.astype(np.float64).to_numpy()).sum())
    add("T1 恒等 P(x,x)=x（STRICT）", diff == 0, f"不相等元素 {diff}")
    r, d = viol(X)
    add("T2 原流违反率 0%", r == 0, f"{r:.4%} {d}")
    for st in ("STRICT", "LOOSE"):
        Y = apply_actions(X, random_actions(X, rng, st), st)
        r, d = viol(Y)
        add(f"T3 随机动作违反率 0%（{st}）", r == 0, f"{r:.4%} {d}")
        cols = IMMUTABLE + (STRICT_ALSO if st == "STRICT" else [])
        bad = [c for c in cols if c in X and not np.array_equal(Y[c].to_numpy(), X[c].astype(np.float64).to_numpy())]
        add(f"T5 不可改特征不变（{st}）", not bad, ",".join(bad))
        dec = [c for c in MONOTONE if (Y[c].to_numpy() < X[c].astype(np.float64).to_numpy() - 1e-9).any()]
        add(f"T6 只增特征不减少（{st}）", not dec, ",".join(dec))
        noisy = np.expm1(np.log1p(X.clip(lower=0).astype(np.float64)) + rng.normal(0, 1.0, X.shape))
        A = pd.DataFrame(noisy, columns=X.columns, index=X.index)
        r0, _ = viol(A)
        Y = project(X, A, st)
        r, d = viol(Y)
        add(f"T4 随机扰动投影后违反率 0%（{st}）", r == 0, f"投影前 {r0:.2%} → 投影后 {r:.4%} {d}")
        dec = [c for c in MONOTONE if (Y[c].to_numpy() < X[c].astype(np.float64).to_numpy() - 1e-9).any()]
        add(f"T6 投影后只增特征不减少（{st}）", not dec, ",".join(dec))
    return out


def main(n):
    rng = np.random.default_rng(0)
    rows = []
    for name in DATASETS:
        d = pd.read_pickle(PROC / f"{name}.pkl")
        X = d[[c for c in d.columns if c not in NON_FEAT]].sample(min(n, len(d)), random_state=0)
        rows += run(name, X, rng)
        print(f"  {name} 完成", flush=True)
    rep = pd.DataFrame(rows)
    RES.mkdir(exist_ok=True)
    rep.to_csv(RES / "G_projection_check.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    print(rep.to_string(index=False))
    if rep["通过"].all():
        print("\n全部通过")
    else:
        print(f"\n失败 {int((~rep['通过']).sum())} 项")
        sys.exit(1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50_000)
