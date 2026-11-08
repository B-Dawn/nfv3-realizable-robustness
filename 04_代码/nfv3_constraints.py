"""
nfv3_constraints.py
NetFlow v3 协议约束与取值域检查（供 04、05 脚本共用）。

只收录可行性检验 A 在三个数据集（50,001,213 条流）上违反率为 0% 的精确约束，
另加取值域约束（计数类为非负整数、协议号与标志位为合法整数）。
依据：03_结果/A_constraint_check_*.csv、03_结果/00_发现记录.md。
"""
import numpy as np
import pandas as pd

# 计数类：真实数据中必为非负整数
INT_FEATURES = [
    "PROTOCOL", "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS",
    "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS", "DURATION_IN", "DURATION_OUT",
    "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
    "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE",
]
BYTE_FLAG_FEATURES = ["TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS"]  # 0–255
TTL_FEATURES = ["MIN_TTL", "MAX_TTL"]  # 0–255


def _col(d, c):
    return d[c].to_numpy(dtype=float) if c in d else None


def check(d: pd.DataFrame, tol_ms: float = 2.0) -> dict:
    """返回 {约束名: 布尔数组（True=违反）}。缺列的约束自动跳过。"""
    v = {}
    g = lambda c: _col(d, c)
    has = lambda *cs: all(c in d for c in cs)
    # 取值域
    ints = [c for c in INT_FEATURES if c in d]
    if ints:
        X = d[ints].to_numpy(dtype=float)
        v["D1 计数类特征为非负整数"] = ((X < 0) | (np.abs(X - np.round(X)) > 1e-6)).any(axis=1)
    fl = [c for c in BYTE_FLAG_FEATURES + TTL_FEATURES if c in d]
    if fl:
        v["D2 TCP 标志与 TTL 位于 0–255"] = ((d[fl] < 0) | (d[fl] > 255)).any(axis=1).to_numpy()
    # 精确协议约束（检验 A：三数据集 0% 违反）
    if has("MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN"):
        v["C03 MIN_IP_PKT_LEN ≤ MAX_IP_PKT_LEN"] = g("MIN_IP_PKT_LEN") > g("MAX_IP_PKT_LEN")
    if has("LONGEST_FLOW_PKT", "MAX_IP_PKT_LEN"):
        v["C04 LONGEST_FLOW_PKT = MAX_IP_PKT_LEN"] = np.abs(g("LONGEST_FLOW_PKT") - g("MAX_IP_PKT_LEN")) > 1e-6
    if has("RETRANSMITTED_IN_PKTS", "IN_PKTS", "RETRANSMITTED_OUT_PKTS", "OUT_PKTS"):
        v["C08 重传包数 ≤ 对应方向包数"] = (g("RETRANSMITTED_IN_PKTS") > g("IN_PKTS")) | (g("RETRANSMITTED_OUT_PKTS") > g("OUT_PKTS"))
    if has("RETRANSMITTED_IN_BYTES", "IN_BYTES", "RETRANSMITTED_OUT_BYTES", "OUT_BYTES"):
        v["C09 重传字节 ≤ 对应方向字节"] = (g("RETRANSMITTED_IN_BYTES") > g("IN_BYTES")) | (g("RETRANSMITTED_OUT_BYTES") > g("OUT_BYTES"))
    if has("TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS"):
        tf, cf, sf = (np.round(g(c)).astype(np.int64) for c in ("TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS"))
        v["C10 TCP_FLAGS = CLIENT | SERVER"] = tf != (cf | sf)
    if "PROTOCOL" in d:
        non_tcp = np.round(g("PROTOCOL")) != 6
        if has("TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS"):
            v["C11 非 TCP 流 TCP 标志为 0"] = non_tcp & ((g("TCP_FLAGS") != 0) | (g("CLIENT_TCP_FLAGS") != 0) | (g("SERVER_TCP_FLAGS") != 0))
        if has("TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT"):
            v["C12 非 TCP 流 TCP 窗口为 0"] = non_tcp & ((g("TCP_WIN_MAX_IN") != 0) | (g("TCP_WIN_MAX_OUT") != 0))
        if has("RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_PKTS"):
            v["C13 非 TCP 流重传为 0"] = non_tcp & ((g("RETRANSMITTED_IN_PKTS") != 0) | (g("RETRANSMITTED_OUT_PKTS") != 0))
    if has("MIN_TTL", "MAX_TTL"):
        v["C14 MIN_TTL ≤ MAX_TTL"] = g("MIN_TTL") > g("MAX_TTL")
    if has("ICMP_TYPE", "ICMP_IPV4_TYPE"):
        v["C15 ICMP_TYPE // 256 = ICMP_IPV4_TYPE"] = np.floor(g("ICMP_TYPE") / 256) != np.round(g("ICMP_IPV4_TYPE"))
    if has("DURATION_IN", "DURATION_OUT", "FLOW_DURATION_MILLISECONDS"):
        fd = g("FLOW_DURATION_MILLISECONDS")
        v["C16 DURATION_IN/OUT ≤ FLOW_DURATION"] = (g("DURATION_IN") > fd + tol_ms) | (g("DURATION_OUT") > fd + tol_ms)
    cnt = [c for c in ("IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS") if c in d]
    if cnt:
        v["C23 计数类非负"] = (d[cnt] < 0).any(axis=1).to_numpy()
    return v


def summary(d: pd.DataFrame) -> dict:
    """返回 {约束名: 违反率}，并附 'ANY 协议约束（C 类）' 与 'ANY 全部' 两项汇总。"""
    v = check(d)
    out = {k: float(np.mean(m)) for k, m in v.items()}
    c_masks = [m for k, m in v.items() if k.startswith("C")]
    out["ANY 协议约束（C 类）"] = float(np.mean(np.any(c_masks, axis=0))) if c_masks else float("nan")
    out["ANY 全部（C 类 + 取值域 D 类）"] = float(np.mean(np.any(list(v.values()), axis=0))) if v else float("nan")
    return out


def log1p_clip(X):
    """MLP 预处理：截断负值后取 log1p。放在本模块中，使保存的模型可在其他脚本中加载。"""
    return np.log1p(np.clip(np.asarray(X, dtype=np.float64), 0, None))


def apply_split(d, mode: str, seed: int = 0):
    """
    返回带 'split' 列的数据副本。
    mode="time"   ：沿用 03 脚本的时间顺序切分（前 70% 训练，其中末 15% 为验证；后 30% 测试）。
    mode="random" ：按攻击类型（Attack 列，含 Benign）分层随机切分，各类别 59.5% 训练 / 10.5% 验证 / 30% 测试，
                    与时间切分的比例一致；固定随机种子，04 与 05 脚本调用结果完全相同。
    """
    if mode == "time":
        return d
    if mode != "random":
        raise ValueError("mode 只能是 time 或 random")
    d = d.copy()
    rng = np.random.default_rng(seed)
    s = np.empty(len(d), dtype=object)
    for _, idx in d.groupby("Attack", sort=True).indices.items():
        idx = rng.permutation(idx)
        n = len(idx)
        n_tr, n_va = int(round(n * 0.595)), int(round(n * 0.105))
        s[idx[:n_tr]] = "train"
        s[idx[n_tr:n_tr + n_va]] = "val"
        s[idx[n_tr + n_va:]] = "test"
    d["split"] = s
    return d
