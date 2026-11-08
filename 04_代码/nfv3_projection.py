"""
nfv3_projection.py
攻击者动作空间与约束投影（NetFlow v3，46 个特征）。

思路：不逐个特征"拉回合法范围"，而是定义攻击者真实能做的动作，
再由"原流 + 动作"重建新流。这样构造出来的流在物理上可实现，且满足检验 A 验证过的全部 C 类约束。

动作（只作用于攻击者发出的正向流量，即 src→dst = IN_*；假设攻击者是流的发起方）：
  A1 追加 dp 个正向包，共 db 字节（填充/冗余包，单包大小 s = db/dp，限制在 [包头, 1500]）
  A2 放慢发送：流时长与正向时长增加 dt 毫秒
  A3 设置初始 TTL：MIN_TTL、MAX_TTL ∈ [1, 255]，MIN ≤ MAX                      （仅宽松设定）
  A4 设置客户端 TCP 窗口：TCP_WIN_MAX_IN ∈ [0, 65535]，仅 TCP 流              （仅宽松设定）
  A5 追加客户端 TCP 标志（只能按位或，不能去掉原有标志），仅 TCP 流            （仅宽松设定）
设定：STRICT = A1 + A2；LOOSE = A1–A5。

派生特征的更新方式：
  - 有精确关系的直接重算：TCP_FLAGS = CLIENT | SERVER；MAX_IP_PKT_LEN = LONGEST_FLOW_PKT；包长分段计数加上新包。
  - 吞吐量、SECOND_BYTES、正向包间隔（IAT）在 CIC/ToN 中只是近似关系，采用"按比例缩放"：
    新值 = 原值 × (字节比) × (时长比)。动作为 0 时缩放因子恰为 1，保证 P(x, x) = x。
  - 正向 IAT 的 MIN/MAX/STDDEV 与 AVG 同比例缩放（近似：假设时间轴整体拉伸，新包均匀插入）。
不可改：PROTOCOL、L7_PROTO、全部反向（OUT_*、DST_TO_SRC_* 中非派生部分、SERVER_*、TCP_WIN_MAX_OUT、DURATION_OUT）、
        重传、DNS、FTP、ICMP 特征。
"""
import numpy as np
import pandas as pd

MTU = 1500
HDR_TCP, HDR_OTHER = 40, 28
BINS = [("NUM_PKTS_UP_TO_128_BYTES", 0, 128), ("NUM_PKTS_128_TO_256_BYTES", 128, 256),
        ("NUM_PKTS_256_TO_512_BYTES", 256, 512), ("NUM_PKTS_512_TO_1024_BYTES", 512, 1024),
        ("NUM_PKTS_1024_TO_1514_BYTES", 1024, 1514)]
FWD_IAT = ["SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX", "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV"]
SETTINGS = ("STRICT", "LOOSE")


def _bin_index(s):
    """单包大小 → 分段编号（≤128 为 0，(128,256] 为 1，……）。"""
    s = np.asarray(s, dtype=float)
    return np.select([s <= 128, s <= 256, s <= 512, s <= 1024], [0, 1, 2, 3], default=4)


def apply_actions(orig: pd.DataFrame, act: dict, setting: str = "STRICT") -> pd.DataFrame:
    """
    由原流与动作重建新流。act 中各数组与 orig 行数相同：
      dp（追加包数，≥0 整数）、db（追加字节）、dt（追加毫秒，≥0）；宽松设定另有 ttl_min、ttl_max、win_in、flags_add。
    缺省的动作视为 0 / 不改。
    """
    if setting not in SETTINGS:
        raise ValueError(setting)
    o = orig.astype(np.float64)
    x = o.copy()
    n = len(o)
    z = np.zeros(n)
    tcp = np.round(o["PROTOCOL"].to_numpy()) == 6
    hdr = np.where(tcp, HDR_TCP, HDR_OTHER)

    # ---- A1 追加正向包 ----
    dp = np.maximum(0, np.round(np.asarray(act.get("dp", z), float)))
    db = np.asarray(act.get("db", z), float)
    db = np.where(dp > 0, np.clip(np.round(db), dp * hdr, dp * MTU), 0.0)
    has = dp > 0
    s_lo = np.where(has, np.floor(db / np.where(has, dp, 1)), 0)
    s_hi = np.where(has, np.ceil(db / np.where(has, dp, 1)), 0)
    n_hi = np.where(has, db - s_lo * dp, 0)  # 大小为 s_hi 的包数（其余 dp - n_hi 个为 s_lo）
    n_lo = dp - n_hi
    x["IN_PKTS"] = o.IN_PKTS + dp
    x["IN_BYTES"] = o.IN_BYTES + db
    longest = np.where(has, np.maximum(o.LONGEST_FLOW_PKT, np.where(n_hi > 0, s_hi, s_lo)), o.LONGEST_FLOW_PKT)
    x["LONGEST_FLOW_PKT"] = longest
    x["MAX_IP_PKT_LEN"] = np.where(has, np.maximum(o.MAX_IP_PKT_LEN, longest), o.MAX_IP_PKT_LEN)
    x["SHORTEST_FLOW_PKT"] = np.where(has, np.minimum(o.SHORTEST_FLOW_PKT, np.where(n_lo > 0, s_lo, s_hi)), o.SHORTEST_FLOW_PKT)
    # MIN_IP_PKT_LEN = 0 是数据中的"缺失"伪迹，保持不变；否则与最短包一致
    x["MIN_IP_PKT_LEN"] = np.where(has & (o.MIN_IP_PKT_LEN > 0),
                                   np.minimum(o.MIN_IP_PKT_LEN, x["SHORTEST_FLOW_PKT"]), o.MIN_IP_PKT_LEN)
    bi_lo, bi_hi = _bin_index(s_lo), _bin_index(s_hi)
    for k, (col, _, _) in enumerate(BINS):
        x[col] = o[col] + np.where(has & (bi_lo == k), n_lo, 0) + np.where(has & (bi_hi == k), n_hi, 0)

    # ---- A2 放慢 ----
    dt = np.maximum(0, np.round(np.asarray(act.get("dt", z), float)))
    x["FLOW_DURATION_MILLISECONDS"] = o.FLOW_DURATION_MILLISECONDS + dt
    x["DURATION_IN"] = o.DURATION_IN + dt

    # ---- 派生：吞吐量与 SECOND_BYTES（按比例缩放；动作为 0 时因子恰为 1）----
    d0, d1 = o.FLOW_DURATION_MILLISECONDS + 1, x.FLOW_DURATION_MILLISECONDS + 1
    tr = d0 / d1                                                   # 时长比
    br = np.where(o.IN_BYTES > 0, x.IN_BYTES / np.where(o.IN_BYTES > 0, o.IN_BYTES, 1), 1.0)
    fresh_tp = x.IN_BYTES * 8000 / d1                              # 原字节为 0 时直接按定义计算
    x["SRC_TO_DST_AVG_THROUGHPUT"] = np.where(o.IN_BYTES > 0, o.SRC_TO_DST_AVG_THROUGHPUT * br * tr, fresh_tp)
    x["DST_TO_SRC_AVG_THROUGHPUT"] = o.DST_TO_SRC_AVG_THROUGHPUT * tr
    # 检验 A：SECOND_BYTES 命名与含义相反——SRC_TO_DST 对应反向字节，DST_TO_SRC 对应正向字节
    x["SRC_TO_DST_SECOND_BYTES"] = o.SRC_TO_DST_SECOND_BYTES * tr
    x["DST_TO_SRC_SECOND_BYTES"] = o.DST_TO_SRC_SECOND_BYTES * br * tr

    # ---- 派生：正向包间隔（整体按比例缩放）----
    p0, p1 = o.IN_PKTS, x.IN_PKTS
    di0, di1 = o.DURATION_IN, x.DURATION_IN
    changed = has | (dt > 0)
    ok = changed & (p0 > 1) & (di0 > 0) & (o.SRC_TO_DST_IAT_AVG > 0)
    f = np.where(ok, (di1 / np.where(di0 > 0, di0, 1)) * ((p0 - 1) / np.where(p1 > 1, p1 - 1, 1)), 1.0)
    for c in FWD_IAT:
        x[c] = o[c] * f
    # 原流无法缩放（单包或时长为 0）但动作改变了流：用定义直接给出均匀间隔
    raw = changed & ~ok & (p1 > 1)
    avg = di1 / np.where(p1 > 1, p1 - 1, 1)
    for c in ("SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX", "SRC_TO_DST_IAT_AVG"):
        x[c] = np.where(raw, avg, x[c])
    x["SRC_TO_DST_IAT_STDDEV"] = np.where(raw, 0.0, x["SRC_TO_DST_IAT_STDDEV"])

    if setting == "LOOSE":
        # ---- A3 TTL ----
        # 只有攻击者真的改了才改；未改时保留原值（含数据中 TTL=0 的"缺失"伪迹），保证 P(x, x) = x
        if "ttl_min" in act or "ttl_max" in act:
            a_raw = np.round(np.asarray(act.get("ttl_min", o.MIN_TTL), float))
            b_raw = np.round(np.asarray(act.get("ttl_max", o.MAX_TTL), float))
            ch = (a_raw != o.MIN_TTL.to_numpy()) | (b_raw != o.MAX_TTL.to_numpy())
            a, b = np.clip(a_raw, 1, 255), np.clip(b_raw, 1, 255)
            x["MIN_TTL"] = np.where(ch, np.minimum(a, b), o.MIN_TTL)
            x["MAX_TTL"] = np.where(ch, np.maximum(a, b), o.MAX_TTL)
        # ---- A4 客户端窗口（仅 TCP；上限取 65535 与原值的较大者，兼容窗口缩放后的大值）----
        if "win_in" in act:
            w_raw = np.round(np.asarray(act["win_in"], float))
            ch = w_raw != o.TCP_WIN_MAX_IN.to_numpy()
            w = np.clip(w_raw, 0, np.maximum(65535, o.TCP_WIN_MAX_IN.to_numpy()))
            x["TCP_WIN_MAX_IN"] = np.where(tcp & ch, w, o.TCP_WIN_MAX_IN)
        # ---- A5 追加客户端 TCP 标志（仅 TCP，只能按位或）----
        if "flags_add" in act:
            fa = np.clip(np.round(np.asarray(act["flags_add"], float)), 0, 255).astype(np.int64)
            cf = np.round(o.CLIENT_TCP_FLAGS.to_numpy()).astype(np.int64)
            x["CLIENT_TCP_FLAGS"] = np.where(tcp, cf | fa, cf).astype(float)
    cf = np.round(x.CLIENT_TCP_FLAGS.to_numpy()).astype(np.int64)
    sf = np.round(x.SERVER_TCP_FLAGS.to_numpy()).astype(np.int64)
    x["TCP_FLAGS"] = np.where(cf | sf == np.round(o.TCP_FLAGS.to_numpy()).astype(np.int64), o.TCP_FLAGS, (cf | sf).astype(float))
    return x[orig.columns]


def actions_from_adv(orig: pd.DataFrame, adv: pd.DataFrame, setting: str = "STRICT") -> dict:
    """把任意对抗样本映射为最接近的动作（对抗样本中的正向增量 → 动作；不可实现的部分丢弃）。"""
    o, a = orig.astype(np.float64), adv.astype(np.float64)
    act = {
        "dp": np.maximum(0, np.round(a.IN_PKTS - o.IN_PKTS)).to_numpy(),
        "db": np.maximum(0, a.IN_BYTES - o.IN_BYTES).to_numpy(),
        "dt": np.maximum(0, np.maximum(a.FLOW_DURATION_MILLISECONDS - o.FLOW_DURATION_MILLISECONDS,
                                       a.DURATION_IN - o.DURATION_IN)).to_numpy(),
    }
    if setting == "LOOSE":
        act["ttl_min"], act["ttl_max"] = a.MIN_TTL.to_numpy(), a.MAX_TTL.to_numpy()
        act["win_in"] = a.TCP_WIN_MAX_IN.to_numpy()
        fl = np.clip(np.round(a.CLIENT_TCP_FLAGS.to_numpy()), 0, 255).astype(np.int64)
        act["flags_add"] = (fl & ~np.round(o.CLIENT_TCP_FLAGS.to_numpy()).astype(np.int64)) & 0xFF
    return act


def project(orig: pd.DataFrame, adv: pd.DataFrame, setting: str = "STRICT") -> pd.DataFrame:
    """约束投影 P(x_adv | x_orig)：先映射为动作，再由动作重建。"""
    return apply_actions(orig, actions_from_adv(orig, adv, setting), setting)


def random_actions(orig: pd.DataFrame, rng, setting: str = "STRICT", max_pkt_mult: float = 10.0,
                   max_dt_mult: float = 10.0) -> dict:
    """
    在动作空间中随机采样一组动作（供黑盒搜索攻击使用）。
    代价上限：追加包数 ≤ max_pkt_mult × 原正向包数（至少 1 个），追加时长 ≤ max_dt_mult × (原时长 + 1000 ms)。
    包数、字节、时长按对数均匀采样，以兼顾小扰动与大扰动。
    """
    n = len(orig)
    p0 = orig.IN_PKTS.to_numpy(float)
    d0 = orig.FLOW_DURATION_MILLISECONDS.to_numpy(float)
    tcp = np.round(orig.PROTOCOL.to_numpy()) == 6
    hdr = np.where(tcp, HDR_TCP, HDR_OTHER)
    use_p = rng.random(n) < 0.7
    dp = np.where(use_p, np.floor(np.exp(rng.uniform(0, np.log1p(max_pkt_mult * np.maximum(p0, 1)), n))), 0)
    size = np.exp(rng.uniform(np.log(hdr), np.log(MTU), n))
    use_t = rng.random(n) < 0.7
    dt = np.where(use_t, np.expm1(rng.uniform(0, np.log1p(max_dt_mult * (d0 + 1000)), n)), 0)
    act = {"dp": dp, "db": dp * size, "dt": dt}
    if setting == "LOOSE":
        keep = rng.random(n) < 0.5
        t1 = rng.integers(1, 256, n)
        t2 = np.where(rng.random(n) < 0.7, t1, rng.integers(1, 256, n))
        act["ttl_min"] = np.where(keep, orig.MIN_TTL, t1)
        act["ttl_max"] = np.where(keep, orig.MAX_TTL, t2)
        act["win_in"] = np.where(rng.random(n) < 0.5, orig.TCP_WIN_MAX_IN, rng.integers(0, 65536, n))
        act["flags_add"] = np.where(rng.random(n) < 0.7, 0, rng.integers(0, 256, n))
    return act
