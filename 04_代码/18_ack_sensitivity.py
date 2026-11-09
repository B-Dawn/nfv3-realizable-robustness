"""
18_ack_sensitivity.py
敏感性分析：攻击者追加的 TCP 正向包若引发服务器额外确认（ACK），反向流量也会随之改变。
主实验假设反向包数不变（UDP，或 TCP 累积/延迟确认）；本脚本放宽该假设，检验结论是否稳健。

用法（项目根目录）：python 04_代码/18_ack_sensitivity.py <源数据集名> --seed s --ratio r [--n 200]
  ratio = 每个追加的 TCP 正向包引发的 ACK 数：1.0 = 逐包确认（最坏情况），0.5 = 延迟确认（每两个段一个 ACK），
          0 = 不加 ACK（应与 11 脚本正式结果 L_formal_<源>_RAW_s<seed>.csv 完全一致，用作正确性检验）。
输入：05_模型/<源>_RAW_<模型>_s<seed>.joblib（12 脚本），01_数据/processed/<源>.pkl
输出：03_结果/Q_ack_<源>_r<ratio>_s<seed>.csv（列与 L_formal 相同，另加 "ACK比例"）

ACK 的建模（只作用于 TCP、追加包数 > 0、且原流有反向包的流——服务器从不回应的流不会回 ACK）：
  k = ceil(ratio × 追加包数) 个 40 字节的反向纯 ACK 包；
  OUT_PKTS += k，OUT_BYTES += 40k；
  反向吞吐量与反向 SECOND_BYTES（数据中命名相反：SRC_TO_DST_SECOND_BYTES 对应反向字节）按反向字节比缩放；
  ≤128 字节分段计数 += k（分段计数统计两个方向的包）；最短包长 / 最小 IP 包长取 min(原值, 40)（原值 0 为缺失伪迹，保持）；
  反向包间隔按 (q0−1)/(q1−1) 缩放（反向时长 DURATION_OUT 不变）；SERVER_TCP_FLAGS 置 ACK 位（0x10），TCP_FLAGS 重算。
  随机数消耗与 11 脚本完全相同，因此 ratio=0 时逐项复现正式结果。
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nfv3_constraints import apply_split  # noqa: E402

_spec = importlib.util.spec_from_file_location("adaptive", HERE / "11_adaptive_search.py")
ad = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ad)

ACK_BYTES = 40
BWD_IAT = ["DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX", "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV"]
RATIO = 0.0


def add_acks(X, Y, act, ratio):
    """X 原流，Y 由动作重建的流（11 脚本 build 的输出），act 动作；返回加上服务器 ACK 后的流。"""
    if ratio <= 0:
        return Y
    o = X.astype(np.float64).reset_index(drop=True)
    y = Y.astype(np.float64).reset_index(drop=True).copy()
    tcp = np.round(o.PROTOCOL.to_numpy()) == 6
    dp = np.maximum(0, np.round(np.asarray(act["dp"], float)))
    q0 = o.OUT_PKTS.to_numpy()
    k = np.where(tcp & (dp > 0) & (q0 > 0), np.ceil(ratio * dp), 0.0)
    has = k > 0
    b0 = o.OUT_BYTES.to_numpy()
    y["OUT_PKTS"] = o.OUT_PKTS + k
    y["OUT_BYTES"] = o.OUT_BYTES + ACK_BYTES * k
    bb = np.where(has & (b0 > 0), y.OUT_BYTES.to_numpy() / np.where(b0 > 0, b0, 1), 1.0)
    y["DST_TO_SRC_AVG_THROUGHPUT"] = y.DST_TO_SRC_AVG_THROUGHPUT * bb
    y["SRC_TO_DST_SECOND_BYTES"] = y.SRC_TO_DST_SECOND_BYTES * bb
    y["NUM_PKTS_UP_TO_128_BYTES"] = y.NUM_PKTS_UP_TO_128_BYTES + k
    y["SHORTEST_FLOW_PKT"] = np.where(has, np.minimum(y.SHORTEST_FLOW_PKT, ACK_BYTES), y.SHORTEST_FLOW_PKT)
    y["MIN_IP_PKT_LEN"] = np.where(has & (y.MIN_IP_PKT_LEN > 0), np.minimum(y.MIN_IP_PKT_LEN, ACK_BYTES), y.MIN_IP_PKT_LEN)
    q1 = y.OUT_PKTS.to_numpy()
    ok = has & (q0 > 1) & (o.DST_TO_SRC_IAT_AVG.to_numpy() > 0)
    f = np.where(ok, (q0 - 1) / np.where(q1 > 1, q1 - 1, 1), 1.0)
    for c in BWD_IAT:
        y[c] = o[c].to_numpy() * f
    raw = has & ~ok & (q1 > 1)
    avg = o.DURATION_OUT.to_numpy() / np.where(q1 > 1, q1 - 1, 1)
    for c in ("DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX", "DST_TO_SRC_IAT_AVG"):
        y[c] = np.where(raw, avg, y[c])
    y["DST_TO_SRC_IAT_STDDEV"] = np.where(raw, 0.0, y["DST_TO_SRC_IAT_STDDEV"])
    sf = np.round(y.SERVER_TCP_FLAGS.to_numpy()).astype(np.int64)
    y["SERVER_TCP_FLAGS"] = np.where(has, sf | 0x10, sf).astype(float)
    cf = np.round(y.CLIENT_TCP_FLAGS.to_numpy()).astype(np.int64)
    sf = np.round(y.SERVER_TCP_FLAGS.to_numpy()).astype(np.int64)
    y["TCP_FLAGS"] = np.where(has, (cf | sf).astype(float), y.TCP_FLAGS)
    y.index = Y.index
    return y[Y.columns]


_build_orig = ad.build


def build_ack(X, act, setting):
    return add_acks(X, _build_orig(X, act, setting), act, RATIO)


ad.build = build_ack  # run() 在 11 脚本命名空间中查找 build，因此被替换


def main(src, n, seed, ratio):
    global RATIO
    RATIO = ratio
    t0 = time.time()
    d = apply_split(pd.read_pickle(ad.PROC / f"{src}.pkl"), "random", seed=seed)
    allf = [c for c in d.columns if c not in ad.NON_FEAT]
    te = d[(d.split == "test") & (d.Label == 1)][allf]
    va = d[d.split == "val"]
    rows = []
    for mname in ("MLP", "RF", "XGB"):
        b = joblib.load(ad.MOD / f"{src}_RAW_{mname}_s{seed}.joblib")
        model, feats = b["model"], b["features"]
        ben = model.predict_proba(va[feats])[:, 1][va.Label.to_numpy() == 0]
        thrs = {f"FPR {f:.1%}": float(np.quantile(ben, 1 - f)) for f in ad.FPRS}
        thrs["默认 0.5"] = 0.5
        for setting in ("STRICT", "LOOSE"):
            rng = np.random.default_rng(seed)
            s_te = model.predict_proba(te[feats])[:, 1]
            det = te[s_te > max(thrs.values())]
            X0 = det.iloc[rng.choice(len(det), size=min(n, len(det)), replace=False)] if len(det) else det
            if not len(X0):
                continue
            tcp_share = float((np.round(X0.PROTOCOL) == 6).mean())
            t1 = time.time()
            hard, sb, viol = ad.run(model, feats, thrs, X0, setting, rng)
            for t, thr in thrs.items():
                for cap in ad.CAPS_REPORT:
                    rows.append({"源": src, "变体": "RAW", "ACK比例": ratio, "种子": seed, "模型": mname, "设定": setting,
                                 "工作点": t, "阈值": thr, "代价上限": f"≤{cap:g}×", "样本数": len(X0),
                                 "TCP占比": tcp_share, "判决型成功率": hard[t][cap], "分数型成功率": sb[t][cap],
                                 "构造流违反率": viol})
            print(f"  {src} r={ratio} s{seed} {mname} {setting}: TCP {tcp_share:.0%}；FPR1% ≤1× 判决型 "
                  f"{hard['FPR 1.0%'][1.0]:.1%} / 分数型 {sb['FPR 1.0%'][1.0]:.1%}，违反率 {viol:.4%}（{time.time() - t1:.0f}s）",
                  flush=True)
    rep = pd.DataFrame(rows)
    out = ad.RES / f"Q_ack_{src}_r{ratio:g}_s{seed}.csv"
    rep.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"构造流最大违反率：{rep['构造流违反率'].max()}（必须为 0）")
    if ratio == 0:
        ref = pd.read_csv(ad.RES / f"L_formal_{src}_RAW_s{seed}.csv")
        key = ["模型", "设定", "工作点", "代价上限"]
        m = rep.merge(ref, on=key, suffixes=("", "_ref"))
        dmax = float(np.max(np.abs(m[["判决型成功率", "分数型成功率"]].to_numpy()
                                   - m[["判决型成功率_ref", "分数型成功率_ref"]].to_numpy())))
        print(f"ratio=0 与 L_formal 对照：匹配 {len(m)}/{len(ref)} 行，最大差异 {dmax}（必须为 0）")
    print(f"已写入 {out}，用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", choices=ad.DATASETS)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ratio", type=float, default=1.0)
    a = ap.parse_args()
    main(a.src, a.n, a.seed, a.ratio)
