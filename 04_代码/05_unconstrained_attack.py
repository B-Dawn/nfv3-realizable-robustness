"""
05_unconstrained_attack.py
可行性检验 B-2：无约束攻击生成的对抗样本中，违反协议约束与取值域约束的比例。

用法（项目根目录）：python 04_代码/05_unconstrained_attack.py <源数据集名> [time|random] [--models MLP,RF,XGB] [--n-tree 50]
  --models 只运行指定模型（如 RF 单独运行，避免大模型超时）；--n-tree 为 RF/XGB 的攻击样本数。
  指定 --models 时输出文件名附加模型后缀，例如 D_attack_<源>_<切分>_RF.csv。
输入：05_模型/<源>_<切分>_{RF,XGB,MLP}.joblib（04 脚本 RAW 变体），01_数据/processed/<源>.pkl
输出：03_结果/D_attack_<源>_<切分>.csv

- 对象：源数据集测试集中被模型正确检出的攻击流（同数据集攻击）。
- MLP：白盒 PGD（L∞，标准化 log1p 空间，ε ∈ {0.1, 0.3, 1.0}，20 步），梯度由权重手工反向传播计算。
- RF / XGB：黑盒 HopSkipJump（ART，L2，原始特征空间，裁剪到训练集各特征的最小/最大值）。
  逐样本运行，每样本查询预算 20,000 次，超出按攻击失败计并记录数量（修复 ART 步长搜索死循环，见 hsj() 说明）。
  决策边界取 04 脚本保存的源 FPR=1% 阈值（以 BlackBoxClassifier 封装），而非默认的 0.5。
  初始对抗点取训练集中被模型判为良性的真实流（随机抽取），避免随机初始化找不到良性起点。
- 统计：原样本违反率（应为 0，检验检查器）；全部对抗样本与攻击成功样本的违反率；
  “仅修补取值域”（计数取整、负值截零、标志与 TTL 截到 0–255）后的攻击成功率与协议约束违反率。
单一随机种子（0），属可行性检验。
"""
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nfv3_constraints import BYTE_FLAG_FEATURES, INT_FEATURES, TTL_FEATURES, apply_split, summary  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "01_数据" / "processed"
RES = ROOT / "03_结果"
MOD = ROOT / "05_模型"
DATASETS = ["NF-UNSW-NB15-v3", "NF-CSE-CIC-IDS2018-v3", "NF-ToN-IoT-v3"]
SEED = 0
N_MLP, N_TREE = 300, 50
EPS = [0.1, 0.3, 1.0]


def domain_repair(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    ints = [c for c in INT_FEATURES if c in X]
    X[ints] = np.round(X[ints].clip(lower=0))
    fl = [c for c in BYTE_FLAG_FEATURES + TTL_FEATURES if c in X]
    X[fl] = X[fl].clip(0, 255)
    return X.clip(lower=0)


# ---------- MLP 白盒 PGD ----------
def mlp_parts(pipe):
    scaler, mlp = pipe.steps[1][1], pipe.steps[2][1]
    return scaler.mean_, scaler.scale_, mlp


def mlp_logit_grad(z, mlp):
    """返回输出 logit 及其对输入 z 的梯度（隐藏层 ReLU，二分类单输出）。"""
    acts = [z]
    h = z
    for W, b in zip(mlp.coefs_[:-1], mlp.intercepts_[:-1]):
        h = np.maximum(h @ W + b, 0)
        acts.append(h)
    logit = (h @ mlp.coefs_[-1] + mlp.intercepts_[-1]).ravel()
    g = np.repeat(mlp.coefs_[-1].T, len(z), axis=0)  # d logit / d h_last
    for i in range(len(mlp.coefs_) - 2, -1, -1):
        g = g * (acts[i + 1] > 0)
        g = g @ mlp.coefs_[i].T
    return logit, g


def pgd(pipe, X: np.ndarray, eps: float, steps: int = 20) -> np.ndarray:
    mu, sd, mlp = mlp_parts(pipe)
    z0 = (np.log1p(np.clip(X, 0, None)) - mu) / sd
    z = z0.copy()
    a = eps / 8
    for _ in range(steps):
        _, g = mlp_logit_grad(z, mlp)
        z = z - a * np.sign(g)  # 降低“攻击”logit，推向良性
        z = z0 + np.clip(z - z0, -eps, eps)
    return np.expm1(z * sd + mu)


# ---------- 树模型黑盒 HopSkipJump ----------
QUERY_BUDGET = 20_000  # 每个样本的查询预算（预测行数）；正常一轮约 5,500 次


class _BudgetExceeded(Exception):
    pass


def hsj(model, X: np.ndarray, Xtr: np.ndarray, thr: float, feats, Xben: np.ndarray):
    """
    逐样本运行 HopSkipJump，每个样本设查询预算。
    原因：ART 1.20.1 的步长搜索 `while not success: epsilon /= 2` 没有下限；树模型的边界恰在分裂阈值上，
    float32/float64 舍入可让当前点落在非对抗一侧，epsilon 减到 0 后永远不成功，形成死循环
    （2026-11-07 在 CIC 的 XGB 上占满 CPU 50 分钟）。超出预算的样本按攻击失败处理（返回原样本）。
    返回 (对抗样本, 超出预算标记)。
    """
    from art.attacks.evasion import HopSkipJump
    from art.estimators.classification import BlackBoxClassifier
    lo, hi = Xtr.min(axis=0).astype(np.float32), Xtr.max(axis=0).astype(np.float32)
    hi = np.where(hi > lo, hi, lo + 1)
    used = {"n": 0}

    def predict(x):
        used["n"] += len(x)
        if used["n"] > QUERY_BUDGET:
            raise _BudgetExceeded
        p = model.predict_proba(pd.DataFrame(np.asarray(x, dtype=np.float32), columns=feats))[:, 1] > thr
        return np.stack([~p, p], axis=1).astype(np.float32)

    clf = BlackBoxClassifier(predict, input_shape=(X.shape[1],), nb_classes=2, clip_values=(lo, hi))
    rng = np.random.default_rng(SEED)
    init = Xben[rng.choice(len(Xben), size=len(X), replace=len(Xben) < len(X))].astype(np.float32)
    X = X.astype(np.float32)
    out, hit = X.copy(), np.zeros(len(X), dtype=bool)
    for i in range(len(X)):
        used["n"] = 0
        atk = HopSkipJump(classifier=clf, targeted=False, norm=2, max_iter=10, max_eval=500, init_eval=100,
                          init_size=100, verbose=False)  # 每个样本新建，避免中断后内部状态残留
        try:
            out[i] = atk.generate(x=X[i:i + 1], x_adv_init=init[i:i + 1])[0]
        except _BudgetExceeded:
            hit[i] = True
    return out, hit


def main(src: str, mode: str, models=("MLP", "RF", "XGB"), n_tree: int = N_TREE):
    t0 = time.time()
    d = apply_split(pd.read_pickle(PROC / f"{src}.pkl"), mode)
    te = d[(d.split == "test") & (d.Label == 1)]
    tr = d[d.split == "train"]
    rng = np.random.default_rng(SEED)
    rows = []
    for mname in models:
        bundle = joblib.load(MOD / f"{src}_{mode}_{mname}.joblib")
        model, feats, thr = bundle["model"], bundle["features"], bundle["threshold"]
        Xte = te[feats]
        detected = Xte[model.predict_proba(Xte)[:, 1] > thr]
        n = N_MLP if mname == "MLP" else n_tree
        if len(detected) == 0:
            print(f"  {src} {mname}: 测试集中无被检出的攻击流，跳过", flush=True)
            continue
        X0 = detected.iloc[rng.choice(len(detected), size=min(n, len(detected)), replace=False)]
        trb = tr[tr.Label == 0][feats]
        trb = trb.iloc[rng.choice(len(trb), size=min(20_000, len(trb)), replace=False)]
        ben_pred = trb[model.predict_proba(trb)[:, 1] <= thr].to_numpy(np.float32)
        base = summary(X0)
        rows.append({"源": src, "切分": mode, "模型": mname, "攻击": "原样本（检查器自检）", "ε": None, "样本数": len(X0),
                     "攻击成功率": 0.0, **{f"违反率|{k}": v for k, v in base.items()}})
        settings = [(f"PGD L∞ ε={e}", e) for e in EPS] if mname == "MLP" else [("HopSkipJump L2", None)]
        for label, e in settings:
            t1 = time.time()
            if mname == "MLP":
                Xadv, n_hit = pgd(model, X0.to_numpy(np.float64), e), 0
            else:
                Xadv, hit = hsj(model, X0.to_numpy(np.float32), tr[feats].to_numpy(np.float32), thr, feats, ben_pred)
                n_hit = int(hit.sum())
            A = pd.DataFrame(Xadv, columns=feats, index=X0.index)
            succ = model.predict_proba(A)[:, 1] <= thr
            R = domain_repair(A)
            succ_r = model.predict_proba(R)[:, 1] <= thr
            for scope, frame, mask, sr in (("全部对抗样本", A, np.ones(len(A), bool), succ.mean()),
                                           ("攻击成功样本", A, succ, succ.mean()),
                                           ("仅修补取值域后·全部", R, np.ones(len(R), bool), succ_r.mean()),
                                           ("仅修补取值域后·仍成功", R, succ_r, succ_r.mean())):
                s = summary(frame[mask]) if mask.any() else {}
                rows.append({"源": src, "切分": mode, "模型": mname, "攻击": f"{label}｜{scope}", "ε": e, "样本数": int(mask.sum()),
                             "攻击成功率": float(sr), "超出查询预算样本数": n_hit,
                             **{f"违反率|{k}": v for k, v in s.items()}})
            print(f"  {src} {mname} {label}: 成功率 {succ.mean():.1%}，修补取值域后 {succ_r.mean():.1%}，"
                  f"超出查询预算 {n_hit}/{len(X0)}（{time.time() - t1:.0f}s）", flush=True)
    rep = pd.DataFrame(rows)
    suffix = "" if list(models) == ["MLP", "RF", "XGB"] else "_" + "-".join(models)
    out = RES / f"D_attack_{src}_{mode}{suffix}.csv"
    rep.to_csv(out, index=False, encoding="utf-8-sig")
    show = ["模型", "攻击", "样本数", "攻击成功率", "违反率|ANY 协议约束（C 类）", "违反率|ANY 全部（C 类 + 取值域 D 类）"]
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    print(rep[[c for c in show if c in rep]].round(4).to_string(index=False))
    print(f"\n已写入 {out}，用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("src", choices=DATASETS)
    ap.add_argument("mode", nargs="?", default="random", choices=["time", "random"])
    ap.add_argument("--models", default="MLP,RF,XGB")
    ap.add_argument("--n-tree", type=int, default=N_TREE)
    a = ap.parse_args()
    ms = [m.strip().upper() for m in a.models.split(",") if m.strip()]
    if not ms or any(m not in ("MLP", "RF", "XGB") for m in ms):
        sys.exit("--models 只能是 MLP、RF、XGB 的组合，用逗号分隔")
    main(a.src, a.mode, ms, a.n_tree)
