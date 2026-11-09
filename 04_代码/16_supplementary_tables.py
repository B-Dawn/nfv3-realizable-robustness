"""Generate supplementary-material tables (LaTeX) from 03_结果 CSVs. Usage: python gen_supp.py <results_dir> <out_tex>"""
import glob, sys, re
import pandas as pd

R, OUT = sys.argv[1].rstrip("/") + "/", sys.argv[2]
DS = {"NF-UNSW-NB15-v3": "UNSW", "NF-CSE-CIC-IDS2018-v3": "CIC", "NF-ToN-IoT-v3": "ToN"}
MODELS = ["RF", "XGB", "MLP"]
OPS = ["FPR 0.1%", "FPR 1.0%", "FPR 5.0%", "默认 0.5"]
OPL = ["FPR 0.1\\%", "FPR 1\\%", "FPR 5\\%", "Default 0.5"]
L = []
w = L.append


def pm(s):
    """'0.123 ± 0.045' -> LaTeX; drop ±0.000."""
    m, sd = [t.strip() for t in str(s).split("±")]
    return m if float(sd) == 0 else f"{m}\\,{{\\scriptsize$\\pm${sd}}}"


# ---------------- S1/S2: full attack results ----------------
A = pd.read_csv(R + "M_summary_attack.csv")


def attack_table(variant, cost, col, title, label, datasets=DS):
    g = A[(A["变体"] == variant) & (A["代价上限"] == cost)]
    w("\\begin{table*}[!htbp]\\centering\\footnotesize")
    w(f"\\caption{{{title}}}\\label{{{label}}}")
    w("\\begin{tabular}{@{}lll" + "c" * 4 + "@{}}\\toprule")
    w("Dataset & Model & Setting & " + " & ".join(OPL) + "\\\\\\midrule")
    first = True
    for ds, dl in datasets.items():
        if not first:
            w("\\midrule")
        first = False
        for m in MODELS:
            for st in ["STRICT", "LOOSE"]:
                r = g[(g["源"] == ds) & (g["模型"] == m) & (g["设定"] == st)].set_index("工作点")
                if r.empty:
                    continue
                cells = [pm(r.loc[o, col]) for o in OPS]
                w(f"{dl} & {m} & {'\\textsc{Strict}' if st == 'STRICT' else '\\textsc{Loose}'} & " + " & ".join(cells) + "\\\\")
    w("\\bottomrule\\end{tabular}\\end{table*}")


w("\\section{Complete Attack Results}\\label{sec:s-attack}")
w("Tables~\\ref{tab:s-score-0.1}--\\ref{tab:s-hard-10} give the evasion success rate (mean $\\pm$ SD over seeds 0, 1, 2; "
  "SD omitted when it is 0.000) for every dataset, model, capability setting, operating point and cost budget. "
  "200 attack flows are used per dataset, model and seed. Table~IV of the main paper is the score-based, $\\bar{c}\\le1$ slice.")
for cost, cl in [("≤0.1×", "0.1"), ("≤1×", "1"), ("≤10×", "10")]:
    attack_table("RAW", cost, "分数型成功率", f"Score-Based Attacker, Cost Budget $\\bar{{c}}\\le{cl}$", f"tab:s-score-{cl}")
for cost, cl in [("≤0.1×", "0.1"), ("≤1×", "1"), ("≤10×", "10")]:
    attack_table("RAW", cost, "判决型成功率", f"Hard-Label Attacker, Cost Budget $\\bar{{c}}\\le{cl}$", f"tab:s-hard-{cl}")
w("\\clearpage")

# ---------------- S3: TTL removal ----------------
D = pd.read_csv(R + "M_summary_detection.csv")
w("\\section{Retraining Without TTL Features (UNSW)}\\label{sec:s-nottl}")
w("We remove MIN\\_TTL and MAX\\_TTL (46 $\\rightarrow$ 44 features) and retrain all models with the same seeds. "
  "Table~\\ref{tab:s-nottl-det} shows that detection is unchanged; Table~\\ref{tab:s-nottl-atk} gives the attack results "
  "of the score-based attacker at $\\bar{c}\\le1$. Table~\\ref{tab:s-ttl} shows why the TTL is a shortcut: in UNSW the "
  "benign and attack flows use almost disjoint TTL values, which the sender sets freely.")
w("\\begin{table}[!htbp]\\centering\\footnotesize\\caption{Detection on UNSW With and Without TTL (Mean $\\pm$ SD)}\\label{tab:s-nottl-det}")
w("\\begin{tabular}{@{}llcc@{}}\\toprule Features & Model & AUROC & TPR@1\\% FPR\\\\\\midrule")
for v, vl in [("RAW", "46 (with TTL)"), ("noTTL", "44 (no TTL)")]:
    for m in MODELS:
        r = D[(D["源"] == "NF-UNSW-NB15-v3") & (D["变体"] == v) & (D["模型"] == m)].iloc[0]
        w(f"{vl} & {m} & {pm(r['AUROC'])} & {pm(r['TPR@FPR1%'])}\\\\")
w("\\bottomrule\\end{tabular}\\end{table}")
g = A[(A["源"] == "NF-UNSW-NB15-v3") & (A["代价上限"] == "≤1×")]
w("\\begin{table}[!htbp]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}"
  "\\caption{UNSW, Score-Based Attacker, $\\bar{c}\\le1$, \\textsc{Loose}: With vs.\\ Without TTL}\\label{tab:s-nottl-atk}")
w("\\begin{tabular}{@{}ll" + "c" * 4 + "@{}}\\toprule Features & Model & " + " & ".join(OPL) + "\\\\\\midrule")
for v, vl in [("RAW", "with TTL"), ("noTTL", "no TTL")]:
    for m in MODELS:
        r = g[(g["变体"] == v) & (g["模型"] == m) & (g["设定"] == "LOOSE")].set_index("工作点")
        w(f"{vl} & {m} & " + " & ".join(pm(r.loc[o, "分数型成功率"]) for o in OPS) + "\\\\")
w("\\bottomrule\\end{tabular}\\end{table}")
ttl = open(R + "E_ttl_summary.txt", encoding="utf-8").read().splitlines()
w("\\begin{table}[!htbp]\\centering\\footnotesize\\caption{Five Most Frequent TTL Values per Class (Share of Flows, Seed-0 Sample)}\\label{tab:s-ttl}")
w("\\begin{tabular}{@{}lll p{5.2cm}@{}}\\toprule Data & Field & Class & Top values (share)\\\\\\midrule")
for line in ttl:
    mm = re.match(r"(\w+) (MIN_TTL|MAX_TTL) \[(良性|攻击)\] 前 5 取值：(.*)", line.strip())
    if mm:
        vals = mm.group(4).replace("，", ", ").replace("%", "\\%")
        w(f"{mm.group(1)} & {mm.group(2).replace('_', '\\_')} & {'benign' if mm.group(3) == '良性' else 'attack'} & {vals}\\\\")
w("\\bottomrule\\end{tabular}\\end{table}")
w("\\clearpage")

# ---------------- S4: candidate constraints ----------------
EN = {
    "C01": "Sum of packet-length histogram counts $=$ total packets",
    "C02": "Sum of histogram counts $\\le$ total packets (allowing packets $>$1514\\,B)",
    "C03": "MIN\\_IP\\_PKT\\_LEN $\\le$ MAX\\_IP\\_PKT\\_LEN",
    "C04": "LONGEST\\_FLOW\\_PKT $=$ MAX\\_IP\\_PKT\\_LEN",
    "C05": "SHORTEST\\_FLOW\\_PKT $=$ MIN\\_IP\\_PKT\\_LEN",
    "C06": "Total bytes $\\in$ [packets$\\times$shortest, packets$\\times$longest]",
    "C07": "Total bytes within the bounds implied by histogram counts",
    "C08": "Retransmitted packets $\\le$ packets (per direction)",
    "C09": "Retransmitted bytes $\\le$ bytes (per direction)",
    "C10": "TCP\\_FLAGS $=$ CLIENT\\_TCP\\_FLAGS $\\lor$ SERVER\\_TCP\\_FLAGS",
    "C11": "Non-TCP flow: all TCP flags $=0$",
    "C12": "Non-TCP flow: TCP windows $=0$",
    "C13": "Non-TCP flow: retransmissions $=0$",
    "C14": "MIN\\_TTL $\\le$ MAX\\_TTL",
    "C15": "$\\lfloor$ICMP\\_TYPE$/256\\rfloor=$ ICMP\\_IPV4\\_TYPE",
    "C16": "DURATION\\_IN/OUT $\\le$ FLOW\\_DURATION ($+2$\\,ms)",
    "C17": "END$-$START $=$ FLOW\\_DURATION ($\\pm2$\\,ms)",
    "C23": "Count features $\\ge0$",
}


def en_name(c):
    k = c[:3]
    if k in EN:
        return EN[k]
    if k == "C18":
        d = "backward" if "DST_TO_SRC" in c else "forward"
        return f"IAT ({d}): MIN $\\le$ AVG $\\le$ MAX, STD $\\ge0$"
    mm = re.match(r"C(19|20|21|22) (\w+) ≈ (\w+)", c)
    base = mm.group(2).replace("_", "\\_")
    num = mm.group(3).replace("_", "\\_")
    if mm.group(1) in ("19", "20"):
        return f"{base} $\\approx$ {num}$/$FLOW\\_DURATION"
    return f"{base} $\\approx$ {num}$\\times8/$(END$-$START)"


w("\\section{All Candidate Constraints and Their Violation Rates}\\label{sec:s-cons}")
w("Table~\\ref{tab:s-cons} lists every candidate relation we checked on all 50{,}001{,}213 flows, with the violation rate (\\%) among "
  "benign (B) and attack (A) flows of each dataset. Only relations with 0 violations in every cell were kept (marked \\checkmark). "
  "Relations C19--C22 are violated by a large share of flows, which is why the projection operator rescales these features "
  "proportionally instead of recomputing them.")
cons = {dl: pd.read_csv(R + f"A_constraint_check_{ds}.csv") for ds, dl in DS.items()}
names = list(dict.fromkeys(cons["UNSW"]["约束"]))
w("\\begin{table*}[!htbp]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\\caption{Candidate Constraints: Violation Rate (\\%) on All Flows}\\label{tab:s-cons}")
w("\\begin{tabular}{@{}l >{\\raggedright\\arraybackslash}p{7.4cm} cccccc c@{}}\\toprule")
w(" & & \\multicolumn{2}{c}{UNSW} & \\multicolumn{2}{c}{CIC} & \\multicolumn{2}{c}{ToN} & \\\\ \\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}")
w("ID & Relation & B & A & B & A & B & A & Kept\\\\\\midrule")
for n in names:
    cells, allzero = [], True
    for dl in ["UNSW", "CIC", "ToN"]:
        c = cons[dl]
        for cls in ["良性", "攻击"]:
            r = c[(c["约束"] == n) & (c["类别"] == cls)].iloc[0]
            allzero &= int(r["违反数"]) == 0
            v = 100 * r["违反率"]
            cells.append("0" if r["违反数"] == 0 else (f"{v:.2f}" if v >= 0.01 else "$<$0.01"))
    w(f"{n[:3]} & {en_name(n)} & " + " & ".join(cells) + f" & {'\\checkmark' if allzero else ''}\\\\")
tot = {dl: int(c[c["约束"] == names[0]]["样本数"].sum()) for dl, c in cons.items()}
w("\\bottomrule\\end{tabular}")
w(f"\\\\[2pt]\\footnotesize Flows checked: UNSW {tot['UNSW']:,}; CIC {tot['CIC']:,}; ToN {tot['ToN']:,}; total {sum(tot.values()):,}.".replace(",", "{,}"))
w("\\end{table*}")

# ---------------- S5: projection checks ----------------
G = pd.read_csv(R + "G_projection_check.csv")
TEN = {"T1": "identity $P(x,0)=x$", "T2": "original flows: 0\\% violations", "T3": "random actions: 0\\% violations",
       "T4": "random perturbation, after projection: 0\\% violations", "T5": "immutable features unchanged",
       "T6": "increase-only features never decrease"}
w("\\section{Projection Operator Checks}\\label{sec:s-proj}")
w(f"All {len(G)} checks ({len(G) // 3} per dataset, 50{{,}}000 real flows each) passed. Table~\\ref{{tab:s-proj}} lists them.")
w("\\begin{table}[!htbp]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\\caption{Projection Checks (Pass/Fail per Dataset)}\\label{tab:s-proj}")
w("\\begin{tabular}{@{}l >{\\raggedright\\arraybackslash}p{7cm} ccc@{}}\\toprule Test & Description & UNSW & CIC & ToN\\\\\\midrule")
for t in dict.fromkeys(G["检验"]):
    k = t[:2]
    desc = TEN[k] + (" (after projection)" if "投影后只增" in t else "")
    st = "\\textsc{Loose}" if "LOOSE" in t else ("\\textsc{Strict}" if "STRICT" in t else "--")
    row = [("pass" if bool(G[(G["数据集"] == ds) & (G["检验"] == t)]["通过"].iloc[0]) else "FAIL") for ds in DS]
    w(f"{k} & {desc}, {st} & " + " & ".join(row) + "\\\\")
w("\\bottomrule\\end{tabular}\\end{table}")

# ---------------- S6: data preparation ----------------
C0 = pd.read_csv(R + "C0_split_summary.csv")
w("\\section{Data Preparation and the Chronological Split}\\label{sec:s-data}")
w("\\emph{Invalid values.} While sampling, infinite or unparsable values were set to 0. Counts per column in the sampled data: ")
parts = []
for ds, dl in DS.items():
    r = C0[(C0["数据集"] == ds) & (C0["切分"].str.contains("无穷"))].iloc[0]
    det = str(r["起始时间"])
    if det == "无":
        parts.append(f"{dl}: none")
    else:
        det = det.replace("（NaN）", " (NaN)").replace("_", "\\_").replace("; ", ", ")
        parts.append(f"{dl}: {det}")
w("; ".join(parts) + ".")
w("\\emph{Chronological split.} Sorting the sample by flow start time and taking the first 70\\% for training (its last 15\\% for "
  "validation) and the last 30\\% for testing places attack types in the test set that never occur in training (CIC: Bot, "
  "Infilteration; ToN: backdoor, mitm, password, ransomware, xss), and the CIC validation split then contains only 23 attack flows. "
  "Table~\\ref{tab:s-time} contrasts within-dataset detection under the two splits (seed 0). Because the chronological split "
  "measures generalization to unseen attack types rather than robustness, all robustness experiments use the stratified random split.")
w("\\begin{table}[!htbp]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\\caption{Within-Dataset Detection, Random vs.\\ Chronological Split (Seed 0)}\\label{tab:s-time}")
w("\\begin{tabular}{@{}ll cc cc@{}}\\toprule & & \\multicolumn{2}{c}{AUROC} & \\multicolumn{2}{c}{TPR at val.\\ threshold} \\\\ \\cmidrule(lr){3-4}\\cmidrule(l){5-6}")
w("Data & Model & Random & Chron. & Random & Chron.\\\\\\midrule")
for ds, dl in DS.items():
    rr = pd.read_csv(R + f"C_baseline_{ds}_random.csv")
    tt = pd.read_csv(R + f"C_baseline_{ds}_time.csv")
    for m in MODELS:
        a = rr[(rr["同数据集"]) & (rr["模型"] == m) & (rr["变体"] == "RAW")].iloc[0]
        b = tt[(tt["同数据集"]) & (tt["模型"] == m) & (tt["变体"] == "RAW")].iloc[0]
        w(f"{dl} & {m} & {a['AUROC']:.3f} & {b['AUROC']:.3f} & {a['TPR@源阈值']:.3f} & {b['TPR@源阈值']:.3f}\\\\")
w("\\bottomrule\\end{tabular}\\end{table}")
w("\\clearpage")

# ---------------- S7: measurement methods, all configs ----------------
P = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(R + "P_methods_*_s*.csv"))])


def ms(x):
    x = x.dropna()
    return "--" if x.empty else (f"{x.mean():.3f}" if x.std() < 5e-4 else f"{x.mean():.3f}\\,{{\\scriptsize$\\pm${x.std():.3f}}}")


w("\\section{Unconstrained, Attack-then-Repair and Realizable Attacks: All Configurations}\\label{sec:s-methods}")
w("Table~\\ref{tab:s-methods} extends Fig.~2 of the main paper to all configurations (1\\% FPR, mean $\\pm$ SD over three seeds). "
  "Unconstrained: PGD for the MLP (200 flows) and HopSkipJump for trees (50 flows). Repair: the unconstrained output projected with $P$. "
  "Realizable: hard-label action-space search with 500 queries at $\\bar{c}\\le1$ and $\\bar{c}\\le10$.")
w("\\begin{table*}[!htbp]\\centering\\footnotesize\\caption{Three Measurements of Evasion at 1\\% FPR}\\label{tab:s-methods}")
w("\\begin{tabular}{@{}lll ccccc@{}}\\toprule Dataset & Model & Setting & Unconstrained & Repair & Repaired-flow violations & Realizable $\\bar{c}\\le1$ & Realizable $\\bar{c}\\le10$\\\\\\midrule")
first = True
for ds, dl in DS.items():
    if not first:
        w("\\midrule")
    first = False
    for m in MODELS:
        for st in ["STRICT", "LOOSE"]:
            q = P[(P["源"] == ds) & (P["模型"] == m) & (P["设定"] == st)]
            a = q[q["方式"].str.startswith("(A)")]
            b1 = q[q["方式"].str.startswith("(B)") & q["代价上限"].astype(str).str.contains("低")]
            b10 = q[q["方式"].str.startswith("(B)") & q["代价上限"].astype(str).str.contains("高")]
            w(f"{dl} & {m} & {'\\textsc{Strict}' if st == 'STRICT' else '\\textsc{Loose}'} & {ms(a['无约束成功率'])} & {ms(a['约束内成功率'])} & "
              f"{ms(a['成功样本违反率'])} & {ms(b1['约束内成功率'])} & {ms(b10['约束内成功率'])}\\\\")
w("\\bottomrule\\end{tabular}\\end{table*}")
w("\\clearpage")

# ---------------- S8: defenses, full ----------------
O = pd.read_csv(R + "O_summary_at.csv")
DF = ["D0 不防御", "D1 无约束增强", "D1′ 无约束对抗训练", "D2 约束内随机增强", "D3 约束内对抗训练"]
DFL = ["D0", "D1", "D1$'$", "D2", "D3"]
w("\\section{Defenses: Detection and Robustness at All Operating Points}\\label{sec:s-def}")
w("Table~\\ref{tab:s-def-auc} gives the AUROC of every defended model; Tables~\\ref{tab:s-def-0}--\\ref{tab:s-def-3} give the "
  "score-based evasion rate at $\\bar{c}\\le1$ at each operating point (mean $\\pm$ SD over three seeds). Thresholds are recalibrated "
  "for each defended model, and the attacker searches anew against it.")
g = O[(O["代价上限"] == "≤1×") & (O["工作点"] == "FPR 1.0%") & (O["设定"] == "STRICT")]
w("\\begin{table}[!htbp]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\\caption{AUROC Under Each Defense (Mean over Three Seeds)}\\label{tab:s-def-auc}")
w("\\begin{tabular}{@{}ll ccccc@{}}\\toprule Data & Model & " + " & ".join(DFL) + "\\\\\\midrule")
for ds, dl in DS.items():
    for m in MODELS:
        r = g[(g["源"] == ds) & (g["模型"] == m)].set_index("防御")
        w(f"{dl} & {m} & " + " & ".join(r.loc[d, "AUROC"].split("±")[0].strip() for d in DF) + "\\\\")
w("\\bottomrule\\end{tabular}\\end{table}")
for k, (op, ol) in enumerate(zip(OPS, OPL)):
    g = O[(O["代价上限"] == "≤1×") & (O["工作点"] == op)]
    w(f"\\begin{{table*}}[!htbp]\\centering\\footnotesize\\caption{{Score-Based Evasion Under Each Defense, {ol}, $\\bar{{c}}\\le1$}}\\label{{tab:s-def-{k}}}")
    w("\\begin{tabular}{@{}lll ccccc@{}}\\toprule Dataset & Model & Setting & " + " & ".join(DFL) + "\\\\\\midrule")
    first = True
    for ds, dl in DS.items():
        if not first:
            w("\\midrule")
        first = False
        for m in MODELS:
            for st in ["STRICT", "LOOSE"]:
                r = g[(g["源"] == ds) & (g["模型"] == m) & (g["设定"] == st)].set_index("防御")
                w(f"{dl} & {m} & {'\\textsc{Strict}' if st == 'STRICT' else '\\textsc{Loose}'} & " + " & ".join(pm(r.loc[d, "分数型成功率"]) for d in DF) + "\\\\")
    w("\\bottomrule\\end{tabular}\\end{table*}")


# ---------------- S9: ACK sensitivity ----------------
import os
if os.path.exists(R + "R_summary_ack.csv"):
    K = pd.read_csv(R + "R_summary_ack.csv")
    w("\\section{Sensitivity to Server Acknowledgments}\\label{sec:s-ack}")
    w("Table~\\ref{tab:s-ack} repeats the score-based attack when each added TCP segment elicits one 40-byte pure acknowledgment "
      "(ratio 1) or one per two segments (ratio 0.5), for flows whose server sends at least one packet. Backward packets and bytes, "
      "the smallest packet length, the packet-length histogram, the server ACK flag, backward throughput and backward "
      "inter-arrival times are updated accordingly; all reconstructed flows satisfy the constraints. Models, attack flows and random "
      "draws are identical to the main experiment, and ratio 0 reproduces it exactly. Over all 216 cells (all operating points and "
      "cost budgets) no success rate decreases by 0.01 or more. Values: mean $\\pm$ SD over three seeds, 1\\% FPR, $\\bar{c}\\le1$.")
    g = K[(K["工作点"] == "FPR 1.0%") & (K["代价上限"] == "≤1×")]
    w("\\begin{table}[!htbp]\\centering\\footnotesize\\caption{Evasion With Server Acknowledgments (Score-Based, 1\\% FPR, $\\bar{c}\\le1$)}\\label{tab:s-ack}")
    w("\\begin{tabular}{@{}lll ccc@{}}\\toprule Data & Model & Setting & No ACK (main) & Ratio 0.5 & Ratio 1\\\\\\midrule")

    def f(r, k):
        m, sd = r[f"分数型均值|r{k}"], r[f"分数型标准差|r{k}"]
        return f"{m:.3f}" if sd < 5e-4 else f"{m:.3f}\\,{{\\scriptsize$\\pm${sd:.3f}}}"
    first = True
    for ds, dl in DS.items():
        if not first:
            w("\\midrule")
        first = False
        for m in MODELS:
            for st in ["STRICT", "LOOSE"]:
                r = g[(g["源"] == ds) & (g["模型"] == m) & (g["设定"] == st)].iloc[0]
                w(f"{dl} & {m} & {'\\textsc{Strict}' if st == 'STRICT' else '\\textsc{Loose}'} & {f(r, '0')} & {f(r, '0.5')} & {f(r, '1')}\\\\")
    w("\\bottomrule\\end{tabular}\\end{table}")

body = "\n".join(L)
doc = r"""\documentclass[journal,onecolumn]{IEEEtran}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,booktabs,array}
\usepackage[hidelinks]{hyperref}
\renewcommand{\thesection}{S\arabic{section}}
\renewcommand{\thetable}{S\arabic{table}}
\begin{document}
\title{Supplementary Material for\\ ``Realizable Evasion of NetFlow Intrusion Detectors in Network Operations: Constraints, Leakage, and Operating Points''}
\author{Nanlei~Chen, Xiaoyu~Ke, Yifan~Hui, Huangze~Fang, Xuyan~Lu, and~Qiulong~Lin}
\maketitle
\sloppy
This document provides the complete results behind the summaries in the main paper. All numbers are generated directly from the
per-seed result files in the public repository (\url{https://github.com/B-Dawn/nfv3-realizable-robustness}, results folder).
Abbreviations: UNSW = NF-UNSW-NB15-v3, CIC = NF-CSE-CIC-IDS2018-v3, ToN = NF-ToN-IoT-v3; $\bar{c}$ = cost budget.
""" + body + "\n\\end{document}\n"
open(OUT, "w", encoding="utf-8").write(doc)
print("written", OUT, len(doc))
