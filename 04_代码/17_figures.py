"""17_figures.py：由 03_结果 中的汇总 CSV 生成论文图 3–5（矢量 PDF）。"""
import glob, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
from pathlib import Path
# 用法（项目根目录）：python 04_代码/17_figures.py [结果目录] [输出目录]
_ROOT = Path(__file__).resolve().parent.parent
R = (sys.argv[1] if len(sys.argv) > 1 else str(_ROOT / "03_结果")).rstrip("/\\") + "/"
O = (sys.argv[2] if len(sys.argv) > 2 else str(_ROOT / "06_写作" / "figs")).rstrip("/\\") + "/"
Path(O).mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],"font.size":8,"axes.titlesize":8,
 "axes.labelsize":8,"legend.fontsize":7,"xtick.labelsize":7,"ytick.labelsize":7,"pdf.fonttype":42,
 "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.6,"lines.linewidth":1.0})
C={"RF":"#0072B2","XGB":"#D55E00","MLP":"#009E73"}; M={"RF":"o","XGB":"s","MLP":"^"}
DS={"NF-UNSW-NB15-v3":"UNSW","NF-CSE-CIC-IDS2018-v3":"CIC","NF-ToN-IoT-v3":"ToN"}
# ---------- Fig.4 trade-off ----------
A=pd.read_csv(R+"M_summary_attack.csv"); A=A[(A["变体"]=="RAW")&(A["代价上限"]=="≤1×")]
wps=["FPR 0.1%","FPR 1.0%","FPR 5.0%","默认 0.5"]; wl=["0.1%","1%","5%","0.5"]
fig,ax=plt.subplots(2,3,figsize=(7.16,3.3),sharey=True,sharex=True)
for j,(ds,lab) in enumerate([("NF-UNSW-NB15-v3","UNSW"),("NF-CSE-CIC-IDS2018-v3","CIC"),("NF-ToN-IoT-v3","ToN")]):
  for i,st in enumerate(["STRICT","LOOSE"]):
    a=ax[i,j]
    for m in ["RF","XGB","MLP"]:
      g=A[(A["源"]==ds)&(A["设定"]==st)&(A["模型"]==m)].set_index("工作点").loc[wps]
      x=np.arange(4)+{"RF":-0.08,"XGB":0,"MLP":0.08}[m]
      a.errorbar(x,g["分数型均值"],yerr=g["分数型标准差"],color=C[m],marker=M[m],ms=3.5,capsize=2,elinewidth=0.6,label=m if (i,j)==(0,0) else None)
    a.set_title(f"{lab}, {st.lower()}"); a.set_xticks(range(4)); a.set_xticklabels(wl); a.set_ylim(-0.05,1.05)
    a.axvspan(2.5,3.5,color="0.92",zorder=0)
    a.grid(axis="y",lw=0.3,color="0.85")
fig.supxlabel("Operating point (target FPR on benign validation flows, or default threshold 0.5)",fontsize=8,y=0.03)
for a in ax[:,0]: a.set_ylabel("Evasion success rate")
fig.legend(loc="upper center",ncol=3,frameon=False,bbox_to_anchor=(0.5,1.02))
fig.tight_layout(rect=(0,0.04,1,0.95)); fig.savefig(O+"fig4_tradeoff.pdf"); plt.close(fig)
# ---------- Fig.3 measurement methods ----------
P=pd.concat([pd.read_csv(f) for f in glob.glob(R+"P_methods_*_s*.csv")])
A3=P[P["方式"].str.startswith("(A)")]; B3=P[P["方式"].str.startswith("(B)")&P["代价上限"].astype(str).str.contains("低")]
cfg=[("NF-UNSW-NB15-v3","RF","STRICT"),("NF-CSE-CIC-IDS2018-v3","XGB","STRICT"),("NF-CSE-CIC-IDS2018-v3","RF","LOOSE"),
     ("NF-UNSW-NB15-v3","MLP","LOOSE"),("NF-ToN-IoT-v3","XGB","STRICT"),("NF-CSE-CIC-IDS2018-v3","XGB","LOOSE"),("NF-CSE-CIC-IDS2018-v3","MLP","STRICT")]
rows=[]
for ds,m,st in cfg:
  a=A3[(A3["源"]==ds)&(A3["模型"]==m)&(A3["设定"]==st)]; b=B3[(B3["源"]==ds)&(B3["模型"]==m)&(B3["设定"]==st)]
  rows.append((f"{DS[ds]}\n{m}\n{st.lower()}",a["无约束成功率"].mean(),a["无约束成功率"].std(),a["约束内成功率"].mean(),a["约束内成功率"].std(),b["约束内成功率"].mean(),b["约束内成功率"].std()))
fig,a=plt.subplots(figsize=(7.16,2.1)); x=np.arange(len(rows)); w=0.26
cols=[("Unconstrained attack","#999999","///",1),("Attack then repair","#E69F00","..",3),("Realizable action search","#0072B2","",5)]
for k,(lab,c,h,idx) in enumerate(cols):
  a.bar(x+(k-1)*w,[r[idx] for r in rows],w,yerr=[r[idx+1] for r in rows],color=c,hatch=h,edgecolor="black",lw=0.4,capsize=1.5,error_kw={"elinewidth":0.5},label=lab)
a.set_xticks(x); a.set_xticklabels([r[0] for r in rows]); a.set_ylabel("Evasion success rate"); a.set_ylim(0,1.12)
a.axvline(2.5,color="0.5",lw=0.6,ls="--"); a.text(1.0,1.06,"overestimation",ha="center",fontsize=7,style="italic"); a.text(4.5,1.06,"underestimation",ha="center",fontsize=7,style="italic")
a.legend(ncol=3,frameon=False,loc="upper center",bbox_to_anchor=(0.5,1.28)); a.grid(axis="y",lw=0.3,color="0.85")
fig.tight_layout(); fig.savefig(O+"fig3_methods.pdf"); plt.close(fig)
# ---------- Fig.5 defense heatmap ----------
T=pd.read_csv(R+"O_summary_at.csv"); T=T[(T["代价上限"]=="≤1×")&(T["工作点"]=="FPR 1.0%")]
dfs=["D0 不防御","D1 无约束增强","D1′ 无约束对抗训练","D2 约束内随机增强","D3 约束内对抗训练"]; dl=["D0","D1","D1′","D2","D3"]
piv=T.pivot_table(index=["源","模型","设定"],columns="防御",values="分数型均值")[dfs]
order=[(d,m,s) for d in DS for m in ["RF","XGB","MLP"] for s in ["STRICT","LOOSE"]]
piv=piv.loc[order]
fig,a=plt.subplots(figsize=(3.5,4.4))
im=a.imshow(piv.values,cmap="Greys",vmin=0,vmax=1,aspect="auto")
for (r,c),v in np.ndenumerate(piv.values):
  a.text(c,r,f"{v:.2f}",ha="center",va="center",fontsize=6.5,color="white" if v>0.55 else "black")
a.set_xticks(range(5)); a.set_xticklabels(dl,fontsize=7.5); a.xaxis.tick_top()
a.set_yticks(range(len(order))); a.set_yticklabels([f"{DS[d]} {m} {s.lower()}" for d,m,s in order],fontsize=6.5)
for y in (5.5,11.5): a.axhline(y,color="#D55E00",lw=2.2)
for s in a.spines.values(): s.set_visible(False)
cb=fig.colorbar(im,ax=a,fraction=0.04,pad=0.02); cb.set_label("Evasion success rate (lower is better)",fontsize=7); cb.ax.tick_params(labelsize=6.5)
fig.tight_layout(); fig.savefig(O+"fig5_defenses.pdf"); plt.close(fig)
print("ok")
