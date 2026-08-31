from __future__ import annotations

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from common import FIGURE_ROOT, LOG_ROOT, OUTPUT_ROOT


mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei"],
    "axes.unicode_minus": False, "figure.dpi": 160, "savefig.dpi": 240,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE="#1f5a94";ORANGE="#e07a2f";GREEN="#2f7d60";GRAY="#6b7280"


def load_json(name): return json.loads((LOG_ROOT/name).read_text(encoding="utf-8"))


def q1_chart():
    m=load_json("q1_metrics.json");items=list(m["oof_mse_by_model_million2"].items())+[("非负加权集成",m["ensemble_oof_mse_million2"])]
    labels={"ridge":"岭回归","extra_trees":"极端随机树","random_forest":"随机森林","hist_gradient":"直方图梯度提升","非负加权集成":"非负加权集成"}
    names=[labels[x[0]] for x in items];values=[x[1] for x in items];colors=[GRAY]*4+[BLUE]
    fig,ax=plt.subplots(figsize=(7.2,3.6));bars=ax.barh(names,values,color=colors)
    ax.invert_yaxis();ax.set_xlabel("时间阻断 OOF MSE / (百万人)$^2$");ax.set_title("问题一候选模型与集成的时序泛化误差",loc="left",fontweight="bold")
    for b,v in zip(bars,values): ax.text(v+3,b.get_y()+b.get_height()/2,f"{v:.1f}",va="center",fontsize=9)
    ax.set_xlim(0,max(values)*1.17);ax.grid(axis="x",alpha=.18);fig.tight_layout();fig.savefig(FIGURE_ROOT/"q1_model_cv.png",bbox_inches="tight");plt.close(fig)


def q2_heatmap():
    d=pd.read_csv(OUTPUT_ROOT/"result_2_group_schedule.csv");table=pd.crosstab(d.venue_id,d.reference_date)
    dates=sorted(d.reference_date.unique());table=table.reindex(index=sorted(d.venue_id.unique()),columns=dates,fill_value=0)
    cmap=LinearSegmentedColormap.from_list("custom",["#f5f7fa","#9cc2e5",BLUE])
    fig,ax=plt.subplots(figsize=(10.2,5.0));im=ax.imshow(table.values,aspect="auto",cmap=cmap,vmin=0,vmax=max(2,table.values.max()))
    ax.set_yticks(range(len(table.index)),table.index);ax.set_xticks(range(len(dates)),[x[5:] for x in dates],rotation=45,ha="right")
    ax.set_xlabel("参考日期（月-日）");ax.set_ylabel("场馆");ax.set_title("问题二可行赛程的场馆-日期负荷",loc="left",fontweight="bold")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            if table.iat[i,j]: ax.text(j,i,str(table.iat[i,j]),ha="center",va="center",fontsize=7,color="white" if table.iat[i,j]>=2 else "#17324d")
    fig.colorbar(im,ax=ax,label="比赛场次",shrink=.78);fig.tight_layout();fig.savefig(FIGURE_ROOT/"q2_venue_date_load.png",bbox_inches="tight");plt.close(fig)


def q3_chart():
    m=load_json("q3_metrics.json")
    labels=["票务价值","转播价值","风险暴露","资源成本"]
    static=[m["ticket_value_static_updated"],m["broadcast_value_static_updated"],m["risk_static_updated"],m["resource_cost_static"]]
    dynamic=[m["ticket_value_dynamic"],m["broadcast_value_dynamic"],m["risk_dynamic"],m["resource_cost_dynamic"]]
    ratios=np.array(dynamic)/np.array(static)
    fig,ax=plt.subplots(figsize=(7.2,3.7));colors=[GREEN if (ratios[i]>=1 if i<2 else ratios[i]<=1) else ORANGE for i in range(4)]
    bars=ax.bar(labels,(ratios-1)*100,color=colors);ax.axhline(0,color="#333",lw=.8);ax.set_ylabel("动态相对静态变化 / %")
    ax.set_title("问题三：更新信息下动态资源方案的结构性变化",loc="left",fontweight="bold")
    for b,v in zip(bars,(ratios-1)*100): ax.text(b.get_x()+b.get_width()/2,v+(0.08 if v>=0 else -0.08),f"{v:+.2f}%",ha="center",va="bottom" if v>=0 else "top",fontsize=9)
    ax.grid(axis="y",alpha=.18);fig.tight_layout();fig.savefig(FIGURE_ROOT/"q3_dynamic_vs_static.png",bbox_inches="tight");plt.close(fig)


def q4_chart():
    d=pd.read_csv(OUTPUT_ROOT/"result_4_schedule_comparison.csv").set_index("indicator_name")
    audit=load_json("q4_weighted_audit.json")
    cost_names=["avg_travel_km_per_transition","fairness_penalty","execution_risk_proxy"]
    cost_labels=["旅行距离","公平性惩罚","风险代理"]
    benefit_names=["avg_rest_hours","broadcast_value_usd_per_match"]
    benefit_labels=["休息时间","每场转播价值"]
    fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(12.8,4.0),gridspec_kw={"width_ratios":[1,1,.9]})
    cost=[d.loc[n,"optimized_schedule_value"]/d.loc[n,"actual_schedule_value"] for n in cost_names]
    ax1.bar(cost_labels,cost,color=[ORANGE if x>1 else GREEN for x in cost]);ax1.axhline(1,color="#333",lw=.8);ax1.set_yscale("log");ax1.set_ylabel("优化/实际（对数轴；越低越好）")
    for i,v in enumerate(cost): ax1.text(i,v*1.08,f"{v:.2f}×",ha="center",fontsize=9)
    ben=[d.loc[n,"optimized_schedule_value"]/d.loc[n,"actual_schedule_value"] for n in benefit_names]
    ax2.bar(benefit_labels,ben,color=[GREEN if x>1 else ORANGE for x in ben]);ax2.axhline(1,color="#333",lw=.8);ax2.set_ylabel("优化/实际（越高越好）")
    for i,v in enumerate(ben): ax2.text(i,v+.025,f"{v:.2f}×",ha="center",fontsize=9)
    scores=audit["scores"];bars=[scores["actual"],scores["optimized"]]
    ax3.bar(["2022实际","模拟优化"],bars,color=[BLUE,GREEN],width=.58)
    diff_lo,diff_hi=audit["sensitivity"]["optimized_minus_actual_range"]
    actual_low=scores["optimized"]-diff_hi;actual_high=scores["optimized"]-diff_lo
    ax3.errorbar([0],[scores["actual"]],yerr=[[scores["actual"]-actual_low],[actual_high-scores["actual"]]],fmt="none",ecolor="#222",capsize=4,lw=1.2)
    for i,v in enumerate(bars): ax3.text(i,v+.008,f"{v:.3f}",ha="center",fontsize=9)
    ax3.set_ylabel("Q2 八项基准权重代理总分");ax3.set_title("误差线：81组代理敏感性",fontsize=10)
    fig.suptitle("问题四：2022 卡塔尔实际赛程与模拟优化赛程的取舍",fontweight="bold",x=.06,ha="left")
    for ax in [ax1,ax2,ax3]: ax.grid(axis="y",alpha=.18)
    fig.tight_layout();fig.savefig(FIGURE_ROOT/"q4_tradeoffs.png",bbox_inches="tight");plt.close(fig)


def run():
    FIGURE_ROOT.mkdir(parents=True,exist_ok=True);q1_chart();q2_heatmap();q3_chart();q4_chart()


if __name__=="__main__": run()
