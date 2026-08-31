from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import INPUT_ROOT, LOG_ROOT, OUTPUT_ROOT, PROJECT_ROOT, dump_json, load_sheets, locate_template, sha256


FILES = {
    "result_1_test_prediction.csv": ("result_1_test_prediction_template.csv", 140),
    "result_1_match_prediction.csv": ("result_1_match_prediction_template.csv", 72),
    "result_2_group_schedule.csv": ("result_2_template.csv", 72),
    "result_3_dynamic_strategy.csv": ("result_3_template.csv", 24),
    "actual_schedule_2022.csv": ("actual_schedule_template.csv", 48),
    "result_4_schedule_comparison.csv": ("result_4_template.csv", 11),
}

Q4_MANUSCRIPT_ROWS = {
    "avg_travel_km_per_transition": ("平均每次转场距离/km", ".4f", "实际"),
    "avg_rest_hours": ("平均休息/h", ".4f", "优化"),
    "timezone_crossings_per_team": ("每队跨时区次数", ".6f", "实际"),
    "venue_load_cv": ("场馆负荷变异系数", ".8f", "实际"),
    "mean_capacity_fill_rate": ("平均容量匹配率", ".8f", "实际（目标0.90）"),
    "global_prime_match_share": ("黄金时段比赛占比", ".8f", "实际"),
    "attendance_per_match": ("每场观众", ",.2f", "实际"),
    "ticket_revenue_usd_per_match": ("每场票务价值/美元", ",.2f", "实际"),
    "broadcast_value_usd_per_match": ("每场转播价值/美元", ",.2f", "优化"),
    "fairness_penalty": ("公平性惩罚", ".8f", "优化"),
    "execution_risk_proxy": ("执行风险代理", ".8f", "优化"),
}

CITATION_KEYS = {"breiman", "fifasec", "highs", "dixon", "ross", "openstax", "fifaapi", "nominatim"}


def header(path: Path):
    with path.open("r",encoding="utf-8-sig",newline="") as fh: return next(csv.reader(fh))


def run():
    failures=[];checks=[]
    def check(name,condition,detail=""):
        checks.append({"check":name,"passed":bool(condition),"detail":detail})
        if not condition: failures.append(name+(f": {detail}" if detail else ""))
    for name,(template,rows) in FILES.items():
        path=OUTPUT_ROOT/name
        check(f"exists:{name}",path.exists())
        if not path.exists(): continue
        check(f"header:{name}",header(path)==header(locate_template(template)))
        d=pd.read_csv(path)
        check(f"rows:{name}",len(d)==rows,f"expected {rows}, got {len(d)}")
        check(f"no_inf:{name}",not np.isinf(d.select_dtypes(include=[np.number]).to_numpy()).any())

    sheets=load_sheets("historical_matches","groups_matches","venues","time_slots","base_predictions","security_requirements","ticket_broadcast","dynamic_resource_limits","dynamic_resource_costs")
    p1t=pd.read_csv(OUTPUT_ROOT/"result_1_test_prediction.csv");p1m=pd.read_csv(OUTPUT_ROOT/"result_1_match_prediction.csv")
    hidden=set(sheets["historical_matches"].loc[sheets["historical_matches"]["dataset_split"].eq("test"),"match_id"])
    check("q1_test_ids",set(p1t["match_id_test"])==hidden);check("q1_test_positive",p1t["predicted_test_tv_viewers"].gt(0).all())
    match_ids=set(sheets["groups_matches"]["match_id"]);check("q1_match_ids",set(p1m["match_id"])==match_ids);check("q1_match_positive",p1m["predicted_tv_viewers"].gt(0).all())

    q2=pd.read_csv(OUTPUT_ROOT/"result_2_group_schedule.csv")
    check("q2_match_ids",set(q2["match_id"])==match_ids and q2["match_id"].is_unique)
    check("q2_venue_utc_unique",not q2.duplicated(["venue_id","utc_datetime"]).any())
    venues=sheets["venues"].set_index("venue_id");slots=sheets["time_slots"].set_index("slot_id")
    check("q2_security_eligibility",all(venues.loc[r.venue_id,"security_level"]>=r.required_security_level for r in q2.itertuples()))
    check("q2_city_country",all(venues.loc[r.venue_id,"city"]==r.city and venues.loc[r.venue_id,"country"]==r.country for r in q2.itertuples()))
    slot_counts=q2.groupby("slot_id").size();check("q2_broadcast_capacity",all(slot_counts[s]<=slots.loc[s,"broadcast_capacity"] for s in slot_counts.index))
    vc=q2.groupby("venue_id").size();check("q2_venue_totals",all(venues.loc[v,"min_total_matches"]<=vc.get(v,0)<=venues.loc[v,"max_total_matches"] for v in venues.index))
    q2local=q2.assign(local_date=q2["local_datetime"].astype(str).str.slice(0,10));vd=q2local.groupby(["venue_id","local_date"]).size()
    check("q2_venue_day",all(n<=venues.loc[v,"max_matches_per_day"] for (v,_),n in vd.items()))
    teams=sorted(set(q2.team_a)|set(q2.team_b));rests=[]
    for t in teams:
        g=q2[(q2.team_a.eq(t))|(q2.team_b.eq(t))].sort_values("round_in_group");rests.extend(pd.to_datetime(g.utc_datetime,utc=True).diff().dt.total_seconds().dropna()/3600)
        check(f"q2_rounds:{t}",list(g.round_in_group)==[1,2,3])
    check("q2_min_rest",min(rests)>=60,f"min={min(rests)}")
    prime=set(sheets["time_slots"].nlargest(int(np.ceil(len(slots)*0.25)),"global_prime_score").slot_id);pc=[]
    for t in teams: pc.append(int(q2[((q2.team_a.eq(t))|(q2.team_b.eq(t)))&q2.slot_id.isin(prime)].shape[0]))
    check("q2_prime_fairness",max(pc)-min(pc)<=2,f"range={min(pc)}-{max(pc)}")
    check("q2_objective_first_only",q2.total_objective_value.notna().sum()==1 and pd.notna(q2.iloc[0].total_objective_value))
    q2_metrics=json.loads((LOG_ROOT/"q2_metrics.json").read_text(encoding="utf-8"))
    q2_ind=q2_metrics["indicators_normalized"]
    q2_objective=0.25*q2_ind["T"]+0.25*q2_ind["B"]+0.15*q2_ind["U"]+0.10*q2_ind["H"]-0.08*q2_ind["C"]-0.07*q2_ind["D"]-0.06*q2_ind["F"]-0.04*q2_ind["R"]
    check("q2_official_objective_formula",abs(q2_objective-q2_metrics["objective_value"])<1e-12)
    check("q2_risk_uses_raw_unit_interval",abs(q2_ind["R"]-q2["risk_index"].mean())<2e-6 and 0<=q2_ind["R"]<=1,f"metric={q2_ind['R']}, csv_mean={q2['risk_index'].mean()}")
    check("q2_complete_cost",abs(q2_metrics["total_complete_cost_musd"]-q2_metrics["total_setup_cost_musd"]-q2_metrics["total_match_cost_musd"])<1e-10)
    check("q2_setup_cost_constant",abs(q2_metrics["total_setup_cost_musd"]-float(venues["setup_cost_musd"].sum()))<1e-10)
    check("q2_explicit_rest_constraints",q2_metrics.get("explicit_rest_constraints")==96,f"count={q2_metrics.get('explicit_rest_constraints')}")
    check("q2_gap_scope_disclosed","prefiltered" in q2_metrics.get("candidate_scope","").lower())

    q3=pd.read_csv(OUTPUT_ROOT/"result_3_dynamic_strategy.csv");third=set(sheets["groups_matches"].loc[sheets["groups_matches"].round_in_group.eq(3),"match_id"])
    check("q3_match_ids",set(q3.match_id)==third and q3.match_id.is_unique)
    for col in ["updated_p_team_a_advance","updated_p_team_b_advance","stakeless_risk","collusion_risk","updated_attractiveness"]:
        check(f"q3_unit_interval:{col}",q3[col].between(0,1).all())
    q3s=q3.merge(q2[["match_id","reference_date","venue_id","required_security_level"]],on="match_id");limits=sheets["dynamic_resource_limits"].set_index("reference_date")
    check("q3_security_bounds",all(r.required_security_level<=r.recommended_security_level<=venues.loc[r.venue_id,"security_level"] for r in q3s.itertuples()))
    costs=sheets["dynamic_resource_costs"].set_index(["resource_type","resource_level"])["unit_cost_index"].to_dict()
    for date,g in q3s.groupby("reference_date"):
        lim=limits.loc[str(date)];check(f"q3_broadcast_cap:{date}",(g.recommended_broadcast_priority==3).sum()<=lim.high_broadcast_capacity)
        check(f"q3_security_cap:{date}",(g.recommended_security_level>=3).sum()<=lim.high_security_capacity)
        check(f"q3_transport_cap:{date}",(g.recommended_transport_level==3).sum()<=lim.enhanced_transport_capacity)
        budget=sum(costs[("broadcast",int(r.recommended_broadcast_priority))]+costs[("security",int(r.recommended_security_level))]+costs[("transport",int(r.recommended_transport_level))] for r in g.itertuples())
        check(f"q3_budget:{date}",budget<=lim.daily_resource_budget_index+1e-9,f"{budget}/{lim.daily_resource_budget_index}")
    check("q3_dynamic_not_worse_total",q3.dynamic_net_value.sum()+1e-7>=q3.static_net_value.sum(),f"{q3.dynamic_net_value.sum()}/{q3.static_net_value.sum()}")
    q3_metrics=json.loads((LOG_ROOT/"q3_metrics.json").read_text(encoding="utf-8"))
    check("q3_attractiveness_raw_bounds",q3_metrics["normalization_bounds"].get("dyn_A")==[0.0,1.0] and q3_metrics["normalization_bounds"].get("pre_A")==[0.0,1.0])
    check("q3_risk_raw_bounds",q3_metrics["normalization_bounds"].get("dyn_R")==[0.0,1.0] and q3_metrics["normalization_bounds"].get("pre_R")==[0.0,1.0])
    check("q3_official_normalization_contract",q3_metrics.get("normalization_contract",{}).get("attractiveness")=="official raw [0,1]" and q3_metrics.get("normalization_contract",{}).get("risk")=="official raw [0,1]")

    actual=pd.read_csv(OUTPUT_ROOT/"actual_schedule_2022.csv");check("q4_actual_unique",actual.match_id.is_unique);check("q4_sources",actual.source_url.str.startswith("https://api.fifa.com/").all())
    q4=pd.read_csv(OUTPUT_ROOT/"result_4_schedule_comparison.csv")
    q4_audit=json.loads((LOG_ROOT/"q4_weighted_audit.json").read_text(encoding="utf-8"))
    expected_weights={"T":0.25,"B":0.25,"U":0.15,"H":0.10,"C":-0.08,"D":-0.07,"F":-0.06,"R":-0.04}
    check("q4_eleven_diagnostics",len(q4)==11 and q4_audit.get("diagnostic_indicator_count")==11)
    check("q4_eight_official_indicators",set(q4_audit.get("normalized_indicators",{}))==set(expected_weights) and q4_audit.get("indicator_count")==8)
    check("q4_official_weights",q4_audit.get("weights")==expected_weights)
    for schedule in ["actual","optimized"]:
        recomputed=sum(expected_weights[k]*q4_audit["normalized_indicators"][k][schedule] for k in expected_weights)
        check(f"q4_weighted_score:{schedule}",abs(recomputed-q4_audit["scores"][schedule])<1e-12)
    sensitivity=q4_audit.get("sensitivity",{})
    check("q4_proxy_sensitivity",sensitivity.get("scenario_count")==81 and sensitivity.get("frozen_bounds") is True and len(sensitivity.get("optimized_minus_actual_range",[]))==2)
    check("q4_common_bounds",set(q4_audit.get("shared_bounds",{}))=={"T","B","C","D_components"})

    manuscript_path=PROJECT_ROOT/"paper"/"main.tex"
    check("manuscript_source_exists",manuscript_path.exists())
    if manuscript_path.exists():
        manuscript=manuscript_path.read_text(encoding="utf-8")
        q4_by_name=q4.set_index("indicator_name")
        check("manuscript_q4_indicator_set",set(q4_by_name.index)==set(Q4_MANUSCRIPT_ROWS))
        for indicator,(label,number_format,result) in Q4_MANUSCRIPT_ROWS.items():
            if indicator not in q4_by_name.index:
                continue
            row=q4_by_name.loc[indicator]
            expected=(
                f"{label} & {float(row['actual_schedule_value']):{number_format}} & "
                f"{float(row['optimized_schedule_value']):{number_format}} & {result}\\\\"
            )
            check(f"manuscript_q4_row:{indicator}",expected in manuscript,expected)
        bibliography_marker="\\begin{thebibliography}"
        check("manuscript_bibliography_exists",bibliography_marker in manuscript)
        if bibliography_marker in manuscript:
            body=manuscript.split(bibliography_marker,1)[0]
            for key in sorted(CITATION_KEYS):
                check(f"manuscript_first_use_citation:{key}",f"\\cite{{{key}}}" in body)

    ai_pdf=OUTPUT_ROOT/"pdf"/"AI 工具使用详情.pdf"
    legacy_ai_pdf=OUTPUT_ROOT/"pdf"/"AI工具使用详情.pdf"
    check("formal_ai_pdf_exists",ai_pdf.exists(),str(ai_pdf.relative_to(PROJECT_ROOT)))
    check("legacy_ai_pdf_absent",not legacy_ai_pdf.exists(),str(legacy_ai_pdf.relative_to(PROJECT_ROOT)))
    manifest={name:sha256(OUTPUT_ROOT/name) for name in FILES if (OUTPUT_ROOT/name).exists()}
    report={"passed":not failures,"failures":failures,"checks":checks,"output_sha256":manifest}
    dump_json(report,"mechanical_validation.json")
    if failures: raise RuntimeError("Mechanical validation failed: "+"; ".join(failures))
    return report


if __name__=="__main__": print(json.dumps(run(),ensure_ascii=False,indent=2))
