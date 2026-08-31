from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from common import EXTERNAL_ROOT, OUTPUT_ROOT, PROJECT_ROOT, dump_json, load_sheets, minmax, write_template_csv
from q1_predict import build_features, fit_time_ensemble


FIFA_URL = "https://api.fifa.com/api/v3/calendar/matches?idSeason=255711&count=100&language=en"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
RETRIEVAL_DATE = "2026-08-30"
ALIASES = {"Korea Republic": "South Korea", "IR Iran": "Iran", "USA": "United States"}


def fetch_json(url: str, output: Path, user_agent="ModelingAgent/1.0 (academic reproducibility)"):
    if output.exists():
        return json.loads(output.read_text(encoding="utf-8"))
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=45) as response:
        data = json.loads(response.read().decode("utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def localized(items):
    if not items:
        return ""
    for item in items:
        if item.get("Locale") in ("en-GB", "en"):
            return item.get("Description", "")
    return items[0].get("Description", "")


def parse_fifa_schedule(payload):
    rows=[]
    all_matches=payload["Results"]
    capacity_proxy={}
    for m in all_matches:
        stadium=m.get("Stadium") or {}; sid=stadium.get("IdStadium")
        if sid and m.get("Attendance"):
            capacity_proxy[sid]=max(capacity_proxy.get(sid,0),int(m["Attendance"]))
    for m in all_matches:
        group=localized(m.get("GroupName"))
        if not group.startswith("Group "):
            continue
        stadium=m.get("Stadium") or {}
        utc=pd.Timestamp(m["Date"]).tz_convert("UTC")
        local=utc.tz_convert("Asia/Qatar")
        home=localized((m.get("Home") or {}).get("TeamName")); away=localized((m.get("Away") or {}).get("TeamName"))
        rows.append({
            "match_id":str(m["IdMatch"]),"competition":"FIFA World Cup Qatar 2022","stage":"Group",
            "group_id":group.replace("Group ",""),"round_in_group":0,"team_a":home,"team_b":away,
            "venue":localized(stadium.get("Name")),"venue_id":stadium.get("IdStadium"),"city":localized(stadium.get("CityName")),"country":"Qatar",
            "date":local.strftime("%Y-%m-%d"),"local_kickoff_time":local.strftime("%H:%M"),"utc_datetime":utc.strftime("%Y-%m-%d %H:%M:%SZ"),
            "goals_a":int(m["HomeTeamScore"]),"goals_b":int(m["AwayTeamScore"]),"source_url":FIFA_URL,"retrieval_date":RETRIEVAL_DATE,
            "attendance":int(m["Attendance"]),"capacity_proxy":capacity_proxy[stadium.get("IdStadium")],
        })
    df=pd.DataFrame(rows).sort_values(["group_id","utc_datetime"]).reset_index(drop=True)
    for _, idx in df.groupby("group_id").groups.items():
        order=list(idx)
        if len(order)!=6: raise RuntimeError("Official group-stage group does not contain six matches")
        df.loc[order,"round_in_group"]=[1,1,2,2,3,3]
    if len(df)!=48: raise RuntimeError(f"Expected 48 Qatar 2022 group matches, got {len(df)}")
    return df


def geocode_stadiums(actual):
    cache=EXTERNAL_ROOT/"stadium_geocodes.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    out={}
    for venue in sorted(actual["venue"].unique()):
        query=urllib.parse.urlencode({"format":"json","limit":1,"q":venue+" Qatar"})
        req=urllib.request.Request(NOMINATIM_URL+"?"+query,headers={"User-Agent":"ModelingAgent/1.0 (academic reproducibility)"})
        with urllib.request.urlopen(req,timeout=30) as response:
            records=json.loads(response.read().decode("utf-8"))
        if not records: raise RuntimeError(f"No OpenStreetMap geocode for {venue}")
        out[venue]={"lat":float(records[0]["lat"]),"lon":float(records[0]["lon"]),"display_name":records[0]["display_name"],"source_url":NOMINATIM_URL+"?"+query,"licence":records[0].get("licence")}
        time.sleep(1.05)
    cache.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    return out


def haversine(lat1,lon1,lat2,lon2):
    r=6371.0088
    p1,p2=np.radians(lat1),np.radians(lat2);dp=np.radians(lat2-lat1);dl=np.radians(lon2-lon1)
    a=np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return float(2*r*np.arcsin(np.sqrt(a)))


def transition_records(df, venue_key, utc_key, coords, timezone_map=None, distance_lookup=None, travel_time_lookup=None, proxy_speed_kmh=60.0):
    teams=sorted(set(df["team_a"])|set(df["team_b"]));records=[]
    for team in teams:
        g=df[(df["team_a"].eq(team))|(df["team_b"].eq(team))].copy()
        g[utc_key]=pd.to_datetime(g[utc_key],utc=True);g=g.sort_values(utc_key)
        if len(g)!=3: continue
        for i in [1,2]:
            v0,v1=g.iloc[i-1][venue_key],g.iloc[i][venue_key]
            if distance_lookup is None:
                distance=haversine(coords[v0]["lat"],coords[v0]["lon"],coords[v1]["lat"],coords[v1]["lon"])
            else:
                distance=float(distance_lookup[(v0,v1)])
            travel_time=(0.5+distance/proxy_speed_kmh) if travel_time_lookup is None else float(travel_time_lookup[(v0,v1)])
            records.append({"team":team,"distance_km":distance,"travel_time_hour":travel_time,
                            "timezone_diff":0.0 if timezone_map is None else float(timezone_map[v0]!=timezone_map[v1]),
                            "rest_hours":(g.iloc[i][utc_key]-g.iloc[i-1][utc_key]).total_seconds()/3600})
    return pd.DataFrame(records)


def transition_metrics(df, venue_key, utc_key, coords, timezone_map=None, distance_lookup=None, travel_time_lookup=None):
    records=transition_records(df,venue_key,utc_key,coords,timezone_map,distance_lookup,travel_time_lookup)
    teams=sorted(set(df["team_a"])|set(df["team_b"]))
    return float(records["distance_km"].mean()),float(records["rest_hours"].mean()),float(records["timezone_diff"].sum()/len(teams))


def actual_tv_predictions(actual, sheets):
    historical=sheets["historical_matches"].sort_values(["date","match_id"]).reset_index(drop=True)
    train=historical[historical["dataset_split"].eq("train")].copy()
    fit=fit_time_ensemble(build_features(train,sheets["teams"]),train["tv_viewers"].to_numpy(float)/1_000_000)
    teams=sheets["teams"].set_index("team_name"); median_elo=float(teams["elo_rating"].median());median_rank=float(teams["strength_rank"].median())
    rows=[];missing=[]
    for _,r in actual.iterrows():
        a=ALIASES.get(r["team_a"],r["team_a"]);b=ALIASES.get(r["team_b"],r["team_b"])
        if a not in teams.index or b not in teams.index: missing.extend([x for x in [a,b] if x not in teams.index])
        ea=float(teams.loc[a,"elo_rating"]) if a in teams.index else median_elo;eb=float(teams.loc[b,"elo_rating"]) if b in teams.index else median_elo
        pa=(1-0.26)/(1+10**(-(ea-eb)/400));pb=1-0.26-pa
        rows.append({"date":pd.Timestamp(r["date"]),"competition":"World Cup","stage":"Group","team_a":a,"team_b":b,"neutral":True,
                     "elo_a":ea,"elo_b":eb,"strength_rank_a":float(teams.loc[a,"strength_rank"]) if a in teams.index else median_rank,
                     "strength_rank_b":float(teams.loc[b,"strength_rank"]) if b in teams.index else median_rank,
                     "odds_a":1/max(pa,0.02),"odds_draw":1/0.26,"odds_b":1/max(pb,0.02)})
    raw=pd.DataFrame(rows);features=build_features(raw,sheets["teams"])

    # Q2's uncertainty is 1-max(pre-match outcome probability).  Actual-team
    # probabilities use the same transparent Elo/draw proxy as the TV model.
    uncertainty=features["outcome_uncertainty"].to_numpy(float)

    # FIFA does not publish the simulated attachment's attractiveness field.
    # Fit a deterministic linear bridge on the official 72-match table using
    # only pre-match uncertainty and team attributes, then apply it to Qatar.
    base=sheets["base_predictions"]
    team_table=sheets["teams"].set_index("team_name")
    medians={c:float(team_table[c].median()) for c in ["fan_base_index","star_index","strength_score"]}
    def team_value(name,column):
        return float(team_table.loc[name,column]) if name in team_table.index else medians[column]
    def design(frame, u):
        values=[]
        for (_,r),ui in zip(frame.iterrows(),u):
            a=ALIASES.get(r["team_a"],r["team_a"]);b=ALIASES.get(r["team_b"],r["team_b"])
            values.append([1.0,float(ui),0.5*(team_value(a,"fan_base_index")+team_value(b,"fan_base_index"))/100,
                           0.5*(team_value(a,"star_index")+team_value(b,"star_index"))/100,
                           0.5*(team_value(a,"strength_score")+team_value(b,"strength_score"))/100])
        return np.asarray(values,float)
    x_train=design(base,base["uncertainty_index"].to_numpy(float));y_train=base["attractiveness_index"].to_numpy(float)/100
    coef=np.linalg.lstsq(x_train,y_train,rcond=None)[0]
    attraction=np.clip(design(actual,uncertainty)@coef,0,1)
    attraction_diag={"method":"OLS bridge trained on official 72-match pre-match U and team attributes",
                     "coefficients":coef.tolist(),"training_rmse":float(np.sqrt(np.mean((x_train@coef-y_train)**2))),
                     "actual_range":[float(attraction.min()),float(attraction.max())]}
    return fit.predict(features)*1_000_000,sorted(set(missing)),uncertainty,attraction,attraction_diag


def common_minmax(actual_values, optimized_values):
    actual_values=np.asarray(actual_values,float);optimized_values=np.asarray(optimized_values,float)
    combined=np.r_[actual_values,optimized_values]
    normalized,lo,hi=minmax(combined)
    return normalized[:len(actual_values)],normalized[len(actual_values):],[lo,hi]


def weighted_q2_audit(actual, optimized, actual_transitions, optimized_transitions, actual_ticket_values,
                      actual_broadcast_values, actual_uncertainty, actual_attractiveness, actual_risk_values,
                      sheets, attraction_diag):
    venues=sheets["venues"].set_index("venue_id")
    setup_per_match=float(venues["setup_cost_musd"].sum()/72)
    optimized_cost=(optimized["venue_id"].map(venues["operation_cost_musd_per_match"]).to_numpy(float)
                    +0.1*optimized["required_security_level"].to_numpy(float)*optimized["venue_id"].map(venues["security_cost_index"]).to_numpy(float)
                    +setup_per_match)
    actual_cost_proxy=float(venues["operation_cost_musd_per_match"].median()+0.1*3*venues["security_cost_index"].median()+setup_per_match)
    actual_cost=np.full(len(actual),actual_cost_proxy)

    normalized={};bounds={}
    for key,a,o in [
        ("T",actual_ticket_values,optimized["ticket_revenue_usd"].to_numpy(float)),
        ("B",actual_broadcast_values,optimized["broadcast_value_usd"].to_numpy(float)),
        ("C",actual_cost,optimized_cost),
    ]:
        an,on,b=common_minmax(a,o);normalized[key]={"actual":float(an.mean()),"optimized":float(on.mean())};bounds[key]=b

    component_weights={"distance_km":0.5,"travel_time_hour":0.3,"timezone_diff":0.2}
    actual_d=np.zeros(len(actual_transitions));optimized_d=np.zeros(len(optimized_transitions));d_bounds={}
    for column,weight in component_weights.items():
        an,on,b=common_minmax(actual_transitions[column],optimized_transitions[column])
        actual_d+=weight*an;optimized_d+=weight*on;d_bounds[column]=b
    normalized["D"]={"actual":float(actual_d.mean()),"optimized":float(optimized_d.mean())};bounds["D_components"]=d_bounds

    base=sheets["base_predictions"]
    normalized["U"]={"actual":float(np.mean(actual_uncertainty)),"optimized":float(base["uncertainty_index"].mean())}
    normalized["H"]={"actual":float(np.mean(actual_attractiveness)),"optimized":float((base["attractiveness_index"]/100).mean())}
    normalized["F"]={"actual":fairness(actual,"is_prime","is_big"),"optimized":fairness(optimized,"is_prime","is_big")}
    normalized["R"]={"actual":float(np.mean(actual_risk_values)),"optimized":float(optimized["risk_index"].mean())}

    official=sheets["objective_weights"]
    official=official[official["problem_id"].eq("P2")]
    name_to_symbol={"ticket_revenue":"T","broadcast_value":"B","uncertainty_index":"U","attractiveness_index":"H",
                    "operation_cost":"C","travel_burden":"D","fairness_penalty":"F","execution_risk":"R"}
    weights={name_to_symbol[r.indicator_name]:float(r.weight)*(1 if r.direction=="maximize" else -1) for r in official.itertuples()}
    expected={"T":0.25,"B":0.25,"U":0.15,"H":0.10,"C":-0.08,"D":-0.07,"F":-0.06,"R":-0.04}
    if weights!=expected: raise RuntimeError(f"Unexpected official P2 weights: {weights}")
    def score(values): return float(sum(weights[k]*values[k] for k in expected))
    actual_values={k:v["actual"] for k,v in normalized.items()};optimized_values={k:v["optimized"] for k,v in normalized.items()}
    scores={"actual":score(actual_values),"optimized":score(optimized_values)}

    # Predeclared proxy sensitivity grid. All scenarios retain the baseline
    # shared bounds, so only the proxy assumptions change. Values outside a
    # frozen empirical bound are capped to [0,1].
    scenarios=[]
    c_lo,c_hi=bounds["C"]
    time_lo,time_hi=bounds["D_components"]["travel_time_hour"]
    base_actual_distance=actual_transitions["distance_km"].to_numpy(float)
    base_d_without_time=np.zeros(len(actual_transitions))
    for column,weight in {"distance_km":0.5,"timezone_diff":0.2}.items():
        lo,hi=bounds["D_components"][column]
        part=np.zeros(len(actual_transitions)) if abs(hi-lo)<1e-12 else np.clip((actual_transitions[column].to_numpy(float)-lo)/(hi-lo),0,1)
        base_d_without_time+=weight*part
    for cost_multiplier in [0.75,1.0,1.25]:
        for h_shift in [-0.08,0.0,0.08]:
            for risk_shift in [-0.10,0.0,0.10]:
                for speed in [40.0,60.0,80.0]:
                    c_raw=actual_cost*cost_multiplier
                    c_n=np.zeros(len(c_raw)) if abs(c_hi-c_lo)<1e-12 else np.clip((c_raw-c_lo)/(c_hi-c_lo),0,1)
                    t_raw=0.5+base_actual_distance/speed
                    t_n=np.zeros(len(t_raw)) if abs(time_hi-time_lo)<1e-12 else np.clip((t_raw-time_lo)/(time_hi-time_lo),0,1)
                    values=dict(actual_values);values["C"]=float(c_n.mean());values["D"]=float((base_d_without_time+0.3*t_n).mean())
                    values["H"]=float(np.clip(actual_attractiveness+h_shift,0,1).mean())
                    values["R"]=float(np.clip(actual_risk_values+risk_shift,0,1).mean())
                    actual_score=score(values);difference=scores["optimized"]-actual_score
                    scenarios.append({"cost_multiplier":cost_multiplier,"attractiveness_shift":h_shift,"risk_shift":risk_shift,
                                      "actual_travel_speed_kmh":speed,"actual_score":actual_score,"optimized_minus_actual":difference})
    minimum=min(scenarios,key=lambda x:x["optimized_minus_actual"]);maximum=max(scenarios,key=lambda x:x["optimized_minus_actual"])
    sensitivity={"scenario_count":len(scenarios),"frozen_bounds":True,
                 "optimized_minus_actual_range":[minimum["optimized_minus_actual"],maximum["optimized_minus_actual"]],
                 "ranking_flips_within_grid":minimum["optimized_minus_actual"]*maximum["optimized_minus_actual"]<0,
                 "minimum_difference_scenario":minimum,"maximum_difference_scenario":maximum}
    return {"indicator_count":8,"diagnostic_indicator_count":11,"weights":weights,"normalized_indicators":normalized,
            "shared_bounds":bounds,"scores":scores,"optimized_minus_actual":scores["optimized"]-scores["actual"],
            "aggregation":{"T_B_C":"per-match values normalized over the combined 48+72 records, then averaged",
                           "U_H_R":"official/proxy raw values already bounded to [0,1], then averaged",
                           "D":"two adjacent venue transitions per team; components jointly Min-Max normalized across both tournaments",
                           "F":"official schedule-level range formula"},
            "proxies":{"U":"1-max(Elo/draw pre-match probabilities)","H":attraction_diag,
                       "C":f"Qatar venue cost unavailable; fixed candidate-median match proxy plus {setup_per_match:.8f} MUSD setup allocation",
                       "D":"Qatar venue-to-venue great-circle distance and 0.5 h + distance/60 km/h travel-time proxy; first-leg origin unavailable",
                       "R":"candidate median climate/security capability with observed occupancy, as in the 11-item diagnostic"},
            "sensitivity":sensitivity,"claim_boundary":"weighted score is a disclosed proxy comparison, not causal evidence or a proof of real-world superiority"}


def prime_score_for_utc(ts, slot_scores):
    hour=ts.hour+ts.minute/60
    by_hour=slot_scores.assign(hour=pd.to_datetime(slot_scores["reference_utc_time"]).dt.hour+pd.to_datetime(slot_scores["reference_utc_time"]).dt.minute/60).groupby("hour")["global_prime_score"].mean()
    def cyclic(h): return min(abs(h-hour),24-abs(h-hour))
    nearest=min(by_hour.index,key=cyclic)
    return float(by_hour.loc[nearest])


def fairness(df,prime_col,big_col):
    teams=sorted(set(df["team_a"])|set(df["team_b"]));pc=[];bc=[]
    for t in teams:
        g=df[(df["team_a"].eq(t))|(df["team_b"].eq(t))];pc.append(int(g[prime_col].sum()));bc.append(int(g[big_col].sum()))
    return 0.5*(max(pc)-min(pc))/3+0.5*(max(bc)-min(bc))/3


def result_row(name,category,actual,optimized,direction,target=None):
    diff=optimized-actual
    if direction=="higher": rel=diff/abs(actual) if actual else np.nan;better=diff>0
    elif direction=="lower": rel=(actual-optimized)/abs(actual) if actual else np.nan;better=diff<0
    else:
        da,do=abs(actual-target),abs(optimized-target);rel=(da-do)/da if da else np.nan;better=do<da
    conclusion="优化方案更优" if better else "实际赛程更优" if (abs(diff)>1e-12 if direction!="target" else abs(abs(actual-target)-abs(optimized-target))>1e-12) else "基本持平"
    if target is not None: conclusion+=f"（目标={target:g}）"
    return {"indicator_name":name,"indicator_category":category,"actual_schedule_value":round(actual,8),"optimized_schedule_value":round(optimized,8),
            "absolute_difference":round(diff,8),"relative_improvement":round(rel,8) if np.isfinite(rel) else np.nan,"preferred_direction":direction,"evaluation_result":conclusion}


def run():
    EXTERNAL_ROOT.mkdir(parents=True,exist_ok=True)
    payload=fetch_json(FIFA_URL,EXTERNAL_ROOT/"fifa_2022_schedule.json")
    actual=parse_fifa_schedule(payload);geo=geocode_stadiums(actual)
    actual_public=actual[["match_id","competition","stage","group_id","round_in_group","team_a","team_b","venue","city","country","date","local_kickoff_time","utc_datetime","goals_a","goals_b","source_url","retrieval_date"]]
    actual_path=write_template_csv(actual_public,"actual_schedule_template.csv","actual_schedule_2022.csv")
    sheets=load_sheets("historical_matches","teams","venues","time_slots","distance_matrix","ticket_broadcast","security_requirements","base_predictions","objective_weights")
    optimized=pd.read_csv(OUTPUT_ROOT/"result_2_group_schedule.csv")
    tv_actual,missing,actual_uncertainty,actual_attractiveness,attraction_diag=actual_tv_predictions(actual,sheets);actual["predicted_tv"]=tv_actual
    threshold=float(sheets["time_slots"]["global_prime_score"].quantile(0.75,interpolation="lower"))
    actual["prime_score"]=[prime_score_for_utc(pd.Timestamp(x),sheets["time_slots"]) for x in actual["utc_datetime"]];actual["is_prime"]=(actual["prime_score"]>=threshold).astype(int)
    venue_caps=actual.groupby("venue")["capacity_proxy"].max();big_actual=set(venue_caps.nlargest(2).index);actual["is_big"]=actual["venue"].isin(big_actual).astype(int)
    venues=sheets["venues"].set_index("venue_id");slot_score=sheets["time_slots"].set_index("slot_id")["global_prime_score"]
    prime_slot_ids=set(sheets["time_slots"].nlargest(int(np.ceil(len(sheets["time_slots"])*0.25)),"global_prime_score")["slot_id"])
    optimized["is_prime"]=optimized["slot_id"].isin(prime_slot_ids).astype(int);big_opt=set(venues.nlargest(4,"capacity").index);optimized["is_big"]=optimized["venue_id"].isin(big_opt).astype(int)
    dist=sheets["distance_matrix"];venue_dist=dist[dist["relation_type"].eq("venue_to_venue")]
    vv=venue_dist.set_index(["origin_id","destination_id"])["distance_km"].to_dict();vv_time=venue_dist.set_index(["origin_id","destination_id"])["travel_time_hour"].to_dict()
    actual_travel,actual_rest,actual_tz=transition_metrics(actual,"venue","utc_datetime",geo)
    opt_travel,opt_rest,opt_tz=transition_metrics(optimized,"venue_id","utc_datetime",None,venues["timezone"].to_dict(),vv,vv_time)
    actual_transitions=transition_records(actual,"venue","utc_datetime",geo)
    optimized_transitions=transition_records(optimized,"venue_id","utc_datetime",None,venues["timezone"].to_dict(),vv,vv_time)
    actual_cv=float(actual.groupby("venue").size().std(ddof=0)/actual.groupby("venue").size().mean());opt_cv=float(optimized.groupby("venue_id").size().std(ddof=0)/optimized.groupby("venue_id").size().mean())
    actual_fill=float((actual["attendance"]/actual["capacity_proxy"]).mean());opt_fill=float((optimized["expected_attendance"]/optimized["venue_id"].map(venues["capacity"])).mean())
    params=sheets["ticket_broadcast"].set_index("match_stage")
    actual_ticket_values=[];actual_broadcast_values=[]
    for _,r in actual.iterrows():
        p=params.loc[f"Group_Match_R{int(r['round_in_group'])}"];actual_ticket_values.append(r["attendance"]*p["base_ticket_price_usd"])
        actual_broadcast_values.append(r["predicted_tv"]*p["broadcast_unit_value_usd"]*r["prime_score"]*p["sponsor_weight"])
    actual_ticket_values=np.asarray(actual_ticket_values,float);actual_broadcast_values=np.asarray(actual_broadcast_values,float)
    # Risk proxy: unavailable climate/security attributes fixed at candidate medians; occupancy remains schedule-specific.
    median_climate=float(venues["climate_risk"].median());median_ratio=float((optimized["required_security_level"]/optimized["venue_id"].map(venues["security_level"])).median())
    actual_risk_values=(0.5*median_climate+0.3*(actual["attendance"]/actual["capacity_proxy"])+0.2*median_ratio).to_numpy(float)
    actual_risk=float(actual_risk_values.mean())
    opt_risk=float(optimized["risk_index"].mean())
    records=[
        result_row("avg_travel_km_per_transition","travel",actual_travel,opt_travel,"lower"),result_row("avg_rest_hours","rest",actual_rest,opt_rest,"higher"),
        result_row("timezone_crossings_per_team","travel",actual_tz,opt_tz,"lower"),result_row("venue_load_cv","venue_use",actual_cv,opt_cv,"lower"),
        result_row("mean_capacity_fill_rate","capacity_match",actual_fill,opt_fill,"target",0.90),result_row("global_prime_match_share","broadcast",float(actual["is_prime"].mean()),float(optimized["is_prime"].mean()),"higher"),
        result_row("attendance_per_match","audience",float(actual["attendance"].mean()),float(optimized["expected_attendance"].mean()),"higher"),
        result_row("ticket_revenue_usd_per_match","commercial",float(actual_ticket_values.mean()),float(optimized["ticket_revenue_usd"].mean()),"higher"),
        result_row("broadcast_value_usd_per_match","commercial",float(actual_broadcast_values.mean()),float(optimized["broadcast_value_usd"].mean()),"higher"),
        result_row("fairness_penalty","fairness",fairness(actual,"is_prime","is_big"),fairness(optimized,"is_prime","is_big"),"lower"),
        result_row("execution_risk_proxy","risk",actual_risk,opt_risk,"lower"),
    ]
    out=pd.DataFrame(records);result_path=write_template_csv(out,"result_4_template.csv","result_4_schedule_comparison.csv")
    weighted=weighted_q2_audit(actual,optimized,actual_transitions,optimized_transitions,actual_ticket_values,actual_broadcast_values,
                               actual_uncertainty,actual_attractiveness,actual_risk_values,sheets,attraction_diag)
    metrics={"fifa_source_url":FIFA_URL,"retrieval_date":RETRIEVAL_DATE,"actual_matches":len(actual),"optimized_matches":len(optimized),"missing_team_attributes":missing,
             "prime_threshold":threshold,"capacity_proxy":"maximum official FIFA attendance at each stadium across the tournament",
             "risk_proxy_limitation":"Actual climate and security capability are unavailable in FIFA endpoint; fixed candidate medians isolate occupancy.",
             "actual_schedule_output":str(actual_path.relative_to(PROJECT_ROOT)),"comparison_output":str(result_path.relative_to(PROJECT_ROOT)),"indicators":records,
             "weighted_q2_audit":"logs/q4_weighted_audit.json","weighted_scores":weighted["scores"],"weighted_score_sensitivity":weighted["sensitivity"]}
    dump_json(weighted,"q4_weighted_audit.json");dump_json(metrics,"q4_metrics.json");return metrics


if __name__=="__main__": print(json.dumps(run(),ensure_ascii=False,indent=2))
