from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from common import OUTPUT_ROOT, PROJECT_ROOT, clip, dump_json, load_sheets, minmax, write_template_csv


SEED = 20260830
N_SIM = 20_000


def standings_and_feedback(groups, live, schedule, predictions):
    teams = sorted(set(groups["team_a"]) | set(groups["team_b"]))
    stats = {t: {"games": 0, "points": 0, "gf": 0, "ga": 0, "xgf": 0.0, "xga": 0.0, "red": 0,
                 "injury": [], "att_ratio": [], "tv_ratio": []} for t in teams}
    pre_att = schedule.set_index("match_id")["expected_attendance"].to_dict()
    pre_tv = predictions.set_index("match_id")["predicted_tv_viewers"].to_dict()
    for _, r in live.iterrows():
        a, b = r["team_a"], r["team_b"]
        pa = 3 if r["goals_a"] > r["goals_b"] else 1 if r["goals_a"] == r["goals_b"] else 0
        pb = 3 if r["goals_b"] > r["goals_a"] else 1 if r["goals_a"] == r["goals_b"] else 0
        for t, pts, gf, ga, xgf, xga, red in [
            (a, pa, r["goals_a"], r["goals_b"], r["xg_a"], r["xg_b"], r["red_cards_a"]),
            (b, pb, r["goals_b"], r["goals_a"], r["xg_b"], r["xg_a"], r["red_cards_b"]),
        ]:
            s = stats[t]
            s["games"] += 1; s["points"] += int(pts); s["gf"] += int(gf); s["ga"] += int(ga)
            s["xgf"] += float(xgf); s["xga"] += float(xga); s["red"] += int(red)
            s["injury"].append(float(r["injury_impact_level"]))
            s["att_ratio"].append(float(np.clip(r["attendance"] / pre_att[r["match_id"]], 0.60, 1.50)))
            s["tv_ratio"].append(float(np.clip(r["tv_viewers"] / pre_tv[r["match_id"]], 0.60, 1.50)))
    for t, s in stats.items():
        g = s["games"]
        if g != 2:
            raise ValueError(f"Expected two completed matches for {t}, got {g}")
        s["h"] = float(np.clip(np.mean(s["injury"]), 0, 3))
        s["r_n"] = float(np.clip(np.mean(s["att_ratio"]), 0.75, 1.25))
        s["r_v"] = float(np.clip(np.mean(s["tv_ratio"]), 0.75, 1.25))
        s["state"] = float(
            0.45 * s["points"] / (3 * g)
            + 0.35 * (0.5 + 0.5 * np.tanh(((s["xgf"] - s["xga"]) / g) / 1.25))
            + 0.20 * np.exp(-0.55 * s["red"] / g)
        )
    return stats


def tie_block_weights(metrics: list[tuple[int, int, int]], cutoff_position: int) -> np.ndarray:
    """Fractional inclusion weights for lexicographic ranking with an exact-tie boundary."""
    n = len(metrics)
    order = sorted(range(n), key=lambda j: metrics[j], reverse=True)
    weights = np.zeros(n)
    pos = 0
    while pos < n:
        end = pos + 1
        while end < n and metrics[order[end]] == metrics[order[pos]]:
            end += 1
        slots = max(0, min(cutoff_position, end) - pos)
        if slots:
            for j in order[pos:end]:
                weights[j] = slots / (end - pos)
        pos = end
    return weights


def third_identity_weights(metrics: list[tuple[int, int, int]]) -> np.ndarray:
    order = sorted(range(len(metrics)), key=lambda j: metrics[j], reverse=True)
    pos = 0
    weights = np.zeros(len(metrics))
    while pos < len(order):
        end = pos + 1
        while end < len(order) and metrics[order[end]] == metrics[order[pos]]:
            end += 1
        if pos <= 2 < end:
            for j in order[pos:end]:
                weights[j] = 1.0 / (end - pos)
        pos = end
    return weights


def simulate_advancement(groups, base, stats):
    third = groups[groups["round_in_group"].eq(3)].copy().reset_index(drop=True)
    third = third.merge(base[["match_id", "expected_goals_a", "expected_goals_b"]], on="match_id", validate="one_to_one")
    lam_a, lam_b = [], []
    for _, r in third.iterrows():
        sa, sb = stats[r["team_a"]], stats[r["team_b"]]
        delta = 0.32 * (sa["state"] - sb["state"]) - 0.055 * (sa["h"] - sb["h"])
        lam_a.append(float(np.clip(r["expected_goals_a"] * np.exp(delta - 0.04 * sa["h"]), 0.15, 4.50)))
        lam_b.append(float(np.clip(r["expected_goals_b"] * np.exp(-delta - 0.04 * sb["h"]), 0.15, 4.50)))
    third["lambda_a"] = lam_a; third["lambda_b"] = lam_b
    rng = np.random.default_rng(SEED)
    goals_a = rng.poisson(np.asarray(lam_a), size=(N_SIM, len(third)))
    goals_b = rng.poisson(np.asarray(lam_b), size=(N_SIM, len(third)))

    team_names = sorted(stats)
    team_ix = {t: j for j, t in enumerate(team_names)}
    groups_by_id = {g: sorted([t for t in team_names if ((groups["group_id"].eq(g)) & (groups["team_a"].eq(t) | groups["team_b"].eq(t))).any()]) for g in sorted(groups["group_id"].unique())}
    third_game_ix = {g: list(third.index[third["group_id"].eq(g)]) for g in groups_by_id}
    advances = np.zeros((N_SIM, len(team_names)), dtype=np.float32)

    for sim in range(N_SIM):
        third_records = []
        group_payload = []
        for group_id, team_list in groups_by_id.items():
            pts = np.array([stats[t]["points"] for t in team_list], dtype=int)
            gf = np.array([stats[t]["gf"] for t in team_list], dtype=int)
            ga = np.array([stats[t]["ga"] for t in team_list], dtype=int)
            for game_ix in third_game_ix[group_id]:
                r = third.loc[game_ix]; ia = team_list.index(r["team_a"]); ib = team_list.index(r["team_b"])
                a, b = int(goals_a[sim, game_ix]), int(goals_b[sim, game_ix])
                gf[ia] += a; ga[ia] += b; gf[ib] += b; ga[ib] += a
                if a > b: pts[ia] += 3
                elif b > a: pts[ib] += 3
                else: pts[ia] += 1; pts[ib] += 1
            metrics = [(int(pts[j]), int(gf[j] - ga[j]), int(gf[j])) for j in range(4)]
            direct = tie_block_weights(metrics, 2)
            third_share = third_identity_weights(metrics)
            for j, t in enumerate(team_list):
                advances[sim, team_ix[t]] += direct[j]
            record_metric = sorted(metrics, reverse=True)[2]
            third_records.append(record_metric)
            group_payload.append((team_list, third_share))
        group_qual = tie_block_weights(third_records, 8)
        for g_j, (team_list, shares) in enumerate(group_payload):
            for j, t in enumerate(team_list):
                advances[sim, team_ix[t]] += shares[j] * group_qual[g_j]

    p = advances.mean(axis=0)
    p_by_team = {t: float(p[team_ix[t]]) for t in team_names}
    conditional = {}
    for j, r in third.iterrows():
        masks = {"draw": goals_a[:, j] == goals_b[:, j], "a_win": goals_a[:, j] > goals_b[:, j], "b_win": goals_b[:, j] > goals_a[:, j]}
        values = {}
        for label, mask in masks.items():
            values[label + "_n"] = int(mask.sum())
            values["p_a_" + label] = float(advances[mask, team_ix[r["team_a"]]].mean()) if mask.any() else np.nan
            values["p_b_" + label] = float(advances[mask, team_ix[r["team_b"]]].mean()) if mask.any() else np.nan
        conditional[r["match_id"]] = values
    return third, p_by_team, conditional, advances, goals_a, goals_b


def updated_match_state(third, p_by_team, conditional, stats, schedule, predictions, base, security):
    sched = schedule.set_index("match_id")
    pred = predictions.set_index("match_id")
    base_ix = base.set_index("match_id")
    sec_ix = security.set_index("match_id")
    rows = []
    for _, r in third.iterrows():
        a, b = r["team_a"], r["team_b"]
        pa, pb = p_by_team[a], p_by_team[b]
        Q = 0.5 * (4 * pa * (1-pa) + 4 * pb * (1-pb))
        f_n = float(np.sqrt(stats[a]["r_n"] * stats[b]["r_n"]))
        f_v = float(np.sqrt(stats[a]["r_v"] * stats[b]["r_v"]))
        S = 0.5 * (stats[a]["state"] + stats[b]["state"])
        H = (stats[a]["h"] + stats[b]["h"]) / 6.0
        A0 = float(base_ix.loc[r["match_id"], "attractiveness_index"]) / 100.0
        F = float(np.clip((0.5*f_n + 0.5*f_v - 0.75)/0.50, 0, 1))
        A = float(np.clip(0.50*A0 + 0.25*Q + 0.12*F + 0.08*S + 0.05*(1-H), 0, 1))
        Npre = float(sched.loc[r["match_id"], "expected_attendance"])
        Vpre = float(pred.loc[r["match_id"], "predicted_tv_viewers"])
        Ntilde = Npre*f_n*(0.88+0.27*Q)*(0.94+0.12*S)*(1-0.10*H)
        Vtilde = Vpre*f_v*(0.87+0.30*Q)*(0.90+0.20*A)*(1-0.06*H)
        stakeless = 0.5*(2*pa-1)**2 + 0.5*(2*pb-1)**2
        cc = conditional[r["match_id"]]
        D = min(cc["p_a_draw"], cc["p_b_draw"])
        G = 0.5*max(cc["p_a_a_win"]-cc["p_a_draw"], 0) + 0.5*max(cc["p_b_b_win"]-cc["p_b_draw"], 0)
        collusion = float(np.clip(D*(1-np.clip(G,0,1))*(1-0.35*stakeless),0,1))
        rows.append({
            "match_id": r["match_id"], "group_id": r["group_id"], "team_a": a, "team_b": b,
            "p_a": pa, "p_b": pb, "Q": Q, "f_n": f_n, "f_v": f_v, "S": S, "H": H,
            "A0": A0, "A": A, "Npre": Npre, "Vpre": Vpre, "Ntilde": Ntilde, "Vtilde": Vtilde,
            "uncertainty0": float(base_ix.loc[r["match_id"], "uncertainty_index"]),
            "stakeless": stakeless, "collusion": collusion,
            "d0": float(sec_ix.loc[r["match_id"], "security_demand_score"]),
            "required_security": int(sched.loc[r["match_id"], "required_security_level"]),
            "venue_id": sched.loc[r["match_id"], "venue_id"],
            "reference_date": str(sched.loc[r["match_id"], "reference_date"]),
        })
    return pd.DataFrame(rows)


def enumerate_shared_candidates(state, sheets):
    venues = sheets["venues"].set_index("venue_id")
    limits = sheets["dynamic_resource_limits"].set_index("reference_date")
    costs = sheets["dynamic_resource_costs"].set_index(["resource_type", "resource_level"])
    ticket = sheets["ticket_broadcast"].set_index("match_stage").loc["Group_Match_R3"]
    rows = []
    for _, r in state.iterrows():
        venue = venues.loc[r["venue_id"]]; K = float(venue["capacity"]); eps = float(ticket["price_elasticity"])
        lim = limits.loc[r["reference_date"]]
        deltas = np.round(np.arange(-float(lim["max_ticket_discount_rate"]), float(lim["max_ticket_increase_rate"])+0.0001, 0.01), 2)
        pre_stake = 1 - float(r["uncertainty0"])
        pre_d = float(np.clip(0.45*r["d0"] + 0.25*np.clip(r["Npre"]/K,0,1) + 0.15*r["A0"] + 0.15*pre_stake/2, 0, 1))
        dyn_d = float(np.clip(0.45*r["d0"] + 0.25*np.clip(r["Ntilde"]/K,0,1) + 0.15*r["A"] + 0.15*(r["stakeless"]+r["collusion"])/2, 0, 1))
        for b in [1,2,3]:
            for s in range(int(r["required_security"]), int(venue["security_level"])+1):
                for t in [1,2,3]:
                    cb = costs.loc[("broadcast", b)]; cs = costs.loc[("security", s)]; ct = costs.loc[("transport", t)]
                    mB=float(cb["demand_multiplier"]); mT=float(ct["demand_multiplier"]); mR=float(cs["risk_multiplier"])
                    n0_pre=min(K, r["Npre"]*mT); n0_dyn=min(K, r["Ntilde"]*mT)
                    for delta in deltas:
                        n_pre=min(K, r["Npre"]*mT*(1+delta)**eps)
                        n_dyn=min(K, r["Ntilde"]*mT*(1+delta)**eps)
                        if n_pre + 1e-9 < 0.88*n0_pre or n_dyn + 1e-9 < 0.88*n0_dyn:
                            continue
                        unit=float(ticket["broadcast_unit_value_usd"]); price=float(ticket["base_ticket_price_usd"])
                        cost=float(cb["unit_cost_index"]+cs["unit_cost_index"]+ct["unit_cost_index"])
                        rows.append({
                            "match_id":r["match_id"],"reference_date":r["reference_date"],"broadcast":b,"security":s,"transport":t,"delta":float(delta),
                            "cost":cost,
                            "pre_att":n_pre,"pre_tv":r["Vpre"]*mB,"pre_ticket":price*(1+delta)*n_pre,"pre_bv":unit*r["Vpre"]*mB,
                            "pre_A":r["A0"],"pre_R":0.4*pre_stake+0.2*pre_d*mR,
                            "dyn_att":n_dyn,"dyn_tv":r["Vtilde"]*mB,"dyn_ticket":price*(1+delta)*n_dyn,"dyn_bv":unit*r["Vtilde"]*mB,
                            "dyn_A":r["A"],"dyn_R":0.4*r["stakeless"]+0.4*r["collusion"]+0.2*dyn_d*mR,
                        })
    cand = pd.DataFrame(rows)
    bounds = {}
    for env in ["pre","dyn"]:
        for name in ["ticket","bv"]:
            col=f"{env}_{name}"; norm=f"{col}_n"; cand[norm],lo,hi=minmax(cand[col]); bounds[col]=[lo,hi]
        # The official P3 contract uses attractiveness and risk directly on
        # their bounded [0,1] scales; only ticket, broadcast and cost receive
        # candidate-set Min-Max normalization.
        for name in ["A","R"]:
            col=f"{env}_{name}"; cand[f"{col}_n"]=cand[col].clip(0.0,1.0)
            bounds[col]=[0.0,1.0]
    cand["cost_n"],lo,hi=minmax(cand["cost"]); bounds["cost"]=[lo,hi]
    return cand, bounds


def solve_resources(cand, env, sheets, time_limit=90):
    n=len(cand); c=np.zeros(n)
    c[:] = -(
        0.35*cand[f"{env}_ticket_n"] + 0.35*cand[f"{env}_bv_n"] + 0.10*cand[f"{env}_A_n"]
        - 0.10*cand["cost_n"] - 0.10*cand[f"{env}_R_n"]
    )
    rows=[];cols=[];vals=[];lo=[];hi=[];rid=0
    def add(idx,coef,l=-np.inf,h=np.inf):
        nonlocal rid
        rows.extend([rid]*len(idx));cols.extend(idx);vals.extend(coef);lo.append(l);hi.append(h);rid+=1
    for _, idx in cand.groupby("match_id").groups.items():
        ii=list(idx);add(ii,[1]*len(ii),1,1)
    limits=sheets["dynamic_resource_limits"].set_index("reference_date")
    for date, day in cand.groupby("reference_date"):
        lim=limits.loc[str(date)]
        idx=list(day.index)
        add(idx,day["broadcast"].eq(3).astype(int).tolist(),-np.inf,int(lim["high_broadcast_capacity"]))
        add(idx,day["security"].ge(3).astype(int).tolist(),-np.inf,int(lim["high_security_capacity"]))
        add(idx,day["transport"].eq(3).astype(int).tolist(),-np.inf,int(lim["enhanced_transport_capacity"]))
        add(idx,day["cost"].astype(float).tolist(),-np.inf,float(lim["daily_resource_budget_index"]))
    A=coo_matrix((vals,(rows,cols)),shape=(rid,n)).tocsr()
    res=milp(c=c,integrality=np.ones(n,dtype=int),bounds=Bounds(np.zeros(n),np.ones(n)),constraints=LinearConstraint(A,np.array(lo),np.array(hi)),options={"time_limit":time_limit,"mip_rel_gap":0.0005})
    if res.x is None: raise RuntimeError(f"P3 {env} optimization infeasible: {res.message}")
    chosen=cand.loc[res.x>0.5].copy()
    if len(chosen)!=cand["match_id"].nunique(): raise RuntimeError("P3 selection count invariant")
    return chosen, {"status":int(res.status),"message":res.message,"gap":float(getattr(res,"mip_gap",np.nan)),"nodes":int(getattr(res,"mip_node_count",-1))}


def contribution(row, env):
    return float(0.35*row[f"{env}_ticket_n"]+0.35*row[f"{env}_bv_n"]+0.10*row[f"{env}_A_n"]-0.10*row["cost_n"]-0.10*row[f"{env}_R_n"])


def run():
    sheets=load_sheets("groups_matches","base_predictions","live_group_results","venues","ticket_broadcast","security_requirements","dynamic_resource_limits","dynamic_resource_costs")
    schedule=pd.read_csv(OUTPUT_ROOT/"result_2_group_schedule.csv")
    predictions=pd.read_csv(OUTPUT_ROOT/"result_1_match_prediction.csv")
    stats=standings_and_feedback(sheets["groups_matches"],sheets["live_group_results"],schedule,predictions)
    third,p_by_team,conditional,advances,goals_a,goals_b=simulate_advancement(sheets["groups_matches"],sheets["base_predictions"],stats)
    if not np.allclose(advances.sum(axis=1),32,atol=2e-5): raise RuntimeError("Advancement mass invariant failed")
    state=updated_match_state(third,p_by_team,conditional,stats,schedule,predictions,sheets["base_predictions"],sheets["security_requirements"])
    cand,bounds=enumerate_shared_candidates(state,sheets)
    static,static_diag=solve_resources(cand,"pre",sheets)
    dynamic,dynamic_diag=solve_resources(cand,"dyn",sheets)
    static=static.set_index("match_id");dynamic=dynamic.set_index("match_id");state_ix=state.set_index("match_id")
    rows=[]
    for match_id in state["match_id"]:
        st=state_ix.loc[match_id]; sc=static.loc[match_id]; dy=dynamic.loc[match_id]
        static_value=contribution(sc,"dyn");dynamic_value=contribution(dy,"dyn")
        rows.append({
            "match_id":match_id,"group_id":st["group_id"],"team_a":st["team_a"],"team_b":st["team_b"],
            "updated_p_team_a_advance":round(float(st["p_a"]),6),"updated_p_team_b_advance":round(float(st["p_b"]),6),
            "updated_expected_attendance":int(round(dy["dyn_att"])),"updated_expected_tv_viewers":int(round(dy["dyn_tv"])),
            "stakeless_risk":round(float(st["stakeless"]),6),"collusion_risk":round(float(st["collusion"]),6),
            "updated_attractiveness":round(float(st["A"]),6),"recommended_broadcast_priority":int(dy["broadcast"]),
            "recommended_security_level":int(dy["security"]),"recommended_transport_level":int(dy["transport"]),
            "recommended_ticket_adjustment":round(float(dy["delta"]),4),"updated_ticket_revenue_usd":round(float(dy["dyn_ticket"]),2),
            "updated_broadcast_value_usd":round(float(dy["dyn_bv"]),2),"resource_cost_index":round(float(dy["cost"]),4),
            "risk_exposure_index":round(float(dy["dyn_R"]),6),"static_net_value":round(static_value,8),
            "dynamic_net_value":round(dynamic_value,8),"improvement_rate":round((dynamic_value-static_value)/abs(static_value),8) if abs(static_value)>1e-12 else np.nan,
        })
    out=pd.DataFrame(rows)
    path=write_template_csv(out,"result_3_template.csv","result_3_dynamic_strategy.csv")
    static_eval=sum(contribution(r,"dyn") for _,r in static.iterrows());dynamic_eval=sum(contribution(r,"dyn") for _,r in dynamic.iterrows())
    metrics={
        "seed":SEED,"simulations":N_SIM,"advance_mass_min":float(advances.sum(axis=1).min()),"advance_mass_max":float(advances.sum(axis=1).max()),
        "team_probability_range":[float(min(p_by_team.values())),float(max(p_by_team.values()))],"candidate_rows":len(cand),"normalization_bounds":bounds,
        "normalization_contract":{"ticket":"candidate-set Min-Max","broadcast":"candidate-set Min-Max","cost":"candidate-set Min-Max","attractiveness":"official raw [0,1]","risk":"official raw [0,1]"},
        "static_solver":static_diag,"dynamic_solver":dynamic_diag,"static_total_evaluated_updated":static_eval,"dynamic_total":dynamic_eval,
        "total_improvement_rate":float((dynamic_eval-static_eval)/abs(static_eval)),
        "mean_attendance_static_updated":float(static["dyn_att"].mean()),"mean_attendance_dynamic":float(dynamic["dyn_att"].mean()),
        "ticket_value_static_updated":float(static["dyn_ticket"].sum()),"ticket_value_dynamic":float(dynamic["dyn_ticket"].sum()),
        "broadcast_value_static_updated":float(static["dyn_bv"].sum()),"broadcast_value_dynamic":float(dynamic["dyn_bv"].sum()),
        "risk_static_updated":float(static["dyn_R"].mean()),"risk_dynamic":float(dynamic["dyn_R"].mean()),
        "resource_cost_static":float(static["cost"].sum()),"resource_cost_dynamic":float(dynamic["cost"].sum()),
        "output":str(path.relative_to(PROJECT_ROOT)),
    }
    dump_json(metrics,"q3_metrics.json")
    dump_json({"team_states":stats,"advance_probabilities":p_by_team,"conditional_probabilities":conditional},"q3_simulation_details.json")
    return metrics


if __name__=="__main__":
    print(json.dumps(run(),ensure_ascii=False,indent=2))
