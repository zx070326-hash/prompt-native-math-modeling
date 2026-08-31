from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from common import OUTPUT_ROOT, PROJECT_ROOT, dump_json, load_sheets, minmax, write_template_csv


ROUND_DATES = {
    1: set(pd.date_range("2026-06-11", "2026-06-15").strftime("%Y-%m-%d")),
    2: set(pd.date_range("2026-06-18", "2026-06-22").strftime("%Y-%m-%d")),
    3: set(pd.date_range("2026-06-26", "2026-06-30").strftime("%Y-%m-%d")),
}


def prepare_travel(distance: pd.DataFrame):
    d = distance.copy()
    for col in ["distance_km", "travel_time_hour", "timezone_diff"]:
        d[col + "_n"], _, _ = minmax(d[col].to_numpy(float))
    d["burden"] = 0.5 * d["distance_km_n"] + 0.3 * d["travel_time_hour_n"] + 0.2 * d["timezone_diff_n"]
    first = d[d["relation_type"].eq("team_to_venue")].set_index(["origin_id", "destination_id"])["burden"].to_dict()
    later = d[d["relation_type"].eq("venue_to_venue")].set_index(["origin_id", "destination_id"])["burden"].to_dict()
    raw_first_distance = d[d["relation_type"].eq("team_to_venue")].set_index(["origin_id", "destination_id"])["distance_km"].to_dict()
    raw_later_distance = d[d["relation_type"].eq("venue_to_venue")].set_index(["origin_id", "destination_id"])["distance_km"].to_dict()
    return first, later, raw_first_distance, raw_later_distance


def travel_for_match_venue(match, venue_id, groups, security, venues, first, later):
    burdens = []
    for side in ["a", "b"]:
        team_id = match[f"team_{side}_id"]
        if match["round_in_group"] == 1:
            burdens.append(first[(team_id, venue_id)])
        else:
            previous = groups[
                (groups["round_in_group"] == match["round_in_group"] - 1)
                & ((groups["team_a_id"] == team_id) | (groups["team_b_id"] == team_id))
            ].iloc[0]
            prev_req = int(security.loc[security["match_id"].eq(previous["match_id"]), "required_security_level"].iloc[0])
            eligible_origins = venues.loc[venues["security_level"] >= prev_req, "venue_id"].tolist()
            burdens.append(float(np.mean([later[(origin, venue_id)] for origin in eligible_origins])))
    return float(np.mean(burdens))


def build_candidates(sheets: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    groups = sheets["groups_matches"].copy()
    venues = sheets["venues"].copy()
    slots = sheets["time_slots"].copy()
    base = sheets["base_predictions"].copy()
    sec = sheets["security_requirements"].copy()
    pred = pd.read_csv(OUTPUT_ROOT / "result_1_match_prediction.csv")
    params = sheets["ticket_broadcast"].set_index("match_stage")
    first, later, _, _ = prepare_travel(sheets["distance_matrix"])

    m = groups.merge(base, on=["match_id", "group_id", "round_in_group", "team_a", "team_b"], validate="one_to_one")
    m = m.merge(sec[["match_id", "security_demand_score", "required_security_level"]], on="match_id", validate="one_to_one")
    m = m.merge(pred[["match_id", "predicted_tv_viewers"]], on="match_id", validate="one_to_one")
    prime_ids = set(slots.nlargest(int(np.ceil(len(slots) * 0.25)), "global_prime_score")["slot_id"])
    big_ids = set(venues.nlargest(int(np.ceil(len(venues) * 0.25)), "capacity")["venue_id"])

    rows = []
    for _, match in m.iterrows():
        r = int(match["round_in_group"])
        stage = f"Group_Match_R{r}"
        p = params.loc[stage]
        for _, venue in venues[venues["security_level"] >= match["required_security_level"]].iterrows():
            travel = travel_for_match_venue(match, venue["venue_id"], groups, sec, venues, first, later)
            expected_attendance = min(float(match["expected_attendance_base"]), float(venue["capacity"]))
            ticket = expected_attendance * float(p["base_ticket_price_usd"])
            cost = float(venue["operation_cost_musd_per_match"]) + 0.1 * float(match["required_security_level"]) * float(venue["security_cost_index"])
            risk = (
                0.5 * float(venue["climate_risk"])
                + 0.3 * expected_attendance / float(venue["capacity"])
                + 0.2 * float(match["required_security_level"]) / float(venue["security_level"])
            )
            for _, slot in slots[slots["date"].astype(str).isin(ROUND_DATES[r])].iterrows():
                utc = pd.Timestamp(slot["reference_utc_time"], tz="UTC")
                local = utc.tz_convert(ZoneInfo(str(venue["timezone"])))
                broadcast = (
                    float(match["predicted_tv_viewers"])
                    * float(p["broadcast_unit_value_usd"])
                    * float(slot["global_prime_score"])
                    * float(p["sponsor_weight"])
                )
                rows.append({
                    "match_id": match["match_id"], "group_id": match["group_id"], "round_in_group": r,
                    "team_a": match["team_a"], "team_b": match["team_b"],
                    "venue_id": venue["venue_id"], "slot_id": slot["slot_id"],
                    "utc_datetime": utc, "local_datetime": local,
                    "local_date": local.strftime("%Y-%m-%d"), "reference_date": str(slot["date"]),
                    "ticket": ticket, "broadcast": broadcast,
                    "uncertainty": float(match["uncertainty_index"]),
                    "attractiveness": float(match["attractiveness_index"]) / 100.0,
                    "cost": cost, "travel": travel, "risk": risk,
                    "expected_attendance": expected_attendance,
                    "predicted_tv_viewers": int(match["predicted_tv_viewers"]),
                    "required_security_level": int(match["required_security_level"]),
                    "prime": int(slot["slot_id"] in prime_ids),
                    "big": int(venue["venue_id"] in big_ids),
                    "global_prime_score": float(slot["global_prime_score"]),
                    "broadcast_capacity": int(slot["broadcast_capacity"]),
                })
    candidates = pd.DataFrame(rows)
    # All venues have a strictly positive minimum match count, so their setup
    # cost is a 17.95 MUSD constant over the feasible region. Allocate that
    # constant evenly across the 72 matches before normalization: this keeps
    # the complete official cost definition while preserving the ranking.
    setup_total = float(venues["setup_cost_musd"].sum())
    if not venues["min_total_matches"].gt(0).all():
        raise RuntimeError("The constant setup-cost argument requires every venue to be used")
    candidates["complete_cost_equivalent"] = candidates["cost"] + setup_total / groups["match_id"].nunique()
    for raw, norm in [("ticket", "ticket_n"), ("broadcast", "broadcast_n"), ("complete_cost_equivalent", "cost_n"), ("travel", "travel_n")]:
        candidates[norm], _, _ = minmax(candidates[raw].to_numpy(float))
    # The official workbook specifies execution_risk as an already bounded
    # [0,1] value. It must not undergo a second candidate-set Min-Max.
    candidates["risk_n"] = candidates["risk"].clip(0.0, 1.0)
    meta = {"prime_slot_ids": sorted(prime_ids), "big_venue_ids": sorted(big_ids)}
    return candidates, meta


def solve(candidates: pd.DataFrame, sheets: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    n = len(candidates)
    # Continuous tail variables: prime max/min and large-venue max/min counts.
    pmax, pmin, bmax, bmin = n, n + 1, n + 2, n + 3
    nvar = n + 4
    c = np.zeros(nvar)
    quality = (
        0.25 * candidates["ticket_n"] + 0.25 * candidates["broadcast_n"]
        + 0.15 * candidates["uncertainty"] + 0.10 * candidates["attractiveness"]
        - 0.08 * candidates["cost_n"] - 0.07 * candidates["travel_n"]
        - 0.04 * candidates["risk_n"]
    ) / 72.0
    c[:n] = -quality.to_numpy(float)
    c[pmax], c[pmin], c[bmax], c[bmin] = 0.01, -0.01, 0.01, -0.01

    rows, cols, vals, lower, upper = [], [], [], [], []
    row_id = 0

    def add(indices, coefficients, lo=-np.inf, hi=np.inf):
        nonlocal row_id
        rows.extend([row_id] * len(indices)); cols.extend(indices); vals.extend(coefficients)
        lower.append(lo); upper.append(hi); row_id += 1

    for _, idx in candidates.groupby("match_id").groups.items():
        ii = list(idx); add(ii, [1] * len(ii), 1, 1)

    # Explicitly encode chronological round order and the true 60-hour rest
    # requirement. Because each match has exactly one selected candidate, the
    # selected kickoff time is a linear weighted sum of its candidate times.
    epoch = pd.Timestamp("2026-01-01", tz="UTC")
    kickoff_hours = (candidates["utc_datetime"] - epoch).dt.total_seconds() / 3600.0
    match_meta = candidates[["match_id", "round_in_group", "team_a", "team_b"]].drop_duplicates("match_id")
    teams = sorted(set(match_meta["team_a"]) | set(match_meta["team_b"]))
    rest_constraint_count = 0
    for team in teams:
        games = match_meta[match_meta["team_a"].eq(team) | match_meta["team_b"].eq(team)].sort_values("round_in_group")
        if games["round_in_group"].tolist() != [1, 2, 3]:
            raise RuntimeError(f"Unexpected round sequence for {team}")
        for previous, following in zip(games.itertuples(index=False), games.iloc[1:].itertuples(index=False)):
            prev_idx = list(candidates.index[candidates["match_id"].eq(previous.match_id)])
            next_idx = list(candidates.index[candidates["match_id"].eq(following.match_id)])
            coefficients = (-kickoff_hours.loc[prev_idx]).tolist() + kickoff_hours.loc[next_idx].tolist()
            add(prev_idx + next_idx, coefficients, 60.0, np.inf)
            rest_constraint_count += 1
    for _, idx in candidates.groupby(["venue_id", "slot_id"]).groups.items():
        ii = list(idx); add(ii, [1] * len(ii), -np.inf, 1)
    slot_caps = sheets["time_slots"].set_index("slot_id")["broadcast_capacity"].to_dict()
    for slot_id, idx in candidates.groupby("slot_id").groups.items():
        ii = list(idx); add(ii, [1] * len(ii), -np.inf, int(slot_caps[slot_id]))
    venue_day_caps = sheets["venues"].set_index("venue_id")["max_matches_per_day"].to_dict()
    for (venue_id, _), idx in candidates.groupby(["venue_id", "local_date"]).groups.items():
        ii = list(idx); add(ii, [1] * len(ii), -np.inf, int(venue_day_caps[venue_id]))
    venue_table = sheets["venues"].set_index("venue_id")
    for venue_id, idx in candidates.groupby("venue_id").groups.items():
        ii = list(idx); v = venue_table.loc[venue_id]
        add(ii, [1] * len(ii), int(v["min_total_matches"]), int(v["max_total_matches"]))

    # Third-round high-demand capacity by reference date.
    limits = sheets["dynamic_resource_limits"].set_index("reference_date")
    high = candidates[(candidates["round_in_group"] == 3) & (candidates["required_security_level"] >= 3)]
    for date, idx in high.groupby("reference_date").groups.items():
        ii = list(idx); add(ii, [1] * len(ii), -np.inf, int(limits.loc[str(date), "high_security_capacity"]))

    teams = sorted(set(candidates["team_a"]) | set(candidates["team_b"]))
    for team in teams:
        mask = (candidates["team_a"].eq(team) | candidates["team_b"].eq(team))
        prime_idx = list(candidates.index[mask & candidates["prime"].eq(1)])
        big_idx = list(candidates.index[mask & candidates["big"].eq(1)])
        add(prime_idx + [pmax], [1] * len(prime_idx) + [-1], -np.inf, 0)
        add(prime_idx + [pmin], [-1] * len(prime_idx) + [1], -np.inf, 0)
        add(big_idx + [bmax], [1] * len(big_idx) + [-1], -np.inf, 0)
        add(big_idx + [bmin], [-1] * len(big_idx) + [1], -np.inf, 0)
    add([pmax, pmin], [1, -1], -np.inf, 2)

    A = coo_matrix((vals, (rows, cols)), shape=(row_id, nvar)).tocsr()
    constraint = LinearConstraint(A, np.asarray(lower), np.asarray(upper))
    bounds = Bounds(np.zeros(nvar), np.r_[np.ones(n), np.full(4, 3.0)])
    integrality = np.r_[np.ones(n, dtype=int), np.zeros(4, dtype=int)]
    result = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraint,
                  options={"time_limit": 300, "mip_rel_gap": 0.001, "presolve": True})
    if result.x is None:
        raise RuntimeError(f"MILP produced no feasible incumbent: {result.message}")
    chosen = candidates.loc[result.x[:n] > 0.5].copy().sort_values(["group_id", "round_in_group", "match_id"])
    if len(chosen) != 72:
        raise RuntimeError(f"Expected 72 assignments, got {len(chosen)}")
    diag = {
        "solver_status": int(result.status), "solver_message": result.message,
        "mip_gap": float(getattr(result, "mip_gap", np.nan)), "mip_node_count": int(getattr(result, "mip_node_count", -1)),
        "candidate_rows": n, "constraint_rows": row_id, "explicit_rest_constraints": rest_constraint_count,
    }
    return chosen, diag


def team_counts(chosen: pd.DataFrame, column: str) -> dict[str, int]:
    teams = sorted(set(chosen["team_a"]) | set(chosen["team_b"]))
    return {team: int(chosen[(chosen["team_a"].eq(team) | chosen["team_b"].eq(team))][column].sum()) for team in teams}


def validate(chosen: pd.DataFrame, sheets: dict[str, pd.DataFrame], meta: dict) -> dict:
    violations = []
    if chosen["match_id"].nunique() != 72: violations.append("match uniqueness")
    if chosen.duplicated(["venue_id", "utc_datetime"]).any(): violations.append("venue-time collision")
    venue_table = sheets["venues"].set_index("venue_id")
    venue_counts = chosen.groupby("venue_id").size()
    for v in venue_table.index:
        count = int(venue_counts.get(v, 0))
        if not int(venue_table.loc[v, "min_total_matches"]) <= count <= int(venue_table.loc[v, "max_total_matches"]):
            violations.append(f"venue total {v}")
    for (v, _), g in chosen.groupby(["venue_id", "local_date"]):
        if len(g) > int(venue_table.loc[v, "max_matches_per_day"]): violations.append(f"venue day {v}")
    slot_caps = sheets["time_slots"].set_index("slot_id")["broadcast_capacity"]
    for s, g in chosen.groupby("slot_id"):
        if len(g) > int(slot_caps.loc[s]): violations.append(f"broadcast {s}")
    for team in sorted(set(chosen["team_a"]) | set(chosen["team_b"])):
        games = chosen[(chosen["team_a"].eq(team) | chosen["team_b"].eq(team))].sort_values("round_in_group")
        if list(games["round_in_group"]) != [1, 2, 3]: violations.append(f"round sequence {team}")
        gaps = games["utc_datetime"].diff().dt.total_seconds().dropna() / 3600
        if (gaps < 60 - 1e-9).any(): violations.append(f"rest {team}")
    prime = team_counts(chosen, "prime"); big = team_counts(chosen, "big")
    if max(prime.values()) - min(prime.values()) > 2: violations.append("prime fairness")
    limits = sheets["dynamic_resource_limits"].set_index("reference_date")
    high = chosen[(chosen["round_in_group"] == 3) & (chosen["required_security_level"] >= 3)]
    for date, g in high.groupby("reference_date"):
        if len(g) > int(limits.loc[str(date), "high_security_capacity"]): violations.append(f"third round high security {date}")
    return {
        "violations": violations,
        "min_rest_hours": float(min(
            (chosen[(chosen["team_a"].eq(t) | chosen["team_b"].eq(t))].sort_values("round_in_group")["utc_datetime"].diff().dt.total_seconds() / 3600).dropna().min()
            for t in prime
        )),
        "prime_count_range": [min(prime.values()), max(prime.values())],
        "big_venue_count_range": [min(big.values()), max(big.values())],
        "venue_total_range": [int(venue_counts.min()), int(venue_counts.max())],
    }


def output_schedule(chosen: pd.DataFrame, sheets: dict[str, pd.DataFrame], diag: dict, validation: dict) -> tuple[pd.DataFrame, dict]:
    venues = sheets["venues"].set_index("venue_id")
    slots = sheets["time_slots"].set_index("slot_id")
    params = sheets["ticket_broadcast"].set_index("match_stage")
    prime_counts = team_counts(chosen, "prime"); big_counts = team_counts(chosen, "big")
    fairness = 0.5 * (max(prime_counts.values()) - min(prime_counts.values())) / 3 + 0.5 * (max(big_counts.values()) - min(big_counts.values())) / 3
    indicator = {
        "T": float(chosen["ticket_n"].mean()), "B": float(chosen["broadcast_n"].mean()),
        "U": float(chosen["uncertainty"].mean()), "H": float(chosen["attractiveness"].mean()),
        "C": float(chosen["cost_n"].mean()), "D": float(chosen["travel_n"].mean()),
        "F": float(fairness), "R": float(chosen["risk_n"].mean()),
    }
    objective = 0.25*indicator["T"] + 0.25*indicator["B"] + 0.15*indicator["U"] + 0.10*indicator["H"] - 0.08*indicator["C"] - 0.07*indicator["D"] - 0.06*indicator["F"] - 0.04*indicator["R"]
    rows = []
    for k, (_, r) in enumerate(chosen.iterrows()):
        v = venues.loc[r["venue_id"]]; s = slots.loc[r["slot_id"]]
        stage = f"Group_Match_R{int(r['round_in_group'])}"; p = params.loc[stage]
        rows.append({
            "match_id": r["match_id"], "group_id": r["group_id"], "round_in_group": int(r["round_in_group"]),
            "team_a": r["team_a"], "team_b": r["team_b"], "venue_id": r["venue_id"], "slot_id": r["slot_id"],
            "city": v["city"], "country": v["country"], "reference_date": str(s["date"]),
            "reference_kickoff_time": s["reference_kickoff_time"],
            "local_datetime": pd.Timestamp(r["local_datetime"]).strftime("%Y-%m-%d %H:%M:%S%z"),
            "utc_datetime": pd.Timestamp(r["utc_datetime"]).strftime("%Y-%m-%d %H:%M:%SZ"),
            "required_security_level": int(r["required_security_level"]),
            "expected_attendance": int(round(r["expected_attendance"])),
            "expected_tv_viewers": int(r["predicted_tv_viewers"]),
            "ticket_revenue_usd": round(float(r["ticket"]), 2),
            "broadcast_value_usd": round(float(r["broadcast"]), 2),
            "travel_cost_index": round(float(r["travel"]), 6),
            "fairness_penalty": round(float(fairness), 6),
            "risk_index": round(float(r["risk"]), 6),
            "total_objective_value": round(float(objective), 8) if k == 0 else np.nan,
        })
    out = pd.DataFrame(rows)
    path = write_template_csv(out, "result_2_template.csv", "result_2_group_schedule.csv")
    metrics = {
        **diag, **validation, "indicators_normalized": indicator, "objective_value": float(objective),
        "total_ticket_revenue_usd": float(chosen["ticket"].sum()),
        "total_broadcast_value_usd": float(chosen["broadcast"].sum()),
        "total_setup_cost_musd": float(venues.loc[sorted(chosen["venue_id"].unique()), "setup_cost_musd"].sum()),
        "total_match_cost_musd": float(chosen["cost"].sum()),
        "total_complete_cost_musd": float(venues.loc[sorted(chosen["venue_id"].unique()), "setup_cost_musd"].sum() + chosen["cost"].sum()),
        "normalization_contract": {
            "ticket_revenue": "all feasible candidates Min-Max",
            "broadcast_value": "all feasible candidates Min-Max",
            "operation_cost": "complete cost equivalent including constant setup allocation, all feasible candidates Min-Max",
            "travel_burden": "all feasible candidates Min-Max",
            "execution_risk": "official bounded raw value in [0,1]; no second Min-Max",
        },
        "candidate_scope": "round-date-prefiltered MILP; reported gap does not bound the unrestricted problem",
        "output": str(path.relative_to(PROJECT_ROOT)),
    }
    return out, metrics


def run() -> dict:
    sheets = load_sheets("groups_matches", "venues", "time_slots", "distance_matrix", "ticket_broadcast", "base_predictions", "security_requirements", "dynamic_resource_limits")
    candidates, meta = build_candidates(sheets)
    chosen, diag = solve(candidates, sheets)
    validation = validate(chosen, sheets, meta)
    if validation["violations"]:
        raise RuntimeError(f"Schedule invariant failure: {validation['violations']}")
    _, metrics = output_schedule(chosen, sheets, diag, validation)
    metrics.update(meta)
    dump_json(metrics, "q2_metrics.json")
    return metrics


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
