"""Independent, read-only audit of the Prompt-Native C-problem run.

This script does not import the solver and does not trust its audit summary. It
reconstructs checks from the official workbook, official CSV templates, and
materialized outputs. The JSON it emits separates mechanical feasibility from
model-quality findings that require interpretation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "input"
RESULTS = ROOT / "results"
REPRO = ROOT / "repro-check" / "results"
WORKBOOK = next(INPUT.rglob("*.xlsx"))


def finite(series: pd.Series) -> bool:
    return bool(np.isfinite(pd.to_numeric(series, errors="coerce")).all())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_stats(series: pd.Series) -> dict:
    x = pd.to_numeric(series, errors="coerce")
    return {
        "count": int(x.notna().sum()),
        "min": float(x.min()),
        "q25": float(x.quantile(0.25)),
        "median": float(x.median()),
        "mean": float(x.mean()),
        "q75": float(x.quantile(0.75)),
        "max": float(x.max()),
    }


def template_columns(name: str) -> list[str]:
    path = next(INPUT.rglob(name))
    return list(pd.read_csv(path, nrows=0).columns)


def output_contract(path: Path, template_name: str) -> dict:
    frame = pd.read_csv(path)
    expected = template_columns(template_name)
    return {
        "columns_exact": list(frame.columns) == expected,
        "expected_columns": expected,
        "actual_columns": list(frame.columns),
        "rows": int(len(frame)),
        "duplicate_rows": int(frame.duplicated().sum()),
    }


def minmax(x: float, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return 0.0 if hi <= lo else (x - lo) / (hi - lo)


def reproduce_comparison() -> dict:
    rows = {}
    for original in sorted(RESULTS.glob("*.csv")):
        replay = REPRO / original.name
        a, b = pd.read_csv(original), pd.read_csv(replay)
        entry = {
            "original_rows": int(len(a)),
            "replay_rows": int(len(b)),
            "exact_sha256": sha256(original) == sha256(replay),
            "original_sha256": sha256(original),
            "replay_sha256": sha256(replay),
            "columns_exact": list(a.columns) == list(b.columns),
        }
        if list(a.columns) == list(b.columns) and len(a) == len(b):
            numeric = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])]
            text_cols = [c for c in a.columns if c not in numeric]
            diffs = []
            for col in numeric:
                av = pd.to_numeric(a[col], errors="coerce").to_numpy(dtype=float)
                bv = pd.to_numeric(b[col], errors="coerce").to_numpy(dtype=float)
                diffs.append(float(np.nanmax(np.abs(av - bv))) if len(av) else 0.0)
            entry["max_abs_numeric_difference"] = max(diffs, default=0.0)
            entry["all_text_fields_equal"] = all(a[c].fillna("").astype(str).equals(b[c].fillna("").astype(str)) for c in text_cols)
        rows[original.name] = entry
    return rows


def q1_audit(data: dict[str, pd.DataFrame]) -> dict:
    test_out = pd.read_csv(RESULTS / "result_1_test_prediction.csv")
    match_out = pd.read_csv(RESULTS / "result_1_match_prediction.csv")
    hist = data["historical_matches"]
    train = hist[hist["dataset_split"].astype(str).str.lower() == "train"]
    hidden = hist[hist["dataset_split"].astype(str).str.lower() == "test"]
    groups = data["groups_matches"]
    return {
        "contracts": {
            "test": output_contract(RESULTS / "result_1_test_prediction.csv", "result_1_test_prediction_template.csv"),
            "group": output_contract(RESULTS / "result_1_match_prediction.csv", "result_1_match_prediction_template.csv"),
        },
        "test_ids_exact": set(test_out["match_id_test"]) == set(hidden["match_id"]),
        "group_ids_exact": set(match_out["match_id"]) == set(groups["match_id"]),
        "group_teams_exact": bool(
            match_out[["match_id", "team_a", "team_b"]]
            .sort_values("match_id")
            .reset_index(drop=True)
            .equals(groups[["match_id", "team_a", "team_b"]].sort_values("match_id").reset_index(drop=True))
        ),
        "predictions_finite_positive": bool(
            finite(test_out["predicted_test_tv_viewers"])
            and finite(match_out["predicted_tv_viewers"])
            and (test_out["predicted_test_tv_viewers"] > 0).all()
            and (match_out["predicted_tv_viewers"] > 0).all()
        ),
        "train_target_million": frame_stats(train["tv_viewers"] / 1e6),
        "hidden_test_prediction_million": frame_stats(test_out["predicted_test_tv_viewers"]),
        "group_prediction_million": frame_stats(match_out["predicted_tv_viewers"] / 1e6),
        "hidden_test_labels_available": bool(hidden["tv_viewers"].notna().any()),
        "interpretation": "Reported MSE/MAE/R2 are temporal holdout validation metrics, not the organizer's hidden-test score.",
    }


def correct_travel_index(data: dict[str, pd.DataFrame], schedule: pd.DataFrame) -> dict:
    gm = data["groups_matches"]
    venues = data["venues"]
    sec = data["security_requirements"].set_index("match_id")
    dist = data["distance_matrix"]

    def burden_components(match: pd.Series, team: str, destination: str) -> tuple[float, float, float]:
        if int(match["round_in_group"]) == 1:
            row = dist[(dist["origin_name"] == team) & (dist["destination_id"] == destination) & (dist["relation_type"] == "team_to_venue")]
        else:
            prev = gm[
                (gm["group_id"] == match["group_id"])
                & (gm["round_in_group"] == int(match["round_in_group"]) - 1)
                & ((gm["team_a"] == team) | (gm["team_b"] == team))
            ].iloc[0]
            eligible = venues.loc[venues["security_level"] >= int(sec.loc[prev["match_id"], "required_security_level"]), "venue_id"]
            row = dist[
                dist["origin_id"].isin(eligible)
                & (dist["destination_id"] == destination)
                & (dist["relation_type"] == "venue_to_venue")
            ]
        return tuple(float(row[c].mean()) for c in ["distance_km", "travel_time_hour", "timezone_diff"])

    all_components = []
    for _, match in gm.iterrows():
        eligible_destinations = venues.loc[
            venues["security_level"] >= int(sec.loc[match["match_id"], "required_security_level"]), "venue_id"
        ]
        for destination in eligible_destinations:
            for team in [match["team_a"], match["team_b"]]:
                all_components.append(burden_components(match, team, destination))
    values = np.asarray(all_components, dtype=float)
    bounds = [(float(values[:, i].min()), float(values[:, i].max())) for i in range(3)]

    reconstructed = []
    for _, out in schedule.iterrows():
        match = gm.loc[gm["match_id"] == out["match_id"]].iloc[0]
        team_values = []
        for team in [match["team_a"], match["team_b"]]:
            components = burden_components(match, team, out["venue_id"])
            normalized = [minmax(value, bound) for value, bound in zip(components, bounds)]
            team_values.append(0.5 * normalized[0] + 0.3 * normalized[1] + 0.2 * normalized[2])
        reconstructed.append(float(np.mean(team_values)))
    reported = schedule["travel_cost_index"].to_numpy(dtype=float)
    reconstructed_array = np.asarray(reconstructed)
    return {
        "component_candidate_bounds": {
            "distance_km": bounds[0],
            "travel_time_hour": bounds[1],
            "timezone_diff": bounds[2],
        },
        "reported_vs_statement_mae": float(np.mean(np.abs(reported - reconstructed_array))),
        "reported_vs_statement_max_abs": float(np.max(np.abs(reported - reconstructed_array))),
        "reported_vs_statement_correlation": float(np.corrcoef(reported, reconstructed_array)[0, 1]),
        "reported_range": [float(reported.min()), float(reported.max())],
        "statement_formula_range": [float(reconstructed_array.min()), float(reconstructed_array.max())],
        "formula_match": bool(np.allclose(reported, reconstructed_array, rtol=1e-6, atol=1e-6)),
    }


def q2_audit(data: dict[str, pd.DataFrame]) -> dict:
    out = pd.read_csv(RESULTS / "result_2_group_schedule.csv")
    gm = data["groups_matches"]
    venues = data["venues"].set_index("venue_id")
    slots = data["time_slots"].copy().set_index("slot_id")
    sec = data["security_requirements"].set_index("match_id")
    limits = data["dynamic_resource_limits"].copy()
    limits["reference_date"] = pd.to_datetime(limits["reference_date"]).dt.date
    limits = limits.set_index("reference_date")

    metadata = out[["match_id", "group_id", "round_in_group", "team_a", "team_b"]].sort_values("match_id").reset_index(drop=True)
    expected_metadata = gm[["match_id", "group_id", "round_in_group", "team_a", "team_b"]].sort_values("match_id").reset_index(drop=True)
    utc = pd.to_datetime(out["utc_datetime"], utc=True)
    local_as_utc = pd.to_datetime(out["local_datetime"], utc=True)
    slot_utc = pd.to_datetime(out["slot_id"].map(slots["reference_utc_time"]), utc=True)
    rest = []
    order_ok = True
    for team in sorted(set(out["team_a"]) | set(out["team_b"])):
        mask = (out["team_a"] == team) | (out["team_b"] == team)
        team_rows = out.loc[mask].assign(_utc=utc[mask]).sort_values("_utc")
        order_ok = order_ok and team_rows["round_in_group"].tolist() == [1, 2, 3]
        rest.extend(team_rows["_utc"].diff().dropna().dt.total_seconds().div(3600).tolist())

    local_dates = out["local_datetime"].astype(str).str.slice(0, 10)
    local_daily = out.assign(_local_date=local_dates).groupby(["venue_id", "_local_date"]).size()
    local_daily_excess = int(sum(max(0, int(n) - int(venues.loc[venue, "max_matches_per_day"])) for (venue, _), n in local_daily.items()))
    venue_counts = out["venue_id"].value_counts()

    slot_counts = out["slot_id"].value_counts()
    slot_excess = int(sum(max(0, int(n) - int(slots.loc[slot, "broadcast_capacity"])) for slot, n in slot_counts.items()))
    utc_counts = out.groupby("utc_datetime").size()
    utc_cap_excess = 0
    for utc_value, count in utc_counts.items():
        matching = data["time_slots"].loc[pd.to_datetime(data["time_slots"]["reference_utc_time"], utc=True) == pd.to_datetime(utc_value, utc=True)]
        capacity = int(matching["broadcast_capacity"].max())
        utc_cap_excess += max(0, int(count) - capacity)

    top_n = int(np.ceil(len(slots) * 0.25))
    gold_slots = set(slots.sort_values(["global_prime_score"], ascending=False).head(top_n).index)
    gold_counts = {}
    for team in sorted(set(out["team_a"]) | set(out["team_b"])):
        mask = (out["team_a"] == team) | (out["team_b"] == team)
        gold_counts[team] = int(out.loc[mask, "slot_id"].isin(gold_slots).sum())

    third_security_excess = 0
    for date_value, group in out.loc[out["round_in_group"] == 3].groupby("reference_date"):
        date_key = pd.to_datetime(date_value).date()
        third_security_excess += max(0, int((group["required_security_level"] >= 3).sum()) - int(limits.loc[date_key, "high_security_capacity"]))

    audit_summary = json.loads((ROOT / "audit" / "audit_summary.json").read_text(encoding="utf-8"))
    cost_bounds = tuple(audit_summary["q2"]["candidate_bounds"]["cost"])
    selected_counts = out["venue_id"].value_counts()
    selected_cost = np.asarray([
        float(venues.loc[row["venue_id"], "setup_cost_musd"]) / int(selected_counts[row["venue_id"]])
        + float(venues.loc[row["venue_id"], "operation_cost_musd_per_match"])
        + 0.1 * int(row["required_security_level"]) * float(venues.loc[row["venue_id"], "security_cost_index"])
        for _, row in out.iterrows()
    ])
    normalized_cost = np.asarray([minmax(x, cost_bounds) for x in selected_cost])
    reported_objective = float(out["total_objective_value"].dropna().iloc[0])
    clipped_objective = reported_objective + 0.08 * float(np.minimum(normalized_cost, 0).sum())

    required_level = out["match_id"].map(sec["required_security_level"])
    venue_level = out["venue_id"].map(venues["security_level"])
    return {
        "contract": output_contract(RESULTS / "result_2_group_schedule.csv", "result_2_template.csv"),
        "match_count_72": len(out) == 72,
        "unique_match_ids": out["match_id"].nunique() == 72,
        "match_metadata_exact": bool(metadata.equals(expected_metadata)),
        "known_venues_and_slots": bool(out["venue_id"].isin(venues.index).all() and out["slot_id"].isin(slots.index).all()),
        "slot_datetime_exact": bool((utc == slot_utc).all()),
        "local_datetime_converts_to_utc": bool((local_as_utc == utc).all()),
        "venue_utc_conflicts": int(out.duplicated(["venue_id", "utc_datetime"]).sum()),
        "round_order_correct": bool(order_ok),
        "min_rest_hours": float(min(rest)),
        "rest_ge_60": bool(min(rest) >= 60.0 - 1e-9),
        "local_date_daily_venue_excess": local_daily_excess,
        "venue_total_min_excess": int(sum(max(0, int(venues.loc[v, "min_total_matches"]) - int(venue_counts.get(v, 0))) for v in venues.index)),
        "venue_total_max_excess": int(sum(max(0, int(venue_counts.get(v, 0)) - int(venues.loc[v, "max_total_matches"])) for v in venues.index)),
        "security_eligibility_violations": int((venue_level.to_numpy() < required_level.to_numpy()).sum()),
        "slot_id_broadcast_capacity_excess": slot_excess,
        "same_utc_broadcast_capacity_excess": int(utc_cap_excess),
        "gold_slot_definition": {"slot_count": top_n, "score_cutoff": float(slots.loc[list(gold_slots), "global_prime_score"].min())},
        "gold_count_range": int(max(gold_counts.values()) - min(gold_counts.values())),
        "gold_fairness_le_2": max(gold_counts.values()) - min(gold_counts.values()) <= 2,
        "third_round_high_security_daily_excess": int(third_security_excess),
        "total_objective_only_first_row": bool(out["total_objective_value"].notna().sum() == 1 and out["total_objective_value"].first_valid_index() == 0),
        "cost_normalization_audit": {
            "candidate_bound_uses_full_setup_cost": list(cost_bounds),
            "selected_cost_allocates_setup_over_venue_matches": [float(selected_cost.min()), float(selected_cost.max())],
            "normalized_cost_range": [float(normalized_cost.min()), float(normalized_cost.max())],
            "selected_rows_below_candidate_min": int((normalized_cost < 0).sum()),
            "reported_objective": reported_objective,
            "objective_if_negative_cost_norm_only_is_clipped_to_zero": clipped_objective,
            "unearned_bonus_from_negative_cost_norm": float(reported_objective - clipped_objective),
            "same_cost_basis": False,
        },
        "travel_formula_audit": correct_travel_index(data, out),
        "global_optimality_certificate_present": False,
    }


def q3_audit(data: dict[str, pd.DataFrame]) -> dict:
    out = pd.read_csv(RESULTS / "result_3_dynamic_strategy.csv")
    schedule = pd.read_csv(RESULTS / "result_2_group_schedule.csv")
    states = pd.read_csv(ROOT / "audit" / "q3_team_states.csv")
    third = data["groups_matches"].loc[data["groups_matches"]["round_in_group"] == 3]
    venues = data["venues"].set_index("venue_id")
    sec = data["security_requirements"].set_index("match_id")
    costs = data["dynamic_resource_costs"].set_index(["resource_type", "resource_level"])
    limits = data["dynamic_resource_limits"].copy()
    limits["reference_date"] = pd.to_datetime(limits["reference_date"]).dt.date
    limits = limits.set_index("reference_date")
    joined = out.merge(schedule[["match_id", "venue_id", "reference_date"]], on="match_id", validate="one_to_one")

    daily = {}
    daily_ok = True
    for date_value, group in joined.groupby("reference_date"):
        key = pd.to_datetime(date_value).date()
        lim = limits.loc[key]
        observed = {
            "high_broadcast": int((group["recommended_broadcast_priority"] == 3).sum()),
            "high_broadcast_limit": int(lim["high_broadcast_capacity"]),
            "high_security": int((group["recommended_security_level"] >= 3).sum()),
            "high_security_limit": int(lim["high_security_capacity"]),
            "enhanced_transport": int((group["recommended_transport_level"] == 3).sum()),
            "enhanced_transport_limit": int(lim["enhanced_transport_capacity"]),
            "budget": float(group["resource_cost_index"].sum()),
            "budget_limit": float(lim["daily_resource_budget_index"]),
        }
        observed["passes"] = bool(
            observed["high_broadcast"] <= observed["high_broadcast_limit"]
            and observed["high_security"] <= observed["high_security_limit"]
            and observed["enhanced_transport"] <= observed["enhanced_transport_limit"]
            and observed["budget"] <= observed["budget_limit"] + 1e-9
        )
        daily_ok = daily_ok and observed["passes"]
        daily[str(key)] = observed

    expected_cost = []
    for _, row in out.iterrows():
        expected_cost.append(
            float(costs.loc[("broadcast", int(row["recommended_broadcast_priority"])), "unit_cost_index"])
            + float(costs.loc[("security", int(row["recommended_security_level"])), "unit_cost_index"])
            + float(costs.loc[("transport", int(row["recommended_transport_level"])), "unit_cost_index"])
        )
    expected_cost = np.asarray(expected_cost)
    required_security = joined["match_id"].map(sec["required_security_level"]).to_numpy()
    venue_security = joined["venue_id"].map(venues["security_level"]).to_numpy()
    deltas_ok = []
    for _, row in joined.iterrows():
        lim = limits.loc[pd.to_datetime(row["reference_date"]).date()]
        deltas_ok.append(-float(lim["max_ticket_discount_rate"]) - 1e-9 <= row["recommended_ticket_adjustment"] <= float(lim["max_ticket_increase_rate"]) + 1e-9)

    dynamic_total = float(out["dynamic_net_value"].sum())
    static_total = float(out["static_net_value"].sum())
    return {
        "contract": output_contract(RESULTS / "result_3_dynamic_strategy.csv", "result_3_template.csv"),
        "match_count_24": len(out) == 24,
        "third_round_ids_exact": set(out["match_id"]) == set(third["match_id"]),
        "probabilities_within_0_1": bool(
            out["updated_p_team_a_advance"].between(0, 1).all()
            and out["updated_p_team_b_advance"].between(0, 1).all()
            and states["advance_probability"].between(0, 1).all()
        ),
        "team_state_count": int(len(states)),
        "advancement_probability_sum": float(states["advance_probability"].sum()),
        "expected_advancement_probability_sum": 32.0,
        "resource_levels_in_domain": bool(
            out["recommended_broadcast_priority"].isin([1, 2, 3]).all()
            and out["recommended_transport_level"].isin([1, 2, 3]).all()
            and out["recommended_security_level"].isin([1, 2, 3, 4]).all()
        ),
        "minimum_and_venue_security_satisfied": bool(
            (out["recommended_security_level"].to_numpy() >= required_security).all()
            and (out["recommended_security_level"].to_numpy() <= venue_security).all()
        ),
        "ticket_adjustments_within_daily_bounds": bool(all(deltas_ok)),
        "updated_attendance_within_venue_capacity": bool(
            (joined["updated_expected_attendance"] >= 0).all()
            and (joined["updated_expected_attendance"] <= joined["venue_id"].map(venues["capacity"]) + 1e-7).all()
        ),
        "resource_cost_recomputed_exact": bool(np.allclose(expected_cost, out["resource_cost_index"].to_numpy(dtype=float))),
        "daily_resource_checks": daily,
        "all_daily_resource_checks_pass": bool(daily_ok),
        "dynamic_total_net": dynamic_total,
        "static_total_net": static_total,
        "dynamic_minus_static": dynamic_total - static_total,
        "dynamic_dominates_static_baseline": dynamic_total >= static_total - 1e-12,
        "matches_dynamic_better": int((out["dynamic_net_value"] > out["static_net_value"] + 1e-12).sum()),
        "matches_dynamic_worse": int((out["dynamic_net_value"] < out["static_net_value"] - 1e-12).sum()),
        "mean_rowwise_improvement_rate": float(out["improvement_rate"].mean()),
        "interpretation": "The static resource decisions are feasible candidates in the updated environment. A valid dynamic maximizer should therefore not have a lower total objective.",
    }


def q4_audit() -> dict:
    actual = pd.read_csv(RESULTS / "actual_schedule.csv")
    comparison = pd.read_csv(RESULTS / "result_4_schedule_comparison.csv")
    requested_families = [
        "travel_distance",
        "rest_time",
        "timezone_crossings",
        "venue_utilization",
        "capacity_matching",
        "prime_time",
        "expected_audience",
        "ticket_and_broadcast_value",
        "fairness",
        "risk",
    ]
    return {
        "contracts": {
            "actual": output_contract(RESULTS / "actual_schedule.csv", "actual_schedule_template.csv"),
            "comparison": output_contract(RESULTS / "result_4_schedule_comparison.csv", "result_4_template.csv"),
        },
        "actual_schedule_rows": int(len(actual)),
        "actual_match_ids_unique": bool(actual["match_id"].is_unique),
        "source_urls": sorted(actual["source_url"].dropna().unique().tolist()),
        "unique_local_kickoff_times": sorted(actual["local_kickoff_time"].dropna().astype(str).unique().tolist()),
        "comparison_indicator_count": int(len(comparison)),
        "comparison_indicators": comparison["indicator_name"].tolist(),
        "problem_requested_indicator_families": requested_families,
        "uses_same_q2_composite_objective_weights": False,
        "substantive_completion": False,
        "interpretation": "The run downloaded a public 2022 schedule, but used a common 18:00 proxy and only five coarse indicators rather than the Q2 indicator system and weights.",
    }


def main() -> None:
    xl = pd.ExcelFile(WORKBOOK)
    data = {sheet: pd.read_excel(WORKBOOK, sheet_name=sheet) for sheet in xl.sheet_names}
    report = {
        "scope": {
            "root": str(ROOT),
            "workbook_sha256": sha256(WORKBOOK),
            "solver_imported": False,
            "solver_audit_trusted": False,
        },
        "reproduction": reproduce_comparison(),
        "q1": q1_audit(data),
        "q2": q2_audit(data),
        "q3": q3_audit(data),
        "q4": q4_audit(),
    }
    destination = ROOT / "audit" / "independent_audit.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
