from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import LOG_ROOT, OUTPUT_ROOT, PROJECT_ROOT, dump_json, load_sheets, write_template_csv


SEED = 20260830


TEAM_NUMERIC = [
    "market_value_musd", "avg_age", "star_index", "fan_base_index",
    "style_attack", "style_defense", "host_flag", "strength_score",
]


def _team_lookup(teams: pd.DataFrame) -> pd.DataFrame:
    cols = ["team_name", "confederation", "home_timezone_region"] + TEAM_NUMERIC
    return teams[cols].drop_duplicates("team_name").set_index("team_name")


def build_features(matches: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    d = matches.copy()
    d["date"] = pd.to_datetime(d["date"])
    lookup = _team_lookup(teams)
    scalar_lookup = teams.set_index("team_name")
    for side in ["a", "b"]:
        if f"elo_{side}" not in d:
            d[f"elo_{side}"] = d[f"team_{side}"].map(scalar_lookup["elo_rating"])
        if f"strength_rank_{side}" not in d:
            d[f"strength_rank_{side}"] = d[f"team_{side}"].map(scalar_lookup["strength_rank"])
    for side in ["a", "b"]:
        joined = lookup.add_suffix(f"_{side}").reindex(d[f"team_{side}"].values).reset_index(drop=True)
        d = pd.concat([d.reset_index(drop=True), joined], axis=1)

    # Pre-match market probabilities, normalized to remove the bookmaker margin.
    raw_probs = np.column_stack([
        1.0 / d["odds_a"].astype(float),
        1.0 / d["odds_draw"].astype(float),
        1.0 / d["odds_b"].astype(float),
    ])
    probs = raw_probs / raw_probs.sum(axis=1, keepdims=True)
    for j, name in enumerate(["p_a", "p_draw", "p_b"]):
        d[name] = probs[:, j]
    d["outcome_entropy"] = -(probs * np.log(np.clip(probs, 1e-12, 1))).sum(axis=1) / np.log(3)
    d["outcome_uncertainty"] = 1.0 - probs.max(axis=1)
    d["elo_mean"] = (d["elo_a"] + d["elo_b"]) / 2
    d["elo_abs_gap"] = (d["elo_a"] - d["elo_b"]).abs()
    d["rank_mean"] = (d["strength_rank_a"] + d["strength_rank_b"]) / 2
    d["rank_abs_gap"] = (d["strength_rank_a"] - d["strength_rank_b"]).abs()
    d["year"] = d["date"].dt.year
    d["year_centered"] = d["year"] - 2018
    d["neutral"] = d["neutral"].astype(int)
    d["host_any"] = d["host_flag_a"].astype("boolean").fillna(False).astype(int) + d["host_flag_b"].astype("boolean").fillna(False).astype(int)
    d["same_confederation"] = (d["confederation_a"] == d["confederation_b"]).astype(int)
    d["same_region"] = (d["home_timezone_region_a"] == d["home_timezone_region_b"]).astype(int)
    d["conf_pair"] = d[["confederation_a", "confederation_b"]].astype(str).apply(lambda r: "|".join(sorted(r)), axis=1)
    d["region_pair"] = d[["home_timezone_region_a", "home_timezone_region_b"]].astype(str).apply(lambda r: "|".join(sorted(r)), axis=1)

    for base in TEAM_NUMERIC:
        a = pd.to_numeric(d[f"{base}_a"], errors="coerce").astype(float)
        b = pd.to_numeric(d[f"{base}_b"], errors="coerce").astype(float)
        d[f"{base}_mean"] = (a + b) / 2
        d[f"{base}_max"] = np.maximum(a, b)
        d[f"{base}_gap"] = np.abs(a - b)
    return d


NUMERIC = [
    "neutral", "elo_mean", "elo_abs_gap", "rank_mean", "rank_abs_gap",
    "p_a", "p_draw", "p_b", "outcome_entropy", "outcome_uncertainty",
    "year_centered", "host_any", "same_confederation", "same_region",
] + [f"{base}_{agg}" for base in TEAM_NUMERIC for agg in ["mean", "max", "gap"]]

CATEGORICAL = ["competition", "stage", "conf_pair", "region_pair"]


def preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ], sparse_threshold=0.0)


def candidate_models() -> dict[str, Pipeline]:
    models = {
        "ridge": Ridge(alpha=12.0),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=700, min_samples_leaf=4, max_features=0.75,
            random_state=SEED, n_jobs=-1,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=700, min_samples_leaf=4, max_features=0.75,
            random_state=SEED, n_jobs=-1,
        ),
        "hist_gradient": HistGradientBoostingRegressor(
            max_iter=350, learning_rate=0.04, max_leaf_nodes=19,
            l2_regularization=3.0, random_state=SEED,
        ),
    }
    return {name: Pipeline([("prep", preprocessor()), ("model", model)]) for name, model in models.items()}


@dataclass
class FittedQ1:
    names: list[str]
    models: list[Pipeline]
    weights: np.ndarray
    metrics: dict

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        matrix = np.column_stack([m.predict(features) for m in self.models])
        return np.maximum(0.0, matrix @ self.weights)


def fit_time_ensemble(train_features: pd.DataFrame, y_million: np.ndarray) -> FittedQ1:
    base_models = candidate_models()
    splitter = TimeSeriesSplit(n_splits=5)
    oof = np.full((len(train_features), len(base_models)), np.nan)
    fold_records: list[dict] = []
    names = list(base_models)
    for fold, (tr, va) in enumerate(splitter.split(train_features), start=1):
        for j, name in enumerate(names):
            model = clone(base_models[name])
            model.fit(train_features.iloc[tr], y_million[tr])
            pred = model.predict(train_features.iloc[va])
            oof[va, j] = pred
            fold_records.append({
                "fold": fold, "model": name, "n_train": len(tr), "n_valid": len(va),
                "valid_start": str(train_features.iloc[va]["date"].min().date()),
                "valid_end": str(train_features.iloc[va]["date"].max().date()),
                "mse_million2": float(mean_squared_error(y_million[va], pred)),
            })
    usable = np.isfinite(oof).all(axis=1)
    raw_weights, _ = nnls(oof[usable], y_million[usable])
    if raw_weights.sum() <= 1e-12:
        raw_weights = np.ones(len(names))
    weights = raw_weights / raw_weights.sum()
    ensemble_oof = oof[usable] @ weights
    per_model = {
        name: float(mean_squared_error(y_million[usable], oof[usable, j]))
        for j, name in enumerate(names)
    }
    metrics = {
        "seed": SEED,
        "target_unit": "million viewers",
        "forbidden_post_match_features": ["goals", "xg", "shots", "possession", "attendance"],
        "fold_records": fold_records,
        "oof_rows": int(usable.sum()),
        "oof_mse_by_model_million2": per_model,
        "ensemble_weights": {name: float(weights[j]) for j, name in enumerate(names)},
        "ensemble_oof_mse_million2": float(mean_squared_error(y_million[usable], ensemble_oof)),
        "ensemble_oof_rmse_million": float(mean_squared_error(y_million[usable], ensemble_oof) ** 0.5),
    }
    fitted = []
    for name in names:
        model = clone(base_models[name])
        model.fit(train_features, y_million)
        fitted.append(model)
    return FittedQ1(names=names, models=fitted, weights=weights, metrics=metrics)


def group_feature_rows(groups: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    d = groups.merge(base[["match_id", "p_a_win", "p_draw", "p_b_win"]], on="match_id", validate="one_to_one")
    d["date"] = pd.Timestamp("2026-06-11") + pd.to_timedelta((d["round_in_group"] - 1) * 6, unit="D")
    d["competition"] = "World Cup"
    d["stage"] = "Group"
    d["neutral"] = True
    d["odds_a"] = 1 / np.clip(d["p_a_win"], 0.02, 0.96)
    d["odds_draw"] = 1 / np.clip(d["p_draw"], 0.02, 0.96)
    d["odds_b"] = 1 / np.clip(d["p_b_win"], 0.02, 0.96)
    return d


def run() -> dict:
    sheets = load_sheets("historical_matches", "teams", "groups_matches", "base_predictions")
    historical = sheets["historical_matches"].sort_values(["date", "match_id"]).reset_index(drop=True)
    features = build_features(historical, sheets["teams"])
    train_mask = historical["dataset_split"].eq("train")
    test_mask = historical["dataset_split"].eq("test")
    if historical.loc[train_mask, "tv_viewers"].isna().any() or historical.loc[test_mask, "tv_viewers"].notna().any():
        raise ValueError("Official train/test target visibility invariant failed")
    train_features = features.loc[train_mask].reset_index(drop=True)
    y = historical.loc[train_mask, "tv_viewers"].to_numpy(float) / 1_000_000.0
    fitted = fit_time_ensemble(train_features, y)

    test_features = features.loc[test_mask].reset_index(drop=True)
    test_pred_million = fitted.predict(test_features)
    test_out = pd.DataFrame({
        "match_id_test": historical.loc[test_mask, "match_id"].to_numpy(),
        "predicted_test_tv_viewers": np.round(test_pred_million, 6),
    })
    test_path = write_template_csv(
        test_out, "result_1_test_prediction_template.csv", "result_1_test_prediction.csv"
    )

    group_raw = group_feature_rows(sheets["groups_matches"], sheets["base_predictions"])
    group_features = build_features(group_raw, sheets["teams"])
    group_pred_people = np.rint(fitted.predict(group_features) * 1_000_000).astype(np.int64)
    group_out = pd.DataFrame({
        "match_id": sheets["groups_matches"]["match_id"],
        "team_a": sheets["groups_matches"]["team_a"],
        "team_b": sheets["groups_matches"]["team_b"],
        "predicted_tv_viewers": group_pred_people,
    })
    group_path = write_template_csv(
        group_out, "result_1_match_prediction_template.csv", "result_1_match_prediction.csv"
    )

    # Compact descriptive evidence for the paper.
    fitted.metrics.update({
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "train_date_range": [str(historical.loc[train_mask, "date"].min().date()), str(historical.loc[train_mask, "date"].max().date())],
        "test_date_range": [str(historical.loc[test_mask, "date"].min().date()), str(historical.loc[test_mask, "date"].max().date())],
        "train_target_mean_million": float(y.mean()),
        "train_target_std_million": float(y.std(ddof=1)),
        "test_prediction_range_million": [float(test_pred_million.min()), float(test_pred_million.max())],
        "group_prediction_range_people": [int(group_pred_people.min()), int(group_pred_people.max())],
        "test_output": str(test_path.relative_to(PROJECT_ROOT)),
        "group_output": str(group_path.relative_to(PROJECT_ROOT)),
    })
    dump_json(fitted.metrics, "q1_metrics.json")
    joblib.dump(fitted, LOG_ROOT / "q1_model.joblib")
    return fitted.metrics


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
