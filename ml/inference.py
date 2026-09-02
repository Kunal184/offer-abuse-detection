"""Inference boundary for the persisted group-aware XGBoost offer-abuse model.

Constructs an as-of snapshot of caller-supplied history, reuses the feature
engineering pipeline in-memory, and scores customers using the frozen artifact.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import xgboost as xgb

from features.feature_engineering import build_feature_matrix
from features.ingestion import validate_and_clean_table


MODEL_PATH = Path(__file__).parent / "outputs" / "model_xgboost_groupaware.joblib"
MODEL_NAME = "xgboost_groupaware"
DECISION_THRESHOLD = 0.5

FEATURE_COLUMNS = (
    "account_age_days",
    "order_count",
    "total_spend",
    "average_spend",
    "time_to_first_order_hours",
    "redemption_count",
    "time_to_first_redemption_hours",
    "order_redemption_rate",
    "max_device_user_count",
    "max_address_user_count",
    "max_payment_user_count",
    "max_ip_user_count",
    "unique_connected_customers",
    "avg_entity_degree",
    "max_entity_degree",
    "cluster_size",
)

GRAPH_SIGNAL_COLUMNS = (
    "max_device_user_count",
    "max_address_user_count",
    "max_payment_user_count",
    "max_ip_user_count",
    "unique_connected_customers",
    "avg_entity_degree",
    "max_entity_degree",
    "cluster_size",
)


@lru_cache(maxsize=4)
def _load_cached_model(model_path_str: str) -> Any:
    resolved_path = Path(model_path_str)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Persisted model not found: {resolved_path}")
    return joblib.load(resolved_path)


def _normalise_as_of(as_of: Any) -> pd.Timestamp:
    """Return a naive UTC timestamp compatible with source CSV timestamps."""
    timestamp = pd.Timestamp(as_of)
    if pd.isna(timestamp):
        raise ValueError("as_of must be a valid timestamp")
    if hasattr(timestamp, "tz") and timestamp.tz is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _as_of_snapshot(
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    offer_redemptions: pd.DataFrame,
    customer_devices: pd.DataFrame,
    customer_addresses: pd.DataFrame,
    customer_payments: pd.DataFrame,
    customer_ips: pd.DataFrame,
    as_of: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    """Create the raw-data snapshot that existed at ``as_of``."""
    inputs = {
        "customers": validate_and_clean_table(customers, "customers")[0],
        "orders": validate_and_clean_table(orders, "orders")[0],
        "offer_redemptions": validate_and_clean_table(offer_redemptions, "offer_redemptions")[0],
        "customer_devices": validate_and_clean_table(customer_devices, "customer_devices")[0],
        "customer_addresses": validate_and_clean_table(customer_addresses, "customer_addresses")[0],
        "customer_payments": validate_and_clean_table(customer_payments, "customer_payments")[0],
        "customer_ips": validate_and_clean_table(customer_ips, "customer_ips")[0],
    }

    snapshot = {
        "customers": inputs["customers"].loc[inputs["customers"]["created_at"] <= as_of].copy(),
        "orders": (
            inputs["orders"].loc[inputs["orders"]["timestamp"] <= as_of].copy()
            if not inputs["orders"].empty
            else pd.DataFrame()
        ),
        "offer_redemptions": (
            inputs["offer_redemptions"].loc[inputs["offer_redemptions"]["timestamp"] <= as_of].copy()
            if not inputs["offer_redemptions"].empty
            else pd.DataFrame()
        ),
    }

    known_customer_ids = set(snapshot["customers"]["customer_id"]) if not snapshot["customers"].empty else set()
    for name in ("customer_devices", "customer_addresses", "customer_payments", "customer_ips"):
        snapshot[name] = (
            inputs[name].loc[inputs[name]["customer_id"].isin(known_customer_ids)].copy()
            if not inputs[name].empty
            else pd.DataFrame()
        )
    return snapshot


def _model_version(model_path: Path) -> str:
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]
    return f"sha256:{digest}"


def compute_shap_explanation(
    feature_snapshot: dict[str, Any] | pd.Series,
    model: Any,
    top_k: int = 5,
) -> dict[str, Any]:
    """Compute exact Tree SHAP explanations for a single 16-feature vector."""
    if isinstance(feature_snapshot, dict):
        vals = [feature_snapshot[c] for c in FEATURE_COLUMNS]
    else:
        vals = [feature_snapshot[c] for c in FEATURE_COLUMNS]

    feature_df = pd.DataFrame([vals], columns=list(FEATURE_COLUMNS))
    booster = model.get_booster()
    dmatrix = xgb.DMatrix(feature_df, feature_names=list(FEATURE_COLUMNS))
    contribs = booster.predict(dmatrix, pred_contribs=True)[0]

    shap_values = contribs[: len(FEATURE_COLUMNS)]
    base_value = float(contribs[len(FEATURE_COLUMNS)])

    contributions = []
    for fname, shap_val in zip(FEATURE_COLUMNS, shap_values):
        fval = float(feature_df.iloc[0][fname])
        shap_v = float(shap_val)

        if shap_v > 0:
            direction = "increases_risk"
            impact = f"{fname} ({fval:.2f}) increases risk by +{shap_v:.4f} log-odds"
        elif shap_v < 0:
            direction = "decreases_risk"
            impact = f"{fname} ({fval:.2f}) decreases risk by {shap_v:.4f} log-odds"
        else:
            direction = "neutral"
            impact = f"{fname} ({fval:.2f}) has neutral impact"

        contributions.append({
            "feature_name": fname,
            "feature_value": fval,
            "shap_value": shap_v,
            "direction": direction,
            "impact": impact,
        })

    pos_contributors = [c for c in contributions if c["shap_value"] > 0]
    pos_contributors.sort(key=lambda x: x["shap_value"], reverse=True)

    neg_contributors = [c for c in contributions if c["shap_value"] < 0]
    neg_contributors.sort(key=lambda x: x["shap_value"])

    return {
        "base_value": base_value,
        "top_positive_contributors": pos_contributors[:top_k],
        "top_negative_contributors": neg_contributors[:top_k],
        "all_contributions": contributions,
    }


def score_customer(
    customer_id: str,
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    offer_redemptions: pd.DataFrame,
    customer_devices: pd.DataFrame,
    customer_addresses: pd.DataFrame,
    customer_payments: pd.DataFrame,
    customer_ips: pd.DataFrame,
    as_of: Any,
    model_path: str | os.PathLike[str] = MODEL_PATH,
    include_explanation: bool = False,
) -> dict[str, Any]:
    """Score one customer from historical data available no later than ``as_of``."""
    as_of_timestamp = _normalise_as_of(as_of)
    snapshot = _as_of_snapshot(
        customers,
        orders,
        offer_redemptions,
        customer_devices,
        customer_addresses,
        customer_payments,
        customer_ips,
        as_of_timestamp,
    )

    if customer_id not in set(snapshot["customers"]["customer_id"]):
        raise ValueError(f"customer_id {customer_id!r} does not exist as of {as_of_timestamp.isoformat()}")

    # In-memory feature matrix calculation — zero temporary file disk I/O
    feature_matrix = build_feature_matrix(data_frames=snapshot, as_of=as_of_timestamp)
    row = feature_matrix.loc[feature_matrix["customer_id"] == customer_id]
    if len(row) != 1:
        raise ValueError(f"Expected exactly one feature row for customer_id {customer_id!r}")

    feature_row = row.iloc[0]
    feature_values = feature_row.loc[list(FEATURE_COLUMNS)].to_frame().T

    resolved_model_path = Path(model_path).resolve()
    model = _load_cached_model(str(resolved_model_path))

    if getattr(model, "n_features_in_", len(FEATURE_COLUMNS)) != len(FEATURE_COLUMNS):
        raise ValueError("Persisted model does not accept the required 16 features")

    probability = float(model.predict_proba(feature_values.to_numpy())[:, 1][0])
    feature_snapshot = {name: feature_row[name].item() for name in FEATURE_COLUMNS}
    graph_signals = {name: feature_snapshot[name] for name in GRAPH_SIGNAL_COLUMNS}

    res = {
        "customer_id": customer_id,
        "abuse_probability": probability,
        "predicted_label": int(probability >= DECISION_THRESHOLD),
        "decision_threshold": DECISION_THRESHOLD,
        "model_name": MODEL_NAME,
        "model_version": _model_version(resolved_model_path),
        "feature_snapshot": feature_snapshot,
        "graph_signals": graph_signals,
        "as_of": as_of_timestamp.isoformat(),
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }

    if include_explanation:
        res["explanation"] = compute_shap_explanation(feature_snapshot, model)

    return res
