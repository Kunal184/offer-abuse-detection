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

    return {
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
