"""Inference boundary for the persisted group-aware XGBoost offer-abuse model.

This module deliberately does not import or invoke the training pipeline.  It
constructs an as-of snapshot of caller-supplied history, reuses the existing
feature-engineering implementation, and scores one customer with the frozen
artifact.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from features.feature_engineering import build_feature_matrix


MODEL_PATH = Path(__file__).parent / "outputs" / "model_xgboost_groupaware.joblib"
MODEL_NAME = "xgboost_groupaware"
DECISION_THRESHOLD = 0.5

# This is the precise order constructed by the current feature-engineering
# pipeline and used when the persisted 16-input XGBoost model was trained.
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


def _normalise_as_of(as_of: Any) -> pd.Timestamp:
    """Return a naive UTC timestamp compatible with the source CSV timestamps."""
    timestamp = pd.Timestamp(as_of)
    if pd.isna(timestamp):
        raise ValueError("as_of must be a valid timestamp")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


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
    """Create the raw-data snapshot that existed at ``as_of``.

    Entity-link files in the current schema do not carry an effective-time
    column. They are therefore constrained to customers known by ``as_of``;
    callers must provide links that were known at that time.
    """
    inputs = {
        "customers": customers.copy(),
        "orders": orders.copy(),
        "offer_redemptions": offer_redemptions.copy(),
        "customer_devices": customer_devices.copy(),
        "customer_addresses": customer_addresses.copy(),
        "customer_payments": customer_payments.copy(),
        "customer_ips": customer_ips.copy(),
    }
    _require_columns(inputs["customers"], {"customer_id", "created_at"}, "customers")
    _require_columns(inputs["orders"], {"customer_id", "timestamp"}, "orders")
    _require_columns(inputs["offer_redemptions"], {"customer_id", "timestamp"}, "offer_redemptions")
    for name in ("customer_devices", "customer_addresses", "customer_payments", "customer_ips"):
        _require_columns(inputs[name], {"customer_id"}, name)

    inputs["customers"]["created_at"] = pd.to_datetime(inputs["customers"]["created_at"])
    inputs["orders"]["timestamp"] = pd.to_datetime(inputs["orders"]["timestamp"])
    inputs["offer_redemptions"]["timestamp"] = pd.to_datetime(inputs["offer_redemptions"]["timestamp"])

    snapshot = {
        "customers": inputs["customers"].loc[inputs["customers"]["created_at"] <= as_of].copy(),
        "orders": inputs["orders"].loc[inputs["orders"]["timestamp"] <= as_of].copy(),
        "offer_redemptions": inputs["offer_redemptions"].loc[
            inputs["offer_redemptions"]["timestamp"] <= as_of
        ].copy(),
    }
    known_customer_ids = set(snapshot["customers"]["customer_id"])
    for name in ("customer_devices", "customer_addresses", "customer_payments", "customer_ips"):
        snapshot[name] = inputs[name].loc[inputs[name]["customer_id"].isin(known_customer_ids)].copy()
    return snapshot


def _feature_matrix_from_snapshot(snapshot: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Adapt in-memory history to the unchanged file-based feature pipeline."""
    with tempfile.TemporaryDirectory(prefix="offer_abuse_inference_") as temporary_dir:
        for name, frame in snapshot.items():
            frame.to_csv(Path(temporary_dir) / f"{name}.csv", index=False)
        return build_feature_matrix(data_dir=temporary_dir)


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
        customers, orders, offer_redemptions, customer_devices, customer_addresses,
        customer_payments, customer_ips, as_of_timestamp,
    )
    if customer_id not in set(snapshot["customers"]["customer_id"]):
        raise ValueError(f"customer_id {customer_id!r} does not exist as of {as_of_timestamp.isoformat()}")

    feature_matrix = _feature_matrix_from_snapshot(snapshot)
    row = feature_matrix.loc[feature_matrix["customer_id"] == customer_id]
    if len(row) != 1:
        raise ValueError(f"expected exactly one feature row for customer_id {customer_id!r}")
    feature_row = row.iloc[0]
    feature_values = feature_row.loc[list(FEATURE_COLUMNS)].to_frame().T

    resolved_model_path = Path(model_path)
    if not resolved_model_path.is_file():
        raise FileNotFoundError(f"persisted model not found: {resolved_model_path}")
    model = joblib.load(resolved_model_path)
    if getattr(model, "n_features_in_", len(FEATURE_COLUMNS)) != len(FEATURE_COLUMNS):
        raise ValueError("persisted model does not accept the required 16 features")

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
