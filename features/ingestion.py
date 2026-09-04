"""Data ingestion and validation module for offer-abuse detection pipeline.

Provides schema validation, timestamp normalization, type coercion,
and data health statistics for all raw historical merchant tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {
    "customers": {"customer_id", "created_at"},
    "orders": {"order_id", "customer_id", "amount", "timestamp"},
    "offer_redemptions": {"redemption_id", "customer_id", "order_id", "offer_id", "timestamp"},
    "customer_devices": {"customer_id", "device_id"},
    "customer_addresses": {"customer_id", "address_id"},
    "customer_payments": {"customer_id", "payment_id"},
    "customer_ips": {"customer_id", "ip_address"},
}


@dataclass
class ValidationReport:
    table_name: str
    total_rows: int
    valid_rows: int
    dropped_rows: int
    missing_columns: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.missing_columns) == 0 and self.dropped_rows == 0


def _normalize_timestamp_series(series: pd.Series) -> pd.Series:
    """Coerce timestamps to UTC naive Timestamp objects."""
    dt_series = pd.to_datetime(series, errors="coerce")
    if hasattr(dt_series.dt, "tz") and dt_series.dt.tz is not None:
        dt_series = dt_series.dt.tz_convert("UTC").dt.tz_localize(None)
    return dt_series


def validate_and_clean_table(
    df: pd.DataFrame,
    table_name: str,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate, clean, and coerce types for a single raw data table."""
    df_clean = df.copy()

    # Strip column name whitespace
    df_clean.columns = [str(col).strip() for col in df_clean.columns]

    required = REQUIRED_COLUMNS.get(table_name, set())
    missing = sorted(list(required - set(df_clean.columns)))

    if missing:
        report = ValidationReport(
            table_name=table_name,
            total_rows=len(df),
            valid_rows=0,
            dropped_rows=len(df),
            missing_columns=missing,
            errors=[f"Table '{table_name}' missing required columns: {missing}"],
        )
        return pd.DataFrame(), report

    total_rows = len(df_clean)
    initial_mask = pd.Series(True, index=df_clean.index)

    # Normalize IDs to string and strip whitespace
    for col in df_clean.columns:
        if "id" in col or col in ("customer_id", "ip_address"):
            if col in required:
                initial_mask &= df.loc[df_clean.index, col].notna()
            df_clean[col] = df_clean[col].astype(str).str.strip()
            # Drop null/empty strings in required ID columns
            if col in required:
                initial_mask &= (df_clean[col] != "") & (df_clean[col] != "nan") & (df_clean[col] != "None")

    # Normalize timestamps
    if table_name == "customers" and "created_at" in df_clean.columns:
        df_clean["created_at"] = _normalize_timestamp_series(df_clean["created_at"])
        initial_mask &= df_clean["created_at"].notna()

    if table_name in ("orders", "offer_redemptions") and "timestamp" in df_clean.columns:
        df_clean["timestamp"] = _normalize_timestamp_series(df_clean["timestamp"])
        initial_mask &= df_clean["timestamp"].notna()

    # Coerce numeric amounts
    if table_name == "orders" and "amount" in df_clean.columns:
        df_clean["amount"] = pd.to_numeric(df_clean["amount"], errors="coerce").fillna(0.0)
    if table_name == "offer_redemptions" and "discount_amount" in df_clean.columns:
        df_clean["discount_amount"] = pd.to_numeric(df_clean["discount_amount"], errors="coerce").fillna(0.0)

    # Filter invalid rows
    valid_df = df_clean.loc[initial_mask].copy()
    dropped_count = total_rows - len(valid_df)

    errors = []
    if dropped_count > 0:
        errors.append(f"Dropped {dropped_count} invalid/corrupt rows from '{table_name}'")

    report = ValidationReport(
        table_name=table_name,
        total_rows=total_rows,
        valid_rows=len(valid_df),
        dropped_rows=dropped_count,
        missing_columns=[],
        errors=errors,
    )
    return valid_df, report


def load_raw_dataset(
    source: str | Path | dict[str, pd.DataFrame | list[dict[str, Any]]],
) -> tuple[dict[str, pd.DataFrame], dict[str, ValidationReport]]:
    """Load and validate all 7 source tables from a directory or dictionary."""
    tables = [
        "customers",
        "orders",
        "offer_redemptions",
        "customer_devices",
        "customer_addresses",
        "customer_payments",
        "customer_ips",
    ]

    cleaned_dataset: dict[str, pd.DataFrame] = {}
    reports: dict[str, ValidationReport] = {}

    if isinstance(source, (str, Path)):
        data_dir = Path(source)
        for table in tables:
            file_path = data_dir / f"{table}.csv"
            if not file_path.exists():
                reports[table] = ValidationReport(
                    table_name=table,
                    total_rows=0,
                    valid_rows=0,
                    dropped_rows=0,
                    missing_columns=list(REQUIRED_COLUMNS.get(table, [])),
                    errors=[f"File not found: {file_path}"],
                )
                cleaned_dataset[table] = pd.DataFrame()
            else:
                raw_df = pd.read_csv(file_path)
                cleaned_df, report = validate_and_clean_table(raw_df, table)
                cleaned_dataset[table] = cleaned_df
                reports[table] = report
    elif isinstance(source, dict):
        for table in tables:
            raw_input = source.get(table, [])
            if isinstance(raw_input, pd.DataFrame):
                raw_df = raw_input
            elif isinstance(raw_input, list):
                raw_df = pd.DataFrame(raw_input)
            else:
                raw_df = pd.DataFrame()

            cleaned_df, report = validate_and_clean_table(raw_df, table)
            cleaned_dataset[table] = cleaned_df
            reports[table] = report
    else:
        raise ValueError(f"Unsupported data source type: {type(source)}")

    return cleaned_dataset, reports
