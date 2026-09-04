"""Database persistence layer for offer abuse detection backend.

Provides SQLite schema initialization, seed data loading from CSVs,
CRUD query execution, WAL mode concurrency setup, and pandas export helpers.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DB_PATH = Path(__file__).resolve().parents[1] / "data" / "offer_abuse.db"


def get_db_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Create and return a SQLite connection with WAL mode enabled."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=20.0)
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for concurrent read/write performance
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=OFF;")
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    """Create all 8 database tables if they do not exist."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                name TEXT,
                email TEXT,
                phone TEXT,
                created_at TIMESTAMP NOT NULL
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                amount REAL NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                status TEXT NOT NULL
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS offer_redemptions (
                redemption_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                order_id TEXT NOT NULL,
                offer_id TEXT NOT NULL,
                discount_amount REAL NOT NULL,
                timestamp TIMESTAMP NOT NULL
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customer_devices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                device_id TEXT NOT NULL
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customer_addresses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                address_id TEXT NOT NULL
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customer_payments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                payment_id TEXT NOT NULL
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customer_ips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                ip_address TEXT NOT NULL
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS activity_logs (
                id TEXT PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                event_type TEXT NOT NULL,
                customer_id TEXT,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                details_json TEXT
            );
        """)
        
        conn.commit()


def seed_database(data_dir: Path | str, db_path: Path = DB_PATH, force: bool = False) -> None:
    """Populate database from seed CSVs if empty or force=True."""
    data_dir = Path(data_dir)
    init_db(db_path)
    
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM customers;")
        count = cursor.fetchone()[0]
        if count > 0 and not force:
            return

        if force:
            for table in [
                "customers", "orders", "offer_redemptions",
                "customer_devices", "customer_addresses", "customer_payments", "customer_ips", "activity_logs"
            ]:
                cursor.execute(f"DELETE FROM {table};")
            conn.commit()

        tables_to_csv = [
            ("customers", "customers.csv"),
            ("orders", "orders.csv"),
            ("offer_redemptions", "offer_redemptions.csv"),
            ("customer_devices", "customer_devices.csv"),
            ("customer_addresses", "customer_addresses.csv"),
            ("customer_payments", "customer_payments.csv"),
            ("customer_ips", "customer_ips.csv"),
        ]

        from features.ingestion import validate_and_clean_table

        for table_name, csv_name in tables_to_csv:
            csv_file = data_dir / csv_name
            if not csv_file.exists():
                continue
            
            df = pd.read_csv(csv_file)
            if df.empty:
                continue

            # Clean and validate table rows using standard ingestion pipeline
            df, _ = validate_and_clean_table(df, table_name)
            if df.empty:
                continue

            if table_name == "customers":
                df = df.drop_duplicates(subset=["customer_id"])
            elif table_name == "orders":
                df = df.drop_duplicates(subset=["order_id"])
            elif table_name == "offer_redemptions":
                df = df.drop_duplicates(subset=["redemption_id"])

            if table_name == "customers" and "created_at" in df.columns:
                df["created_at"] = pd.to_datetime(df["created_at"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
            elif table_name in ("orders", "offer_redemptions") and "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")

            df.to_sql(table_name, conn, if_exists="append", index=False)

        conn.commit()


def load_dataset_from_db(db_path: Path = DB_PATH) -> dict[str, pd.DataFrame]:
    """Fetch all 7 core tables from SQLite into a dictionary of DataFrames."""
    init_db(db_path)
    dataset: dict[str, pd.DataFrame] = {}
    
    tables = [
        "customers",
        "orders",
        "offer_redemptions",
        "customer_devices",
        "customer_addresses",
        "customer_payments",
        "customer_ips",
    ]

    with get_db_connection(db_path) as conn:
        for table in tables:
            df = pd.read_sql_query(f"SELECT * FROM {table};", conn)
            
            if table == "customers" and "created_at" in df.columns and not df.empty:
                df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
            elif table in ("orders", "offer_redemptions") and "timestamp" in df.columns and not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                
            dataset[table] = df

    return dataset


def insert_customer(customer_data: dict[str, Any], db_path: Path = DB_PATH) -> None:
    """Insert a new customer into SQLite."""
    init_db(db_path)
    created_at = customer_data.get("created_at") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(created_at, datetime):
        created_at = created_at.strftime("%Y-%m-%d %H:%M:%S")

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO customers (customer_id, name, email, phone, created_at)
            VALUES (?, ?, ?, ?, ?);
            """,
            (
                str(customer_data["customer_id"]),
                customer_data.get("name"),
                customer_data.get("email"),
                customer_data.get("phone"),
                created_at,
            ),
        )
        conn.commit()


def insert_order(order_data: dict[str, Any], db_path: Path = DB_PATH) -> None:
    """Insert a new order into SQLite."""
    init_db(db_path)
    timestamp = order_data.get("timestamp") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(timestamp, datetime):
        timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO orders (order_id, customer_id, amount, timestamp, status)
            VALUES (?, ?, ?, ?, ?);
            """,
            (
                str(order_data["order_id"]),
                str(order_data["customer_id"]),
                float(order_data.get("amount", 0.0)),
                timestamp,
                str(order_data.get("status", "completed")),
            ),
        )
        conn.commit()


def insert_offer_redemption(redemption_data: dict[str, Any], db_path: Path = DB_PATH) -> None:
    """Insert a new offer redemption into SQLite."""
    init_db(db_path)
    timestamp = redemption_data.get("timestamp") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(timestamp, datetime):
        timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO offer_redemptions (redemption_id, customer_id, order_id, offer_id, discount_amount, timestamp)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (
                str(redemption_data["redemption_id"]),
                str(redemption_data["customer_id"]),
                str(redemption_data["order_id"]),
                str(redemption_data.get("offer_id", "WELCOME50")),
                float(redemption_data.get("discount_amount", 0.0)),
                timestamp,
            ),
        )
        conn.commit()


def insert_entity_associations(
    customer_id: str,
    device_id: str | None = None,
    address_id: str | None = None,
    payment_id: str | None = None,
    ip_address: str | None = None,
    db_path: Path = DB_PATH,
) -> None:
    """Associate new entity tokens (device, address, payment, IP) with a customer in SQLite."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        if device_id:
            cursor.execute(
                "INSERT INTO customer_devices (customer_id, device_id) VALUES (?, ?);",
                (customer_id, str(device_id)),
            )
        if address_id:
            cursor.execute(
                "INSERT INTO customer_addresses (customer_id, address_id) VALUES (?, ?);",
                (customer_id, str(address_id)),
            )
        if payment_id:
            cursor.execute(
                "INSERT INTO customer_payments (customer_id, payment_id) VALUES (?, ?);",
                (customer_id, str(payment_id)),
            )
        if ip_address:
            cursor.execute(
                "INSERT INTO customer_ips (customer_id, ip_address) VALUES (?, ?);",
                (customer_id, str(ip_address)),
            )
        conn.commit()


def insert_activity_log(log_entry: dict[str, Any], db_path: Path = DB_PATH) -> None:
    """Insert a real-time activity log into SQLite."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO activity_logs (id, timestamp, event_type, customer_id, severity, message, details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (
                str(log_entry["id"]),
                str(log_entry["timestamp"]),
                str(log_entry["event_type"]),
                log_entry.get("customer_id"),
                str(log_entry["severity"]),
                str(log_entry["message"]),
                json.dumps(log_entry.get("details", {})),
            ),
        )
        conn.commit()


def fetch_activity_logs(limit: int = 100, db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    """Fetch recent activity logs from SQLite."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM activity_logs ORDER BY timestamp DESC LIMIT ?;",
            (limit,),
        )
        rows = cursor.fetchall()
        logs = []
        for r in rows:
            details = {}
            if r["details_json"]:
                try:
                    details = json.loads(r["details_json"])
                except Exception:
                    pass
            logs.append({
                "id": r["id"],
                "timestamp": r["timestamp"],
                "event_type": r["event_type"],
                "customer_id": r["customer_id"],
                "severity": r["severity"],
                "message": r["message"],
                "details": details,
            })
        return logs
