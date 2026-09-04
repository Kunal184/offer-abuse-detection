import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import networkx as nx

from features.ingestion import load_raw_dataset


FEATURE_COLUMNS = (
    "account_age_days",
    "order_count",
    "total_spend",
    "spend_to_discount_ratio",
    "time_to_first_order_hours",
    "redemption_count",
    "high_value_promo_ratio",
    "time_to_first_redemption_hours",
    "order_redemption_rate",
    "max_device_user_count",
    "max_address_user_count",
    "max_payment_user_count",
    "max_ip_user_count",
    "unique_connected_customers",
    "shared_entity_ratio",
    "avg_entity_degree",
    "max_entity_degree",
    "cluster_size",
    "cluster_creation_span_hours",
    "cluster_redemptions_1h",
    "min_account_creation_delta_minutes",
)


def build_feature_matrix(
    data_dir: str | Path | None = "data",
    data_frames: dict[str, pd.DataFrame | list[dict[str, Any]]] | None = None,
    as_of: Any | None = None,
) -> pd.DataFrame:
    """Build the feature matrix from historical source tables.

    Supports either loading from ``data_dir`` or taking pre-loaded ``data_frames``.
    Supports temporal anchoring via optional ``as_of``.
    """
    if data_frames is not None:
        source = data_frames
    elif data_dir is not None:
        source = data_dir
    else:
        source = "data"

    dataset, validation_reports = load_raw_dataset(source)

    customers = dataset["customers"]
    orders = dataset["orders"]
    offer_redemptions = dataset["offer_redemptions"]
    customer_devices = dataset["customer_devices"]
    customer_addresses = dataset["customer_addresses"]
    customer_payments = dataset["customer_payments"]
    customer_ips = dataset["customer_ips"]

    if customers.empty:
        # Return empty DataFrame with required schema if no customers
        cols = ["customer_id"] + list(FEATURE_COLUMNS)
        return pd.DataFrame(columns=cols)

    known_cids = set(customers["customer_id"])

    # Handle temporal snapshot if as_of is provided
    if as_of is not None:
        as_of_ts = pd.Timestamp(as_of)
        if hasattr(as_of_ts, "tz") and as_of_ts.tz is not None:
            as_of_ts = as_of_ts.tz_convert("UTC").tz_localize(None)

        customers = customers.loc[customers["created_at"] <= as_of_ts].copy()
        if not orders.empty:
            orders = orders.loc[orders["timestamp"] <= as_of_ts].copy()
        if not offer_redemptions.empty:
            offer_redemptions = offer_redemptions.loc[offer_redemptions["timestamp"] <= as_of_ts].copy()

        known_cids = set(customers["customer_id"])
        customer_devices = customer_devices.loc[customer_devices["customer_id"].isin(known_cids)].copy()
        customer_addresses = customer_addresses.loc[customer_addresses["customer_id"].isin(known_cids)].copy()
        customer_payments = customer_payments.loc[customer_payments["customer_id"].isin(known_cids)].copy()
        customer_ips = customer_ips.loc[customer_ips["customer_id"].isin(known_cids)].copy()

        max_dataset_time = as_of_ts
    else:
        # Determine maximum timestamp across generated dataset
        max_customer_time = customers["created_at"].max()
        max_order_time = orders["timestamp"].max() if not orders.empty else pd.Timestamp.min
        max_redemption_time = (
            offer_redemptions["timestamp"].max() if not offer_redemptions.empty else pd.Timestamp.min
        )
        max_dataset_time = max(max_customer_time, max_order_time, max_redemption_time)

        customer_devices = customer_devices[customer_devices["customer_id"].isin(known_cids)].copy()
        customer_addresses = customer_addresses[customer_addresses["customer_id"].isin(known_cids)].copy()
        customer_payments = customer_payments[customer_payments["customer_id"].isin(known_cids)].copy()
        customer_ips = customer_ips[customer_ips["customer_id"].isin(known_cids)].copy()

    ref_timestamp = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz=None)
    customers["created_at"] = pd.to_datetime(customers["created_at"])
    if customers["created_at"].dt.tz is not None:
        customers["created_at"] = customers["created_at"].dt.tz_convert("UTC").dt.tz_localize(None)

    df_features = pd.DataFrame()
    df_features["customer_id"] = customers["customer_id"]

    # Account age (days)
    df_features["account_age_days"] = (ref_timestamp - customers["created_at"]).dt.total_seconds() / (24 * 3600.0)

    # Order statistics
    if not orders.empty:
        orders["timestamp"] = pd.to_datetime(orders["timestamp"])
        if orders["timestamp"].dt.tz is not None:
            orders["timestamp"] = orders["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)

        completed_orders = orders[orders["status"] == "completed"].copy()
        order_stats = completed_orders.groupby("customer_id").agg(
            order_count=("order_id", "count"),
            total_spend=("amount", "sum"),
            first_order_time=("timestamp", "min"),
        ).reset_index()

        df_features = df_features.merge(order_stats, on="customer_id", how="left")
        df_features["order_count"] = df_features["order_count"].fillna(0).astype(int)
        df_features["total_spend"] = df_features["total_spend"].fillna(0.0)

        # Time to first order (hours)
        cust_order_time = customers[["customer_id", "created_at"]].merge(
            order_stats[["customer_id", "first_order_time"]], on="customer_id", how="left"
        )
        df_features["time_to_first_order_hours"] = (
            cust_order_time["first_order_time"] - cust_order_time["created_at"]
        ).dt.total_seconds() / 3600.0
        df_features["time_to_first_order_hours"] = df_features["time_to_first_order_hours"].fillna(-1.0)
        df_features.drop(columns=["first_order_time"], errors="ignore", inplace=True)
    else:
        df_features["order_count"] = 0
        df_features["total_spend"] = 0.0
        df_features["time_to_first_order_hours"] = -1.0

    offers = dataset.get("offers", pd.DataFrame())

    # Offer Redemption statistics
    if not offer_redemptions.empty:
        offer_redemptions["timestamp"] = pd.to_datetime(offer_redemptions["timestamp"])
        if offer_redemptions["timestamp"].dt.tz is not None:
            offer_redemptions["timestamp"] = offer_redemptions["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)

        if not offers.empty and "offer_id" in offer_redemptions.columns and "offer_id" in offers.columns:
            merged_red = offer_redemptions.merge(offers[["offer_id", "code"]], on="offer_id", how="left")
            merged_red["is_high_val"] = merged_red["code"].isin(["WELCOME50", "REFDOUBLE"]).astype(int)
        elif "offer_id" in offer_redemptions.columns:
            # Fall back to checking if promo code strings match directly or via offer_id
            merged_red = offer_redemptions.copy()
            merged_red["is_high_val"] = merged_red["offer_id"].isin(["WELCOME50", "REFDOUBLE"]).astype(int)
        else:
            merged_red = offer_redemptions.copy()
            merged_red["is_high_val"] = 0

        redemption_stats = merged_red.groupby("customer_id").agg(
            redemption_count=("redemption_id", "count"),
            high_val_count=("is_high_val", "sum"),
            total_discount_amount=("discount_amount", "sum"),
            first_redemption_time=("timestamp", "min"),
            orders_with_redemption=("order_id", "nunique"),
        ).reset_index()

        df_features = df_features.merge(redemption_stats, on="customer_id", how="left")
        df_features["redemption_count"] = df_features["redemption_count"].fillna(0).astype(int)
        df_features["high_val_count"] = df_features["high_val_count"].fillna(0).astype(int)
        df_features["total_discount_amount"] = df_features["total_discount_amount"].fillna(0.0)
        df_features["orders_with_redemption"] = df_features["orders_with_redemption"].fillna(0).astype(int)

        # High value promo ratio
        df_features["high_value_promo_ratio"] = np.where(
            df_features["redemption_count"] > 0,
            df_features["high_val_count"] / df_features["redemption_count"],
            0.0
        )
        df_features.drop(columns=["high_val_count"], errors="ignore", inplace=True)

        # Time to first redemption (hours)
        cust_red_time = customers[["customer_id", "created_at"]].merge(
            redemption_stats[["customer_id", "first_redemption_time"]], on="customer_id", how="left"
        )
        df_features["time_to_first_redemption_hours"] = (
            cust_red_time["first_redemption_time"] - cust_red_time["created_at"]
        ).dt.total_seconds() / 3600.0
        df_features["time_to_first_redemption_hours"] = df_features[
            "time_to_first_redemption_hours"
        ].fillna(-1.0)
        df_features.drop(columns=["first_redemption_time"], errors="ignore", inplace=True)
    else:
        df_features["redemption_count"] = 0
        df_features["high_value_promo_ratio"] = 0.0
        df_features["total_discount_amount"] = 0.0
        df_features["orders_with_redemption"] = 0
        df_features["time_to_first_redemption_hours"] = -1.0

    # Spend to discount ratio (normalized domain-agnostic metric)
    df_features["spend_to_discount_ratio"] = df_features["total_spend"] / (df_features["total_discount_amount"] + 1.0)
    df_features.drop(columns=["total_discount_amount"], errors="ignore", inplace=True)

    # Order redemption rate
    df_features["order_redemption_rate"] = np.where(
        df_features["order_count"] > 0,
        df_features["orders_with_redemption"] / df_features["order_count"],
        0.0,
    )
    df_features.drop(columns=["orders_with_redemption"], errors="ignore", inplace=True)

    # ---------------------------------------------------------
    # 2. Graph & Relational Features (NetworkX)
    # ---------------------------------------------------------
    G = nx.Graph()

    for cid in customers["customer_id"]:
        G.add_node(f"c_{cid}", node_type="customer")

    def add_entity_edges(df: pd.DataFrame, entity_col: str, prefix: str) -> None:
        if df.empty:
            return
        for _, row in df.iterrows():
            c_node = f"c_{row['customer_id']}"
            e_node = f"{prefix}_{row[entity_col]}"
            G.add_node(e_node, node_type=prefix)
            G.add_edge(c_node, e_node)

    add_entity_edges(customer_devices, "device_id", "device")
    add_entity_edges(customer_addresses, "address_id", "address")
    add_entity_edges(customer_payments, "payment_id", "payment")
    add_entity_edges(customer_ips, "ip_address", "ip")

    components = list(nx.connected_components(G))
    node_to_component_size = {}
    node_to_cluster_span_hours = {}

    cust_created_map = customers.set_index("customer_id")["created_at"].to_dict()

    # Pre-map redemptions per customer for cluster redemption velocity
    cust_red_timestamps = {}
    if not offer_redemptions.empty:
        for cid, group in offer_redemptions.groupby("customer_id"):
            cust_red_timestamps[str(cid)] = list(pd.to_datetime(group["timestamp"]))

    for comp in components:
        comp_cust_ids = [str(n)[2:] for n in comp if str(n).startswith("c_")]
        cust_count = len(comp_cust_ids)
        for n in comp:
            node_to_component_size[n] = cust_count

        if cust_count > 1:
            times = [cust_created_map[cid] for cid in comp_cust_ids if cid in cust_created_map]
            if len(times) > 1:
                span_h = (max(times) - min(times)).total_seconds() / 3600.0
            else:
                span_h = 0.0
        else:
            span_h = 0.0

        for n in comp:
            node_to_cluster_span_hours[n] = span_h

    graph_features = []
    for cid in customers["customer_id"]:
        c_node = f"c_{cid}"
        c_created_at = cust_created_map.get(cid)

        device_degrees = [G.degree(nbr) for nbr in G.neighbors(c_node) if str(nbr).startswith("device_")]
        address_degrees = [G.degree(nbr) for nbr in G.neighbors(c_node) if str(nbr).startswith("address_")]
        payment_degrees = [G.degree(nbr) for nbr in G.neighbors(c_node) if str(nbr).startswith("payment_")]
        ip_degrees = [G.degree(nbr) for nbr in G.neighbors(c_node) if str(nbr).startswith("ip_")]

        max_device_user_count = max(device_degrees) if device_degrees else 0
        max_address_user_count = max(address_degrees) if address_degrees else 0
        max_payment_user_count = max(payment_degrees) if payment_degrees else 0
        max_ip_user_count = max(ip_degrees) if ip_degrees else 0

        neighbors = list(G.neighbors(c_node))
        two_hop_customers = set()
        for ent in neighbors:
            for nbr_c in G.neighbors(ent):
                if nbr_c != c_node and str(nbr_c).startswith("c_"):
                    two_hop_customers.add(nbr_c)
        unique_connected_customers = len(two_hop_customers)

        # Minimum creation delta to connected accounts in cluster (minutes)
        if two_hop_customers and pd.notnull(c_created_at):
            other_creation_deltas = []
            for nbr_c in two_hop_customers:
                other_cid = nbr_c[2:]
                other_t = cust_created_map.get(other_cid)
                if pd.notnull(other_t):
                    delta_m = abs((c_created_at - other_t).total_seconds()) / 60.0
                    other_creation_deltas.append(delta_m)
            min_account_creation_delta_minutes = min(other_creation_deltas) if other_creation_deltas else -1.0
        else:
            min_account_creation_delta_minutes = -1.0

        # Cluster Redemption Burst Velocity (max redemptions in 1h window across cluster)
        all_cluster_cids = {cid} | {nbr_c[2:] for nbr_c in two_hop_customers}
        cluster_red_ts = []
        for cl_cid in all_cluster_cids:
            cluster_red_ts.extend(cust_red_timestamps.get(cl_cid, []))

        if len(cluster_red_ts) > 1:
            sorted_ts = sorted(cluster_red_ts)
            # Find maximum redemptions within any 1-hour window
            max_burst = 1
            l_idx = 0
            for r_idx in range(len(sorted_ts)):
                while (sorted_ts[r_idx] - sorted_ts[l_idx]).total_seconds() > 3600:
                    l_idx += 1
                max_burst = max(max_burst, r_idx - l_idx + 1)
            cluster_redemptions_1h = max_burst
        elif len(cluster_red_ts) == 1:
            cluster_redemptions_1h = 1
        else:
            cluster_redemptions_1h = 0

        all_entity_degrees = [G.degree(ent) for ent in neighbors]
        avg_entity_degree = float(np.mean(all_entity_degrees)) if all_entity_degrees else 0.0
        max_entity_degree = max(all_entity_degrees) if all_entity_degrees else 0

        shared_entities_count = sum(1 for ent in neighbors if G.degree(ent) > 1)
        shared_entity_ratio = float(shared_entities_count) / float(len(neighbors)) if neighbors else 0.0

        cluster_size = node_to_component_size.get(c_node, 1)
        cluster_creation_span_hours = node_to_cluster_span_hours.get(c_node, 0.0)

        graph_features.append({
            "customer_id": cid,
            "max_device_user_count": max_device_user_count,
            "max_address_user_count": max_address_user_count,
            "max_payment_user_count": max_payment_user_count,
            "max_ip_user_count": max_ip_user_count,
            "unique_connected_customers": unique_connected_customers,
            "shared_entity_ratio": shared_entity_ratio,
            "avg_entity_degree": avg_entity_degree,
            "max_entity_degree": max_entity_degree,
            "cluster_size": cluster_size,
            "cluster_creation_span_hours": cluster_creation_span_hours,
            "cluster_redemptions_1h": cluster_redemptions_1h,
            "min_account_creation_delta_minutes": min_account_creation_delta_minutes,
        })

    df_graph_features = pd.DataFrame(graph_features)
    df_features = df_features.merge(df_graph_features, on="customer_id", how="left")

    df_features.fillna(0, inplace=True)

    # Reorder columns explicitly to match FEATURE_COLUMNS contract
    ordered_cols = ["customer_id"] + list(FEATURE_COLUMNS)
    return df_features[ordered_cols]


def main():
    output_dir = "data"
    features_df = build_feature_matrix(data_dir=output_dir)

    assert features_df["customer_id"].is_unique, "Error: Duplicate customer_id found in features!"
    null_counts = features_df.isnull().sum().sum()
    assert null_counts == 0, f"Error: Found {null_counts} null values in feature matrix!"

    output_path = os.path.join(output_dir, "customer_features.csv")
    features_df.to_csv(output_path, index=False)

    print("\n--- Feature Engineering Completed ---")
    print(f"Feature table saved to: {output_path}")
    print(f"Dataset Shape: {features_df.shape}")
    print(f"Total nulls: {null_counts}")
    print("\nFeature Columns:")
    for col in features_df.columns:
        print(f" - {col}")


if __name__ == "__main__":
    main()
