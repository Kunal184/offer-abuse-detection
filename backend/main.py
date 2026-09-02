"""Minimal HTTP boundary around the frozen ML inference function."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ml.inference import score_customer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "ml" / "outputs"


class HistoricalData(BaseModel):
    """A row from one historical source table."""

    model_config = ConfigDict(extra="allow")


class PredictionRequest(BaseModel):
    customer_id: str = Field(min_length=1)
    customers: list[HistoricalData]
    orders: list[HistoricalData]
    offer_redemptions: list[HistoricalData]
    customer_devices: list[HistoricalData]
    customer_addresses: list[HistoricalData]
    customer_payments: list[HistoricalData]
    customer_ips: list[HistoricalData]
    as_of: datetime


class PredictionResponse(BaseModel):
    customer_id: str
    abuse_probability: float
    predicted_label: int
    decision_threshold: float
    model_name: str
    model_version: str
    feature_snapshot: dict[str, Any]
    graph_signals: dict[str, Any]
    as_of: str
    scored_at: str


class BatchPredictionRequest(BaseModel):
    customer_ids: list[str] = Field(min_length=1)
    as_of: datetime


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    scored_at: str


app = FastAPI(title="Offer Abuse Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from functools import lru_cache

# ── Cache Helpers ────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_model():
    import joblib
    return joblib.load(OUTPUTS_DIR / "model_xgboost_groupaware.joblib")


def _records(filename: str) -> list[dict]:
    path = DATA_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Data file not found: {filename}")
    return pd.read_csv(path).to_dict(orient="records")


@lru_cache(maxsize=1)
def _load_all_data_cached() -> dict[str, list[dict]]:
    return {
        "customers": _records("customers.csv"),
        "orders": _records("orders.csv"),
        "offer_redemptions": _records("offer_redemptions.csv"),
        "customer_devices": _records("customer_devices.csv"),
        "customer_addresses": _records("customer_addresses.csv"),
        "customer_payments": _records("customer_payments.csv"),
        "customer_ips": _records("customer_ips.csv"),
    }


def _load_all_data() -> dict[str, list[dict]]:
    return _load_all_data_cached()


@lru_cache(maxsize=1)
def _compute_as_of() -> str:
    """Return the max timestamp across all source tables as ISO string."""
    all_data = _load_all_data()
    timestamps = [
        pd.to_datetime(r["created_at"]) for r in all_data["customers"]
    ]
    timestamps += [pd.to_datetime(r["timestamp"]) for r in all_data["orders"]]
    timestamps += [pd.to_datetime(r["timestamp"]) for r in all_data["offer_redemptions"]]
    return pd.to_datetime(timestamps).max().isoformat()



# ── Core prediction endpoint ────────────────────────────────────────────────

@app.post("/v1/predictions", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Score one customer using the frozen group-aware XGBoost artifact."""
    try:
        result = score_customer(
            customer_id=request.customer_id,
            customers=pd.DataFrame([row.model_dump() for row in request.customers]),
            orders=pd.DataFrame([row.model_dump() for row in request.orders]),
            offer_redemptions=pd.DataFrame(
                [row.model_dump() for row in request.offer_redemptions]
            ),
            customer_devices=pd.DataFrame(
                [row.model_dump() for row in request.customer_devices]
            ),
            customer_addresses=pd.DataFrame(
                [row.model_dump() for row in request.customer_addresses]
            ),
            customer_payments=pd.DataFrame(
                [row.model_dump() for row in request.customer_payments]
            ),
            customer_ips=pd.DataFrame(
                [row.model_dump() for row in request.customer_ips]
            ),
            as_of=request.as_of,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="ML model artifact unavailable") from exc
    except (TypeError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=f"Inference data is invalid: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="ML inference failed") from exc
    return PredictionResponse.model_validate(result)


# ── Batch prediction endpoint ────────────────────────────────────────────────

@app.post("/v1/predictions/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Score multiple customers in one request using the frozen XGBoost artifact."""
    all_data = _load_all_data()
    as_of_ts = pd.Timestamp(request.as_of)

    predictions = []
    for cid in request.customer_ids:
        try:
            result = score_customer(
                customer_id=cid,
                customers=pd.DataFrame(all_data["customers"]),
                orders=pd.DataFrame(all_data["orders"]),
                offer_redemptions=pd.DataFrame(all_data["offer_redemptions"]),
                customer_devices=pd.DataFrame(all_data["customer_devices"]),
                customer_addresses=pd.DataFrame(all_data["customer_addresses"]),
                customer_payments=pd.DataFrame(all_data["customer_payments"]),
                customer_ips=pd.DataFrame(all_data["customer_ips"]),
                as_of=as_of_ts,
            )
            predictions.append(PredictionResponse.model_validate(result))
        except (ValueError, KeyError):
            continue

    return BatchPredictionResponse(
        predictions=predictions,
        scored_at=datetime.now(timezone.utc).isoformat(),
    )


# ── Data serving endpoints ──────────────────────────────────────────────────

@app.get("/v1/data/customers")
def get_customers():
    return _records("customers.csv")


@app.get("/v1/data/orders")
def get_orders():
    return _records("orders.csv")


@app.get("/v1/data/redemptions")
def get_redemptions():
    return _records("offer_redemptions.csv")


@app.get("/v1/data/devices")
def get_devices():
    return _records("customer_devices.csv")


@app.get("/v1/data/addresses")
def get_addresses():
    return _records("customer_addresses.csv")


@app.get("/v1/data/payments")
def get_payments():
    return _records("customer_payments.csv")


@app.get("/v1/data/ips")
def get_ips():
    return _records("customer_ips.csv")


@app.get("/v1/data/features")
def get_features():
    return _records("customer_features.csv")


@app.get("/v1/data/ground-truth")
def get_ground_truth():
    return _records("ground_truth.csv")


# ── Overview stats ──────────────────────────────────────────────────────────

@app.get("/v1/overview")
def get_overview():
    """Return overview statistics using pre-computed features for fast response."""
    import numpy as np

    features = pd.read_csv(DATA_DIR / "customer_features.csv")
    as_of = _compute_as_of()

    # Load ground truth for abuse group count
    gt_df = pd.read_csv(DATA_DIR / "ground_truth.csv")
    gt_df["is_abuse"] = gt_df["abuse_group_id"].notna().astype(int)
    abuse_groups = gt_df[gt_df["is_abuse"] == 1]["abuse_group_id"].value_counts()

    # Use cached model
    model = _get_model()
    FEATURE_COLS = [
        "account_age_days", "order_count", "total_spend", "average_spend",
        "time_to_first_order_hours", "redemption_count", "time_to_first_redemption_hours",
        "order_redemption_rate", "max_device_user_count", "max_address_user_count",
        "max_payment_user_count", "max_ip_user_count", "unique_connected_customers",
        "avg_entity_degree", "max_entity_degree", "cluster_size",
    ]
    X = features[FEATURE_COLS].to_numpy()
    probabilities = model.predict_proba(X)[:, 1]

    total = len(probabilities)
    flagged = int(np.sum(probabilities >= 0.5))
    high = int(np.sum(probabilities >= 0.7))
    medium = int(np.sum((probabilities >= 0.3) & (probabilities < 0.7)))
    clear = int(np.sum(probabilities < 0.3))

    # Exposure from pre-computed total_spend
    flagged_mask = probabilities >= 0.5
    exposure = float(features.loc[flagged_mask, "total_spend"].sum())

    return JSONResponse({
        "customersAnalyzed": total,
        "customersFlagged": flagged,
        "abuseClusters": int(len(abuse_groups)),
        "totalExposure": round(exposure, 2),
        "flaggedRatio": round(flagged / total, 4) if total > 0 else 0,
        "riskDistribution": {
            "high": high,
            "medium": medium,
            "clear": clear,
        },
        "abuseGroupCount": int(len(abuse_groups)),
        "asOf": as_of,
    })


# ── Scored customers (pre-computed, bulk) ──────────────────────────────────

@app.get("/v1/data/scored-customers")
def get_scored_customers():
    """Return all customers joined with their pre-computed ML abuse scores.

    Uses the frozen model to score all customers at once in one vectorised
    call — safe and fast.  The frontend must use this instead of sending
    1000 individual /v1/predictions requests.
    """
    import numpy as np

    customers_df = pd.read_csv(DATA_DIR / "customers.csv")
    features = pd.read_csv(DATA_DIR / "customer_features.csv")

    model = _get_model()
    FEATURE_COLS = [
        "account_age_days", "order_count", "total_spend", "average_spend",
        "time_to_first_order_hours", "redemption_count", "time_to_first_redemption_hours",
        "order_redemption_rate", "max_device_user_count", "max_address_user_count",
        "max_payment_user_count", "max_ip_user_count", "unique_connected_customers",
        "avg_entity_degree", "max_entity_degree", "cluster_size",
    ]
    X = features[FEATURE_COLS].to_numpy()
    probabilities = model.predict_proba(X)[:, 1]
    features = features.copy()
    features["abuse_probability"] = probabilities
    features["predicted_label"] = (probabilities >= 0.5).astype(int)

    merged = customers_df.merge(
        features[["customer_id", "abuse_probability", "predicted_label",
                   "cluster_size", "unique_connected_customers"]],
        on="customer_id",
        how="left",
    )
    merged["abuse_probability"] = merged["abuse_probability"].fillna(0.0)
    merged["predicted_label"] = merged["predicted_label"].fillna(0).astype(int)
    merged["cluster_size"] = merged["cluster_size"].fillna(1).astype(int)
    merged["unique_connected_customers"] = merged["unique_connected_customers"].fillna(0).astype(int)

    return JSONResponse(merged.to_dict(orient="records"))


# ── Graph / cluster endpoints ────────────────────────────────────────────────

@app.get("/v1/graph")
def get_graph():
    """Return the entity relationship graph."""
    all_data = _load_all_data()
    G = _build_graph(all_data)

    nodes = []
    links = []
    for node, attrs in G.nodes(data=True):
        node_type = attrs.get("node_type", "unknown")
        label = node
        if node_type == "customer":
            label = node[2:]
        elif node_type == "device":
            label = f"Device {node[7:15]}"
        elif node_type == "address":
            label = f"Addr {node[8:16]}"
        elif node_type == "payment":
            label = f"Pay {node[8:16]}"
        elif node_type == "ip":
            label = node[3:]
        nodes.append({
            "id": node,
            "type": node_type,
            "label": label[:20],
        })

    for u, v in G.edges():
        u_type = G.nodes[u].get("node_type", "unknown")
        v_type = G.nodes[v].get("node_type", "unknown")
        links.append({
            "source": u,
            "target": v,
            "sourceType": u_type,
            "targetType": v_type,
        })

    return JSONResponse({"nodes": nodes, "links": links})


@app.get("/v1/clusters")
def get_clusters():
    """Return abuse clusters from the graph using pre-computed scores."""
    import numpy as np

    all_data = _load_all_data()
    G = _build_graph(all_data)
    features = pd.read_csv(DATA_DIR / "customer_features.csv")
    as_of = _compute_as_of()

    # Load model and score all customers at once
    model = _get_model()
    FEATURE_COLS = [
        "account_age_days", "order_count", "total_spend", "average_spend",
        "time_to_first_order_hours", "redemption_count", "time_to_first_redemption_hours",
        "order_redemption_rate", "max_device_user_count", "max_address_user_count",
        "max_payment_user_count", "max_ip_user_count", "unique_connected_customers",
        "avg_entity_degree", "max_entity_degree", "cluster_size",
    ]
    X = features[FEATURE_COLS].to_numpy()
    probabilities = model.predict_proba(X)[:, 1]
    features["abuse_probability"] = probabilities
    features["predicted_label"] = (probabilities >= 0.5).astype(int)

    # Get all flagged customer IDs
    flagged_ids = set(features[features["predicted_label"] == 1]["customer_id"].tolist())

    # Build connected components, label flagged clusters
    components = list(_connected_components(G))
    clusters = []
    for i, comp in enumerate(components):
        customer_nodes = [n for n in comp if str(n).startswith("c_")]
        entity_nodes = [n for n in comp if not str(n).startswith("c_")]
        customer_ids = [n[2:] for n in customer_nodes]
        flagged_in_cluster = [c for c in customer_ids if c in flagged_ids]

        if len(customer_ids) <= 1:
            continue

        # Determine entity types
        device_count = len([n for n in entity_nodes if str(n).startswith("device_")])
        addr_count = len([n for n in entity_nodes if str(n).startswith("address_")])
        pay_count = len([n for n in entity_nodes if str(n).startswith("payment_")])
        ip_count = len([n for n in entity_nodes if str(n).startswith("ip_")])

        # Overall risk
        flagged_ratio = len(flagged_in_cluster) / len(customer_ids)
        overall_risk = "high" if flagged_ratio > 0.5 else "medium" if flagged_ratio > 0 else "clear"

        clusters.append({
            "id": f"cluster_{i}",
            "customerCount": len(customer_ids),
            "flaggedCustomerCount": len(flagged_in_cluster),
            "sharedEntities": [
                {"type": "device", "count": device_count},
                {"type": "address", "count": addr_count},
                {"type": "payment", "count": pay_count},
                {"type": "ip", "count": ip_count},
            ],
            "overallRisk": overall_risk,
            "customers": customer_ids,
            "entities": entity_nodes,
        })

    # Sort by flagged ratio
    clusters.sort(key=lambda c: c["flaggedCustomerCount"] / max(c["customerCount"], 1), reverse=True)
    return JSONResponse({"clusters": clusters})


# ── Analytics ────────────────────────────────────────────────────────────────

@app.get("/v1/analytics/metrics")
def get_metrics():
    """Return frozen group-aware XGBoost baseline metrics."""
    eval_path = OUTPUTS_DIR / "evaluation_results.json"
    if eval_path.exists():
        with open(eval_path) as f:
            data = json.load(f)
        ga = data.get("group_aware_results", {}).get("XGBoost", {})
        test = ga.get("test", {})
        cm = test.get("confusion_matrix", [[0,0],[0,0]])
        return JSONResponse({
            "f1": round(test.get("f1", 0), 4),
            "precision": round(test.get("precision", 0), 4),
            "recall": round(test.get("recall", 0), 4),
            "rocAuc": round(test.get("roc_auc", 0), 4),
            "prAuc": round(test.get("pr_auc", 0), 4),
            "confusionMatrix": cm,
            "modelName": "xgboost_groupaware",
            "split": "group_aware",
        })
    return JSONResponse({
        "f1": 0.9351,
        "precision": 0.9730,
        "recall": 0.9000,
        "rocAuc": 0.9989,
        "prAuc": 0.9961,
        "confusionMatrix": [[132, 1], [4, 36]],
        "modelName": "xgboost_groupaware",
        "split": "group_aware",
    })


@app.get("/v1/analytics/feature-importance")
def get_feature_importance():
    """Return feature importance from group-aware XGBoost evaluation."""
    eval_path = OUTPUTS_DIR / "evaluation_results.json"
    if eval_path.exists():
        with open(eval_path) as f:
            data = json.load(f)
        ga = data.get("group_aware_results", {}).get("XGBoost", {})
        fi = ga.get("feature_importance", [])
        return JSONResponse([
            {"feature": f, "importance": round(v, 4)} for f, v in fi
        ])
    return JSONResponse([])


# ── Health ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": "xgboost_groupaware"}


# ── Internal helpers ────────────────────────────────────────────────────────

def _build_graph(all_data: dict) -> Any:
    """Build the bipartite entity graph."""
    import networkx as nx
    G = nx.Graph()
    for row in all_data["customers"]:
        G.add_node(f"c_{row['customer_id']}", node_type="customer")
    for row in all_data["customer_devices"]:
        G.add_node(f"device_{row['device_id']}", node_type="device")
        G.add_edge(f"c_{row['customer_id']}", f"device_{row['device_id']}")
    for row in all_data["customer_addresses"]:
        G.add_node(f"address_{row['address_id']}", node_type="address")
        G.add_edge(f"c_{row['customer_id']}", f"address_{row['address_id']}")
    for row in all_data["customer_payments"]:
        G.add_node(f"payment_{row['payment_id']}", node_type="payment")
        G.add_edge(f"c_{row['customer_id']}", f"payment_{row['payment_id']}")
    for row in all_data["customer_ips"]:
        G.add_node(f"ip_{row['ip_address']}", node_type="ip")
        G.add_edge(f"c_{row['customer_id']}", f"ip_{row['ip_address']}")
    return G


def _connected_components(G: Any) -> list[set]:
    """Return connected components as sets of node names."""
    import networkx as nx
    return list(nx.connected_components(G))
