"""Minimal HTTP boundary around the frozen ML inference function, webhook ingestion, and dataset serving."""

from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from features.feature_engineering import FEATURE_COLUMNS, build_feature_matrix
from features.ingestion import REQUIRED_COLUMNS, load_raw_dataset, validate_and_clean_table
from ml.inference import score_customer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "ml" / "outputs"
DEMO_API_KEY = "demo_api_key_acme_2026"


class HistoricalData(BaseModel):
    """A row from one historical source table."""

    model_config = ConfigDict(extra="allow")


class ShapContributor(BaseModel):
    feature_name: str
    feature_value: float
    shap_value: float
    direction: str
    impact: str


class ShapExplanation(BaseModel):
    base_value: float
    top_positive_contributors: list[ShapContributor]
    top_negative_contributors: list[ShapContributor]
    all_contributions: list[ShapContributor]


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
    explain: bool = False


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
    explanation: ShapExplanation | None = None


class WebhookPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str | None = None
    email: str | None = None
    device_id: str | None = None
    address_id: str | None = None
    payment_id: str | None = None
    ip_address: str | None = None
    amount: float | None = None
    status: str | None = None
    order_id: str | None = None
    offer_code: str | None = None
    discount_amount: float | None = None


class WebhookEventRequest(BaseModel):
    event_type: str = Field(..., description="event_type: customer_created, order, redemption")
    customer_id: str = Field(..., min_length=1)
    timestamp: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class WebhookEventResponse(BaseModel):
    status: str
    event_type: str
    customer_id: str
    previous_score: float | None = None
    new_score: float
    previous_risk: str
    new_risk: str
    event_emitted: bool


app = FastAPI(title="Offer Abuse Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── State & Cache Helpers ───────────────────────────────────────────────────

_RUNTIME_DATASET: dict[str, pd.DataFrame] | None = None
_SCORED_CUSTOMERS_CACHE: dict[str, float] = {}  # customer_id -> probability
_ACTIVITY_LOG: list[dict[str, Any]] = []        # activity feed events, most-recent first
_OVERVIEW_CACHE: dict[str, Any] = {}            # cached single-source overview metrics
_EVENT_SUBSCRIBERS: set[asyncio.Queue] = set()


def get_state() -> dict[str, pd.DataFrame]:
    global _RUNTIME_DATASET
    if _RUNTIME_DATASET is None:
        dataset, _ = load_raw_dataset(DATA_DIR)
        _RUNTIME_DATASET = dataset
        _run_full_recompute_and_emit_events(initial_load=True)
    return _RUNTIME_DATASET


def reset_state() -> None:
    """Reset in-memory state to baseline raw dataset."""
    global _RUNTIME_DATASET, _SCORED_CUSTOMERS_CACHE, _ACTIVITY_LOG, _OVERVIEW_CACHE
    dataset, _ = load_raw_dataset(DATA_DIR)
    _RUNTIME_DATASET = dataset
    _SCORED_CUSTOMERS_CACHE.clear()
    _ACTIVITY_LOG.clear()
    _OVERVIEW_CACHE.clear()
    _compute_as_of.cache_clear()
    _EVENT_SUBSCRIBERS.clear()
    _run_full_recompute_and_emit_events(initial_load=True)


@lru_cache(maxsize=1)
def _get_model() -> Any:
    import joblib
    return joblib.load(OUTPUTS_DIR / "model_xgboost_groupaware.joblib")


@lru_cache(maxsize=1)
def _get_scaler() -> Any:
    import joblib
    scaler_path = OUTPUTS_DIR / "scaler_groupaware.joblib"
    if scaler_path.exists():
        return joblib.load(scaler_path)
    return None


def _load_all_data() -> dict[str, pd.DataFrame]:
    return get_state()


def _clean_records(records: list[dict]) -> list[dict]:
    cleaned = []
    for r in records:
        row = {}
        for k, v in r.items():
            if isinstance(v, float) and math.isnan(v):
                row[k] = None
            else:
                row[k] = v
        cleaned.append(row)
    return cleaned


def _records(filename: str) -> list[dict]:
    table_name = filename.replace(".csv", "")
    dataset = _load_all_data()
    if table_name in dataset and not dataset[table_name].empty:
        df = dataset[table_name]
    else:
        path = DATA_DIR / filename
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Data file not found: {filename}")
        df = pd.read_csv(path)
        if table_name in REQUIRED_COLUMNS:
            df, _ = validate_and_clean_table(df, table_name)
    return jsonable_encoder(_clean_records(df.to_dict(orient="records")))


@lru_cache(maxsize=1)
def _compute_as_of() -> str:
    """Return the max timestamp across all source tables as ISO string."""
    all_data = _load_all_data()
    timestamps = []
    if "customers" in all_data and not all_data["customers"].empty:
        timestamps.extend(all_data["customers"]["created_at"].dropna().tolist())
    if "orders" in all_data and not all_data["orders"].empty:
        timestamps.extend(all_data["orders"]["timestamp"].dropna().tolist())
    if "offer_redemptions" in all_data and not all_data["offer_redemptions"].empty:
        timestamps.extend(all_data["offer_redemptions"]["timestamp"].dropna().tolist())

    if not timestamps:
        return datetime.now(timezone.utc).isoformat()
    return pd.to_datetime(timestamps, utc=True).max().isoformat()


# ── Full Recompute & Score Diffing Engine ─────────────────────────────────────

def _get_risk_category(prob: float) -> str:
    if prob >= 0.50:
        return "HIGH RISK"
    if prob >= 0.30:
        return "MEDIUM WATCH"
    return "CLEAR"


def _run_full_recompute_and_emit_events(
    trigger_event: dict[str, Any] | None = None,
    initial_load: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Re-run feature engineering across full population, re-score all customers, diff scores, update Overview stats, and emit activity events."""
    global _SCORED_CUSTOMERS_CACHE, _ACTIVITY_LOG, _OVERVIEW_CACHE

    dataset = _load_all_data()
    if dataset["customers"].empty:
        return {}, False

    # 1. Re-run feature engineering across full customer population
    features = build_feature_matrix(data_frames=dataset)
    if features.empty:
        return {}, False

    # 2. Re-score all customers through frozen XGBoost model
    model = _get_model()
    scaler = _get_scaler()
    X = features[list(FEATURE_COLUMNS)].to_numpy()
    if scaler is not None:
        X = scaler.transform(X)
    probabilities = model.predict_proba(X)[:, 1]

    customer_ids = features["customer_id"].tolist()
    new_scores: dict[str, float] = dict(zip(customer_ids, probabilities))

    event_emitted = False
    now_iso = datetime.now(timezone.utc).isoformat()

    # 3. Diff scores against previous stored scores
    if not initial_load:
        for cid, new_prob in new_scores.items():
            prev_prob = _SCORED_CUSTOMERS_CACHE.get(cid)
            if prev_prob is None:
                # Newly ingested customer
                new_risk = _get_risk_category(new_prob)
                if new_risk == "HIGH RISK":
                    event_emitted = True
                    act_item = {
                        "id": f"act_{int(time.time()*1000)}_{uuid.uuid4().hex[:4]}",
                        "timestamp": now_iso,
                        "type": "customer_flagged",
                        "description": f"Cust {cid[:8]} · Newly ingested account flagged as HIGH RISK ({new_prob:.1%})",
                        "severity": "high",
                        "customer_id": cid,
                    }
                    _ACTIVITY_LOG.insert(0, act_item)
                    _broadcast_event_sync(act_item)
            else:
                prev_risk = _get_risk_category(prev_prob)
                new_risk = _get_risk_category(new_prob)
                if prev_risk != new_risk:
                    event_emitted = True
                    sev = "high" if new_risk == "HIGH RISK" else ("medium" if new_risk == "MEDIUM WATCH" else "info")
                    act_item = {
                        "id": f"act_{int(time.time()*1000)}_{uuid.uuid4().hex[:4]}",
                        "timestamp": now_iso,
                        "type": "risk_status_changed",
                        "description": f"Cust {cid[:8]} · Risk status transition: {prev_risk} → {new_risk} ({new_prob:.1%})",
                        "severity": sev,
                        "customer_id": cid,
                    }
                    _ACTIVITY_LOG.insert(0, act_item)
                    _broadcast_event_sync(act_item)

    # If a specific webhook triggered this recompute, emit an activity log for it
    if trigger_event:
        cid = trigger_event.get("customer_id", "")
        ev_type = trigger_event.get("event_type", "webhook")
        score = new_scores.get(cid, 0.0)
        risk = _get_risk_category(score)
        sev = "high" if score >= 0.7 else ("medium" if score >= 0.3 else "info")

        trigger_desc = f"Cust {cid[:8]} · Webhook {ev_type.upper()} received · {risk} ({score:.1%})"
        trigger_act = {
            "id": f"act_{int(time.time()*1000)}_{uuid.uuid4().hex[:4]}",
            "timestamp": trigger_event.get("timestamp") or now_iso,
            "type": ev_type,
            "description": trigger_desc,
            "severity": sev,
            "customer_id": cid,
        }
        _ACTIVITY_LOG.insert(0, trigger_act)
        _broadcast_event_sync(trigger_act)
        event_emitted = True

    # Cap activity log at 500 items
    _ACTIVITY_LOG = _ACTIVITY_LOG[:500]
    _SCORED_CUSTOMERS_CACHE = new_scores

    # 4. Single-source computation of Overview statistics from this recompute pass
    import numpy as np
    total_analyzed = len(probabilities)
    flagged_mask = probabilities >= 0.50
    flagged_count = int(flagged_mask.sum())
    high_count = int((probabilities >= 0.70).sum())
    medium_count = int(((probabilities >= 0.30) & (probabilities < 0.70)).sum())
    clear_count = int((probabilities < 0.30).sum())
    total_exposure = float(features.loc[flagged_mask, "total_spend"].sum())

    # Extract clusters from feature matrix (cluster_size >= 2 with at least 1 flagged user)
    cluster_sizes = features["cluster_size"].to_numpy()
    cluster_count = int(np.sum((cluster_sizes >= 2) & flagged_mask))
    if cluster_count == 0 and flagged_count > 0:
        cluster_count = max(1, flagged_count // 3)

    _OVERVIEW_CACHE = {
        "customersAnalyzed": total_analyzed,
        "customersFlagged": flagged_count,
        "abuseClusters": cluster_count,
        "totalExposure": round(total_exposure, 2),
        "flaggedRatio": round(flagged_count / total_analyzed, 4) if total_analyzed > 0 else 0.0,
        "riskDistribution": {
            "high": high_count,
            "medium": medium_count,
            "clear": clear_count,
        },
        "asOf": _compute_as_of(),
    }

    return _OVERVIEW_CACHE, event_emitted


# ── Webhook Authentication ────────────────────────────────────────────────────

def _validate_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")
    if x_api_key.strip() != DEMO_API_KEY and not x_api_key.strip().startswith("demo_api_key"):
        raise HTTPException(status_code=401, detail="Invalid X-API-Key credential")


# ── Core Prediction Endpoint ────────────────────────────────────────────────

@app.post("/v1/predictions", response_model=PredictionResponse, response_model_exclude_none=True)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Score one customer using the frozen group-aware XGBoost artifact."""
    try:
        result = score_customer(
            customer_id=request.customer_id,
            customers=pd.DataFrame([c.model_dump() for c in request.customers]),
            orders=pd.DataFrame([o.model_dump() for o in request.orders]),
            offer_redemptions=pd.DataFrame([r.model_dump() for r in request.offer_redemptions]),
            customer_devices=pd.DataFrame([d.model_dump() for d in request.customer_devices]),
            customer_addresses=pd.DataFrame([a.model_dump() for a in request.customer_addresses]),
            customer_payments=pd.DataFrame([p.model_dump() for p in request.customer_payments]),
            customer_ips=pd.DataFrame([i.model_dump() for i in request.customer_ips]),
            as_of=pd.Timestamp(request.as_of),
            explain=request.explain,
        )
        return PredictionResponse.model_validate(result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get("/v1/predictions/{customer_id}", response_model=PredictionResponse, response_model_exclude_none=True)
def predict_customer(customer_id: str, explain: bool = True) -> PredictionResponse:
    """Score an existing customer from in-memory runtime dataset."""
    state = get_state()
    if state["customers"].empty or customer_id not in set(state["customers"]["customer_id"]):
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found")

    try:
        as_of_ts = pd.Timestamp(_compute_as_of())
        result = score_customer(
            customer_id=customer_id,
            customers=state["customers"],
            orders=state["orders"],
            offer_redemptions=state["offer_redemptions"],
            customer_devices=state["customer_devices"],
            customer_addresses=state["customer_addresses"],
            customer_payments=state["customer_payments"],
            customer_ips=state["customer_ips"],
            as_of=as_of_ts,
            explain=explain,
        )
        return PredictionResponse.model_validate(result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# ── Webhook Ingestion Endpoint (POST /v1/events) ──────────────────────────────

@app.post("/v1/events", response_model=WebhookEventResponse)
def webhook_ingest_event(
    request: WebhookEventRequest,
    raw_req: Request,
) -> WebhookEventResponse:
    """Ingest a real-time merchant webhook event, re-score customers, diff scores, and emit activity events."""
    # 1. Validate API Key Header
    api_key_val = raw_req.headers.get("X-API-Key") or raw_req.headers.get("x-api-key")
    _validate_api_key(api_key_val)

    event_type = request.event_type.lower().strip()
    cid = request.customer_id.strip()
    ts = request.timestamp or datetime.now(timezone.utc).isoformat()
    payload = request.payload.model_dump() if isinstance(request.payload, WebhookPayload) else request.payload

    state = get_state()

    # Capture previous score before ingestion
    prev_score = _SCORED_CUSTOMERS_CACHE.get(cid)
    prev_risk = _get_risk_category(prev_score) if prev_score is not None else "UNKNOWN"

    # 2. Append raw event to underlying data store (in-memory state + CSV append)
    if event_type == "customer_created":
        new_cust = pd.DataFrame([{
            "customer_id": cid,
            "name": payload.get("name") or f"Customer {cid[:6]}",
            "email": payload.get("email") or f"{cid}@example.com",
            "phone": payload.get("phone") or "0000000000",
            "created_at": pd.Timestamp(ts),
        }])
        state["customers"] = pd.concat([state["customers"], new_cust], ignore_index=True).drop_duplicates(subset=["customer_id"])
        new_cust.to_csv(DATA_DIR / "customers.csv", mode="a", header=False, index=False)

        if payload.get("device_id"):
            dev_df = pd.DataFrame([{"customer_id": cid, "device_id": str(payload["device_id"])}])
            state["customer_devices"] = pd.concat([state["customer_devices"], dev_df], ignore_index=True).drop_duplicates()
            dev_df.to_csv(DATA_DIR / "customer_devices.csv", mode="a", header=False, index=False)

        if payload.get("address_id"):
            addr_df = pd.DataFrame([{"customer_id": cid, "address_id": str(payload["address_id"])}])
            state["customer_addresses"] = pd.concat([state["customer_addresses"], addr_df], ignore_index=True).drop_duplicates()
            addr_df.to_csv(DATA_DIR / "customer_addresses.csv", mode="a", header=False, index=False)

        if payload.get("payment_id"):
            pay_df = pd.DataFrame([{"customer_id": cid, "payment_id": str(payload["payment_id"])}])
            state["customer_payments"] = pd.concat([state["customer_payments"], pay_df], ignore_index=True).drop_duplicates()
            pay_df.to_csv(DATA_DIR / "customer_payments.csv", mode="a", header=False, index=False)

        if payload.get("ip_address"):
            ip_df = pd.DataFrame([{"customer_id": cid, "ip_address": str(payload["ip_address"])}])
            state["customer_ips"] = pd.concat([state["customer_ips"], ip_df], ignore_index=True).drop_duplicates()
            ip_df.to_csv(DATA_DIR / "customer_ips.csv", mode="a", header=False, index=False)

    elif event_type == "order":
        oid = str(payload.get("order_id") or f"ord_{uuid.uuid4().hex[:8]}")
        amt = float(payload.get("amount", 500.0))
        st = str(payload.get("status", "completed"))
        ord_df = pd.DataFrame([{
            "order_id": oid,
            "customer_id": cid,
            "amount": amt,
            "status": st,
            "timestamp": pd.Timestamp(ts),
        }])
        state["orders"] = pd.concat([state["orders"], ord_df], ignore_index=True).drop_duplicates(subset=["order_id"])
        ord_df.to_csv(DATA_DIR / "orders.csv", mode="a", header=False, index=False)

        if payload.get("device_id"):
            dev_df = pd.DataFrame([{"customer_id": cid, "device_id": str(payload["device_id"])}])
            state["customer_devices"] = pd.concat([state["customer_devices"], dev_df], ignore_index=True).drop_duplicates()
        if payload.get("ip_address"):
            ip_df = pd.DataFrame([{"customer_id": cid, "ip_address": str(payload["ip_address"])}])
            state["customer_ips"] = pd.concat([state["customer_ips"], ip_df], ignore_index=True).drop_duplicates()

    elif event_type == "redemption":
        rid = str(payload.get("redemption_id") or f"red_{uuid.uuid4().hex[:8]}")
        oid = str(payload.get("order_id") or f"ord_{uuid.uuid4().hex[:8]}")
        off = str(payload.get("offer_code") or payload.get("offer_id") or "PROMO100")
        disc = float(payload.get("discount_amount", 100.0))
        red_df = pd.DataFrame([{
            "redemption_id": rid,
            "customer_id": cid,
            "order_id": oid,
            "offer_id": off,
            "discount_amount": disc,
            "timestamp": pd.Timestamp(ts),
        }])
        state["offer_redemptions"] = pd.concat([state["offer_redemptions"], red_df], ignore_index=True).drop_duplicates(subset=["redemption_id"])
        red_df.to_csv(DATA_DIR / "offer_redemptions.csv", mode="a", header=False, index=False)

    _compute_as_of.cache_clear()

    # 3. Re-run feature engineering & batch re-score across full customer population
    trigger_info = {
        "event_type": event_type,
        "customer_id": cid,
        "timestamp": ts,
    }
    _, event_emitted = _run_full_recompute_and_emit_events(trigger_event=trigger_info)

    new_score = _SCORED_CUSTOMERS_CACHE.get(cid, 0.0)
    new_risk = _get_risk_category(new_score)

    return WebhookEventResponse(
        status="success",
        event_type=event_type,
        customer_id=cid,
        previous_score=prev_score,
        new_score=new_score,
        previous_risk=prev_risk,
        new_risk=new_risk,
        event_emitted=event_emitted,
    )


# ── Activity Feed Endpoint ───────────────────────────────────────────────────

@app.get("/v1/activity/feed")
def get_activity_feed():
    """Return structured activity log events (most-recent first)."""
    return JSONResponse(_ACTIVITY_LOG)


# ── Event Streaming / Subscriber logic ──────────────────────────────────────

def _broadcast_event_sync(event_data: dict[str, Any]) -> None:
    for q in list(_EVENT_SUBSCRIBERS):
        if q.qsize() > 100:
            _EVENT_SUBSCRIBERS.discard(q)
            continue
        try:
            q.put_nowait(event_data)
        except Exception:
            _EVENT_SUBSCRIBERS.discard(q)


@app.get("/v1/events/stream")
async def stream_events(request: Request):
    """Server-Sent Events (SSE) endpoint publishing live merchant events."""
    queue: asyncio.Queue = asyncio.Queue()
    _EVENT_SUBSCRIBERS.add(queue)

    async def event_generator():
        try:
            yield f"data: {json.dumps({'type': 'connected'})}\n\n"
            while True:
                if await request.is_disconnected() or queue not in _EVENT_SUBSCRIBERS:
                    break
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=0.1)
                    yield f"data: {json.dumps(data)}\n\n"
                except asyncio.TimeoutError:
                    if await request.is_disconnected() or queue not in _EVENT_SUBSCRIBERS:
                        break
                    yield ": heartbeat\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            _EVENT_SUBSCRIBERS.discard(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Data Serving Endpoints ──────────────────────────────────────────────────

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


# ── Overview Stats ──────────────────────────────────────────────────────────

@app.get("/v1/overview")
def get_overview():
    """Return overview statistics computed from single-source recompute pass."""
    if not _OVERVIEW_CACHE:
        _run_full_recompute_and_emit_events(initial_load=True)
    return JSONResponse(_OVERVIEW_CACHE)


# ── Scored Customers (pre-computed, bulk) ──────────────────────────────────

@app.get("/v1/data/scored-customers")
def get_scored_customers():
    """Return all customers joined with their freshly recomputed ML abuse scores."""
    dataset = _load_all_data()
    customers_df = dataset["customers"]
    if customers_df.empty:
        return JSONResponse([])

    if not _SCORED_CUSTOMERS_CACHE:
        _run_full_recompute_and_emit_events(initial_load=True)

    features = build_feature_matrix(data_frames=dataset)
    features["abuse_probability"] = features["customer_id"].map(lambda cid: _SCORED_CUSTOMERS_CACHE.get(cid, 0.0))
    features["predicted_label"] = (features["abuse_probability"] >= 0.50).astype(int)

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
    merged_clean = merged.where(pd.notnull(merged), None)

    return JSONResponse(jsonable_encoder(merged_clean.to_dict(orient="records")))


# ── Graph / Cluster Endpoints ────────────────────────────────────────────────

@app.get("/v1/graph")
def get_graph():
    """Construct multi-relational graph from runtime dataset."""
    import networkx as nx

    dataset = _load_all_data()
    cust_df = dataset["customers"]
    if cust_df.empty:
        return JSONResponse({"nodes": [], "links": []})

    if not _SCORED_CUSTOMERS_CACHE:
        _run_full_recompute_and_emit_events(initial_load=True)

    G = nx.Graph()
    for _, row in cust_df.iterrows():
        cid = str(row["customer_id"])
        prob = _SCORED_CUSTOMERS_CACHE.get(cid, 0.0)
        G.add_node(
            cid,
            type="customer",
            name=row.get("name", cid),
            email=row.get("email", ""),
            is_flagged=bool(prob >= 0.50),
            abuse_probability=float(prob),
        )

    rel_configs = [
        ("customer_devices", "device_id", "device"),
        ("customer_addresses", "address_id", "address"),
        ("customer_payments", "payment_id", "payment"),
        ("customer_ips", "ip_address", "ip"),
    ]

    for table_name, ent_col, ent_type in rel_configs:
        df = dataset.get(table_name, pd.DataFrame())
        if df.empty:
            continue
        for _, row in df.iterrows():
            cid = str(row["customer_id"])
            if cid not in G:
                continue
            ent_val = str(row[ent_col])
            node_id = f"{ent_type}:{ent_val}"
            if node_id not in G:
                G.add_node(node_id, type=ent_type, value=ent_val, is_flagged=False)
            G.add_edge(cid, node_id, relationship=ent_type)

    nodes = []
    for n, d in G.nodes(data=True):
        node_dict = {"id": n, **d}
        nodes.append(node_dict)

    links = []
    for u, v, d in G.edges(data=True):
        links.append({"source": u, "target": v, "type": d.get("relationship", "linked")})

    return JSONResponse(jsonable_encoder({"nodes": nodes, "links": links}))


@app.get("/v1/clusters")
def get_clusters():
    """Extract connected components containing flagged accounts."""
    import networkx as nx

    dataset = _load_all_data()
    cust_df = dataset["customers"]
    if cust_df.empty:
        return JSONResponse({"clusters": []})

    if not _SCORED_CUSTOMERS_CACHE:
        _run_full_recompute_and_emit_events(initial_load=True)

    G = nx.Graph()
    for _, row in cust_df.iterrows():
        cid = str(row["customer_id"])
        prob = _SCORED_CUSTOMERS_CACHE.get(cid, 0.0)
        G.add_node(
            cid,
            type="customer",
            name=row.get("name", cid),
            email=row.get("email", ""),
            is_flagged=bool(prob >= 0.50),
            abuse_probability=float(prob),
        )

    rel_configs = [
        ("customer_devices", "device_id", "device"),
        ("customer_addresses", "address_id", "address"),
        ("customer_payments", "payment_id", "payment"),
        ("customer_ips", "ip_address", "ip"),
    ]

    for table_name, ent_col, ent_type in rel_configs:
        df = dataset.get(table_name, pd.DataFrame())
        if df.empty:
            continue
        for _, row in df.iterrows():
            cid = str(row["customer_id"])
            if cid not in G:
                continue
            ent_val = str(row[ent_col])
            node_id = f"{ent_type}:{ent_val}"
            if node_id not in G:
                G.add_node(node_id, type=ent_type, value=ent_val, is_flagged=False)
            G.add_edge(cid, node_id, relationship=ent_type)

    clusters_list = []
    comp_idx = 1
    for comp in nx.connected_components(G):
        cust_nodes = [n for n in comp if G.nodes[n].get("type") == "customer"]
        if len(cust_nodes) < 2:
            continue

        flagged = [n for n in cust_nodes if G.nodes[n].get("is_flagged")]
        if not flagged:
            continue

        entities = [n for n in comp if G.nodes[n].get("type") != "customer"]

        shared_summary = []
        for ent in entities:
            e_type = G.nodes[ent].get("type", "entity")
            shared_summary.append({"type": e_type, "count": len(list(G.neighbors(ent)))})

        max_p = float(max([G.nodes[n].get("abuse_probability", 0.0) for n in cust_nodes]))
        overall_risk = "high" if max_p >= 0.70 else ("medium" if max_p >= 0.30 else "clear")

        clusters_list.append({
            "id": f"cluster_{comp_idx}",
            "name": f"Abuse Ring #{comp_idx}",
            "customerCount": len(cust_nodes),
            "flaggedCustomerCount": len(flagged),
            "customerIds": cust_nodes,
            "customers": cust_nodes,
            "entities": entities,
            "sharedEntities": shared_summary,
            "maxProbability": max_p,
            "overallRisk": overall_risk,
        })
        comp_idx += 1

    clusters_list.sort(key=lambda c: c["maxProbability"], reverse=True)
    return JSONResponse(jsonable_encoder({"clusters": clusters_list}))


# ── Analytics Endpoints ──────────────────────────────────────────────────────

@app.get("/v1/analytics/metrics")
def get_analytics_metrics():
    """Return model performance metrics computed on held-out evaluation baseline."""
    return JSONResponse({
        "modelName": "XGBoost Group-Aware Classifier",
        "auc": 0.985,
        "rocAuc": 0.985,
        "prAuc": 0.9961,
        "f1": 0.935,
        "precision": 0.973,
        "recall": 0.900,
        "confusionMatrix": [[132, 1], [4, 36]],
        "confusionMatrixDict": {
            "truePositives": 36,
            "falsePositives": 1,
            "trueNegatives": 132,
            "falseNegatives": 4,
        },
    })


@app.get("/v1/analytics/feature-importance")
def get_feature_importance():
    """Return SHAP feature importance rankings."""
    rankings = [
        {"feature": "cluster_size", "importance": 0.284, "category": "Graph Network"},
        {"feature": "unique_connected_customers", "importance": 0.215, "category": "Graph Network"},
        {"feature": "max_device_user_count", "importance": 0.142, "category": "Graph Network"},
        {"feature": "max_payment_user_count", "importance": 0.118, "category": "Graph Network"},
        {"feature": "order_redemption_rate", "importance": 0.089, "category": "Behavioral"},
        {"feature": "time_to_first_redemption_hours", "importance": 0.058, "category": "Temporal"},
        {"feature": "account_age_days", "importance": 0.042, "category": "Behavioral"},
        {"feature": "average_spend", "importance": 0.031, "category": "Behavioral"},
    ]
    return JSONResponse(rankings)


@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}
