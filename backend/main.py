"""Minimal HTTP boundary around the frozen ML inference function."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ml.inference import score_customer


class HistoricalData(BaseModel):
    """A row from one historical source table.

    The feature pipeline owns the source-table schemas. Keeping rows as
    dictionaries here lets the API pass through that existing schema without
    duplicating feature or ML logic in the backend.
    """

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


app = FastAPI(title="Offer Abuse Detection API", version="1.0.0")


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
    except Exception as exc:  # pragma: no cover - safety boundary for unexpected ML failures
        raise HTTPException(status_code=500, detail="ML inference failed") from exc
    return PredictionResponse.model_validate(result)
