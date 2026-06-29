"""
app.py
──────
FastAPI service — deploy directly on HuggingFace Spaces (SDK: fastapi).
Endpoints:
  GET  /              → welcome message
  GET  /health        → health check
  POST /forecast/send     → 48-hour forecast for outgoing transactions
  POST /forecast/receive  → 48-hour forecast for incoming transactions
"""

import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from forecast import recursive_forecast

# ── Load models at startup ────────────────────────────────────────────────────
MODELS_DIR = "models"

def _load(filename):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise RuntimeError(
            f"Missing model file: {path}\n"
            "→ Run train.py first to generate all .pkl files."
        )
    return joblib.load(path)

model_amount_send    = _load("model_amount_send.pkl")
model_count_send     = _load("model_count_send.pkl")
model_amount_receive = _load("model_amount_receive.pkl")
model_count_receive  = _load("model_count_receive.pkl")
feature_cols         = _load("feature_columns.pkl")

# ── Schemas ───────────────────────────────────────────────────────────────────
class HourRow(BaseModel):
    Time:     str   = Field(..., example="2024-01-01T14:00:00")
    Amount:   float = Field(..., gt=0)
    tx_count: Optional[int] = Field(default=1, ge=1)

class ForecastRequest(BaseModel):
    history: List[HourRow] = Field(
        ..., min_items=24,
        description="Hourly aggregated history — minimum 24 rows required."
    )
    steps: Optional[int] = Field(default=48, ge=1, le=168)

class ForecastPoint(BaseModel):
    Time:        str
    pred_amount: float
    pred_count:  int

class ForecastResponse(BaseModel):
    type:     str
    steps:    int
    forecast: List[ForecastPoint]

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Transaction Forecasting API",
    description=(
        "48-hour ahead forecast for Send and Receive transaction volumes.\n\n"
        "Built for GP project — Ain Shams University."
    ),
    version="1.0.0",
)


def _run(request: ForecastRequest, tx_type: str, model_amount, model_count):
    df = pd.DataFrame([r.dict() for r in request.history])
    df = df.rename(columns={"Time": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df["Amount"]   = df["Amount"].astype(float)
    df["tx_count"] = df["tx_count"].fillna(1).astype(int)
    df = df.sort_values("Datetime").reset_index(drop=True)

    result = recursive_forecast(
        df, model_amount, model_count, feature_cols, steps=request.steps
    )
    return ForecastResponse(type=tx_type, steps=request.steps, forecast=result)


@app.get("/")
def root():
    return {
        "message": "Transaction Forecasting API is running.",
        "docs":    "/docs",
        "endpoints": ["/forecast/send", "/forecast/receive"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/forecast/send", response_model=ForecastResponse)
def forecast_send(request: ForecastRequest):
    """Forecast OUTGOING (send) transaction volume for the next N hours."""
    return _run(request, "send", model_amount_send, model_count_send)


@app.post("/forecast/receive", response_model=ForecastResponse)
def forecast_receive(request: ForecastRequest):
    """Forecast INCOMING (receive) transaction volume for the next N hours."""
    return _run(request, "receive", model_amount_receive, model_count_receive)
