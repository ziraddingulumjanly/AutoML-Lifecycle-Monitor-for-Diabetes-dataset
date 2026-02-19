from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from api.schemas import PredictRequest, PredictResponse
from api.model_loader import load_latest, validate_input
from api.logging_setup import setup_json_logging

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
PRED_LOG = LOG_DIR / "predictions.jsonl"

app = FastAPI(title="MLOps Inference API", version="1.0.0")
logger = setup_json_logging()

REQ_COUNT = Counter("http_requests_total", "Total HTTP requests", ["endpoint", "status"])
REQ_LAT = Histogram("http_request_latency_seconds", "Request latency", ["endpoint"])

_loaded = {"model": None}

def get_model():
    if _loaded["model"] is None:
        _loaded["model"] = load_latest()
        logger.info({"event": "model_loaded", "version": _loaded["model"].version})
    return _loaded["model"]

@app.get("/health")
def health():
    REQ_COUNT.labels(endpoint="/health", status="200").inc()
    return {"status": "ok"}

@app.get("/model_info")
def model_info():
    m = get_model()
    REQ_COUNT.labels(endpoint="/model_info", status="200").inc()
    return {
        "version": m.version,
        "metadata": m.metadata,
        "metrics": m.metrics.get("test_report"),
    }

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    endpoint = "/predict"
    with REQ_LAT.labels(endpoint=endpoint).time():
        try:
            m = get_model()
            records = req.records
            if isinstance(records, dict):
                rows = [records]
            else:
                rows = records

            df = pd.DataFrame(rows)
            validate_input(df, m.metadata)

            y = m.predict_df(df)
            preds = [float(v) for v in y.tolist()]

            # log predictions
            for r, p in zip(rows, preds):
                PRED_LOG.open("a", encoding="utf-8").write(json.dumps({
                    "ts": pd.Timestamp.utcnow().isoformat(),
                    "model_version": m.version,
                    "features": r,
                    "prediction": p,
                }) + "\n")

            REQ_COUNT.labels(endpoint=endpoint, status="200").inc()
            return PredictResponse(version=m.version, predictions=preds)

        except Exception as e:
            logger.error({"event": "predict_error", "error": str(e)})
            REQ_COUNT.labels(endpoint=endpoint, status="500").inc()
            raise HTTPException(status_code=400, detail=str(e))
