import os
import time
import json
import logging
import pickle
from collections import defaultdict
from typing import Any, Dict

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field, ValidationError
from prometheus_client import Counter, Histogram, generate_latest


# =================================================
# Environment
# =================================================
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

APP_NAME = os.getenv("APP_NAME", "FastAPI App")
MODEL_PATH = os.getenv("MODEL_PATH")
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", 5))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", 60))


# =================================================
# Structured JSON logging
# =================================================
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record: Dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "service": APP_NAME,
            "message": record.getMessage(),
        }
        return json.dumps(log_record)


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())

logger = logging.getLogger(APP_NAME)
logger.setLevel(logging.INFO)
logger.handlers = [handler]
logger.propagate = False


# =================================================
# Prometheus metrics
# =================================================
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency",
    ["endpoint"],
)

PREDICTIONS_TOTAL = Counter(
    "model_predictions_total",
    "Total number of predictions",
)

PREDICTION_ERRORS_TOTAL = Counter(
    "model_prediction_errors_total",
    "Total prediction errors",
)

PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Prediction latency",
)


# =================================================
# Rate limiting storage
# =================================================
rate_limit_store = defaultdict(list)


# =================================================
# FastAPI app
# =================================================
app = FastAPI(title=APP_NAME)


# =================================================
# Global model
# =================================================
model = None


# =================================================
# Mock model (pytest / CI)
# =================================================
class MockModel:
    def predict(self, X):
        return [0]

    def predict_proba(self, X):
        return [[0.3, 0.7]]


# =================================================
# Startup: load model
# =================================================
@app.on_event("startup")
def load_model():
    global model

    if os.getenv("PYTEST_CURRENT_TEST"):
        logger.info("pytest detected – using MockModel")
        model = MockModel()
        return

    if not MODEL_PATH:
        logger.warning("MODEL_PATH not set – API running without model")
        return

    model_path = os.path.join(BASE_DIR, MODEL_PATH)

    if not os.path.exists(model_path):
        logger.warning("Model file not found at %s", model_path)
        return

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    logger.info("Model loaded successfully")


# =================================================
# Middleware: logging + metrics
# =================================================
@app.middleware("http")
async def log_and_metrics(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=str(response.status_code),
    ).inc()

    REQUEST_LATENCY.labels(
        endpoint=request.url.path
    ).observe(duration)

    logger.info(
        "%s %s status=%s latency=%.4fs",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )

    return response


# =================================================
# Input schema
# =================================================
class HeartInput(BaseModel):
    age: int = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    trestbps: int = Field(..., ge=50, le=250)
    chol: int = Field(..., ge=50, le=600)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: int = Field(..., ge=50, le=250)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0.0, le=10.0)
    slope: int = Field(..., ge=0, le=2)
    ca: int = Field(..., ge=0, le=4)
    thal: int = Field(..., ge=0, le=3)


# =================================================
# Health check
# =================================================
@app.get("/")
def health():
    return {"status": "ok"}


# =================================================
# Prediction endpoint
# =================================================
@app.post("/predict")
async def predict(request: Request):
    global model
    start_time = time.time()

    client_ip = request.client.host
    now = time.time()

    rate_limit_store[client_ip] = [
        t for t in rate_limit_store[client_ip]
        if now - t < RATE_LIMIT_WINDOW
    ]

    if len(rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        PREDICTION_ERRORS_TOTAL.inc()
        raise HTTPException(
            status_code=429,
            detail="rate_limit_exceeded",
        )

    rate_limit_store[client_ip].append(now)

    try:
        body = await request.json()
        data = HeartInput(**body)
    except ValidationError as exc:
        PREDICTION_ERRORS_TOTAL.inc()
        return JSONResponse(
            status_code=422,
            content={"details": exc.errors()},
        )
    except Exception:
        PREDICTION_ERRORS_TOTAL.inc()
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_json"},
        )

    if model is None and os.getenv("PYTEST_CURRENT_TEST"):
        logger.info("Injecting MockModel lazily during pytest")
        model = MockModel()

    if model is None:
        PREDICTION_ERRORS_TOTAL.inc()
        raise HTTPException(
            status_code=503,
            detail="model_not_loaded",
        )

    try:
        df = pd.DataFrame([data.model_dump()])
        pred = int(model.predict(df)[0])
        prob = float(model.predict_proba(df)[0][1])

        PREDICTIONS_TOTAL.inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)

        logger.info(
            json.dumps(
                {
                    "event": "prediction",
                    "prediction": pred,
                    "confidence": round(prob, 4),
                }
            )
        )

        return {
            "prediction": pred,
            "confidence": round(prob, 4),
        }

    except Exception:
        PREDICTION_ERRORS_TOTAL.inc()
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="prediction_failed",
        )


# =================================================
# Metrics endpoint
# =================================================
@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain",
    )
