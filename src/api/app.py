import os
import time
import logging
import pickle
import pandas as pd

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field, ValidationError
from prometheus_client import Counter, Histogram, generate_latest
from dotenv import load_dotenv
from collections import defaultdict


# -------------------------------------------------
# Load environment variables (FIXED FOR WINDOWS)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

APP_NAME = os.getenv("APP_NAME", "FastAPI App")
MODEL_PATH = os.getenv("MODEL_PATH")
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", 5))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", 60))

if not MODEL_PATH:
    raise RuntimeError("MODEL_PATH not set in environment variables")

MODEL_PATH = os.path.join(BASE_DIR, MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at: {MODEL_PATH}")


# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(APP_NAME)

# -------------------------------------------------
# Prometheus metrics
# -------------------------------------------------
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency",
    ["endpoint"]
)

# -------------------------------------------------
# Rate limiting storage (in-memory)
# -------------------------------------------------
rate_limit_store = defaultdict(list)

# -------------------------------------------------
# FastAPI app
# -------------------------------------------------
app = FastAPI(title=APP_NAME)

# -------------------------------------------------
# Middleware: logging + metrics
# -------------------------------------------------
@app.middleware("http")
async def log_and_metrics(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    logger.info(
        f"{request.method} {request.url.path} | "
        f"Status {response.status_code} | "
        f"{duration:.4f}s"
    )

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=str(response.status_code)
    ).inc()

    REQUEST_LATENCY.labels(
        endpoint=request.url.path
    ).observe(duration)

    return response

# -------------------------------------------------
# Load trained model
# -------------------------------------------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# -------------------------------------------------
# Input schema with validation
# -------------------------------------------------
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

# -------------------------------------------------
# Health check
# -------------------------------------------------
@app.get("/")
def health():
    return {"status": "API is running"}

# -------------------------------------------------
# Prediction endpoint with rate limiting
# -------------------------------------------------
@app.post("/predict")
async def predict(request: Request):
    client_ip = request.client.host
    now = time.time()

    # -------- Rate limiting --------
    rate_limit_store[client_ip] = [
        t for t in rate_limit_store[client_ip]
        if now - t < RATE_LIMIT_WINDOW
    ]

    if len(rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        logger.warning(f"Rate limit exceeded | IP={client_ip}")
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limit_exceeded",
                "message": f"Max {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds"
            }
        )

    rate_limit_store[client_ip].append(now)

    # -------- JSON + validation --------
    try:
        body = await request.json()
        data = HeartInput(**body)
    except ValidationError as e:
        logger.warning(f"Validation ERROR | IP={client_ip} | {e.errors()}")
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "details": e.errors()
            }
        )
    except Exception:
        logger.warning(f"Invalid JSON | IP={client_ip}")
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_json",
                "message": "Request body must be valid JSON"
            }
        )

    # -------- Prediction --------
    try:
        df = pd.DataFrame([data.dict()])
        pred = int(model.predict(df)[0])
        prob = float(model.predict_proba(df)[0][1])

        logger.info(
            f"Prediction SUCCESS | IP={client_ip} | pred={pred} | prob={prob:.4f}"
        )

        return {
            "prediction": pred,
            "confidence": round(prob, 4)
        }

    except Exception as e:
        logger.exception("Prediction FAILURE")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "prediction_failed",
                "message": "Internal model error"
            }
        )

# -------------------------------------------------
# Prometheus metrics endpoint
# -------------------------------------------------
@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain"
    )
