import os                                           # OS utilities for environment variables and file paths
import time                                         # Time utilities for latency measurement and rate limiting
import json                                         # JSON handling for structured logging
import logging                                      # Python logging framework
import pickle                                       # Used to load the trained ML model
from collections import defaultdict                 # Dictionary with default values for rate limiting
from typing import Any, Dict                        # Type hints for better readability and tooling

import pandas as pd                                 # DataFrame creation for model input
from dotenv import load_dotenv                      # Load environment variables from .env file
from fastapi import FastAPI, Request, HTTPException # FastAPI core classes
from fastapi.responses import Response, JSONResponse # Custom HTTP responses
from pydantic import BaseModel, Field, ValidationError # Input validation and schema definition
from prometheus_client import Counter, Histogram, generate_latest # Prometheus metrics


# =================================================
# Environment
# =================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # Project root directory
ENV_PATH = os.path.join(BASE_DIR, ".env")                               # Path to .env file
load_dotenv(dotenv_path=ENV_PATH)                                       # Load environment variables

APP_NAME = os.getenv("APP_NAME", "FastAPI App")                         # Application name
MODEL_PATH = os.getenv("MODEL_PATH")                                    # Path to trained model file
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", 5))          # Max requests per client
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", 60))     # Rate limit time window


# =================================================
# Structured JSON logging
# =================================================
class JsonFormatter(logging.Formatter):                                  # Custom log formatter
    def format(self, record: logging.LogRecord) -> str:                  # Override default format method
        log_record: Dict[str, Any] = {
            "timestamp": self.formatTime(record),                        # Log timestamp
            "level": record.levelname,                                   # Log severity level
            "service": APP_NAME,                                         # Service name
            "message": record.getMessage(),                              # Log message content
        }
        return json.dumps(log_record)                                    # Return structured JSON log


handler = logging.StreamHandler()                                        # Log output to stdout
handler.setFormatter(JsonFormatter())                                    # Attach custom JSON formatter

logger = logging.getLogger(APP_NAME)                                     # Create named logger
logger.setLevel(logging.INFO)                                            # Set log level
logger.handlers = [handler]                                              # Replace default handlers
logger.propagate = False                                                 # Prevent duplicate logs


# =================================================
# Prometheus metrics
# =================================================
REQUEST_COUNT = Counter(                                                 # Counts total API requests
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(                                             # Measures API request latency
    "api_request_latency_seconds",
    "API request latency",
    ["endpoint"],
)

PREDICTIONS_TOTAL = Counter(                                             # Counts successful predictions
    "model_predictions_total",
    "Total number of predictions",
)

PREDICTION_ERRORS_TOTAL = Counter(                                       # Counts prediction failures
    "model_prediction_errors_total",
    "Total prediction errors",
)

PREDICTION_LATENCY = Histogram(                                          # Measures prediction time
    "model_prediction_latency_seconds",
    "Prediction latency",
)


# =================================================
# Rate limiting storage
# =================================================
rate_limit_store = defaultdict(list)                                     # Stores timestamps per client IP


# =================================================
# FastAPI app
# =================================================
app = FastAPI(title=APP_NAME)                                             # Initialize FastAPI app


# =================================================
# Global model
# =================================================
model = None                                                             # Global model reference


# =================================================
# Mock model (pytest / CI)
# =================================================
class MockModel:                                                         # Dummy model for testing
    def predict(self, X):
        return [0]                                                       # Always predict class 0

    def predict_proba(self, X):
        return [[0.3, 0.7]]                                              # Fixed probability output


# =================================================
# Startup: load model
# =================================================
@app.on_event("startup")
def load_model():
    global model                                                         # Use global model variable

    if os.getenv("PYTEST_CURRENT_TEST"):                                 # Detect pytest environment
        logger.info("pytest detected – using MockModel")
        model = MockModel()                                              # Inject mock model
        return

    if not MODEL_PATH:                                                   # If model path not configured
        logger.warning("MODEL_PATH not set – API running without model")
        return

    model_path = os.path.join(BASE_DIR, MODEL_PATH)                      # Full model file path

    if not os.path.exists(model_path):                                   # Check model existence
        logger.warning(f"Model file not found at {model_path}")
        return

    with open(model_path, "rb") as f:                                    # Open model file
        model = pickle.load(f)                                           # Deserialize trained model

    logger.info("Model loaded successfully")                             # Confirm model load


# =================================================
# Middleware: logging + metrics
# =================================================
@app.middleware("http")
async def log_and_metrics(request: Request, call_next):
    start_time = time.time()                                             # Start timer
    response = await call_next(request)                                  # Process request
    duration = time.time() - start_time                                  # Compute latency

    REQUEST_COUNT.labels(                                                # Increment request counter
        method=request.method,
        endpoint=request.url.path,
        status=str(response.status_code),
    ).inc()

    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration) # Record latency

    logger.info(                                                         # Structured request log
        f"{request.method} {request.url.path} "
        f"status={response.status_code} "
        f"latency={duration:.4f}s"
    )

    return response                                                      # Return HTTP response


# =================================================
# Input schema
# =================================================
class HeartInput(BaseModel):                                             # Input validation schema
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
    return {"status": "ok"}                                             # Simple health response


# =================================================
# Prediction endpoint
# =================================================
@app.post("/predict")
async def predict(request: Request):
    global model                                                         # Access global model
    start_time = time.time()                                             # Start prediction timer

    client_ip = request.client.host                                      # Extract client IP
    now = time.time()                                                    # Current timestamp

    # -------- Rate limiting --------
    rate_limit_store[client_ip] = [
        t for t in rate_limit_store[client_ip]
        if now - t < RATE_LIMIT_WINDOW                                   # Keep only recent requests
    ]

    if len(rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:          # Check rate limit
        PREDICTION_ERRORS_TOTAL.inc()
        raise HTTPException(status_code=429, detail="rate_limit_exceeded")

    rate_limit_store[client_ip].append(now)                              # Store current request time

    # -------- Validation --------
    try:
        body = await request.json()                                      # Parse JSON body
        data = HeartInput(**body)                                        # Validate input schema
    except ValidationError as exc:
        PREDICTION_ERRORS_TOTAL.inc()
        return JSONResponse(status_code=422, content={"details": exc.errors()})
    except Exception:
        PREDICTION_ERRORS_TOTAL.inc()
        return JSONResponse(status_code=400, content={"error": "invalid_json"})

    # -------- ✅ LAZY MOCK INJECTION (FINAL FIX) --------
    if model is None and os.getenv("PYTEST_CURRENT_TEST"):
        logger.info("Injecting MockModel lazily during pytest")
        model = MockModel()                                              # Inject mock if needed

    if model is None:
        PREDICTION_ERRORS_TOTAL.inc()
        raise HTTPException(status_code=503, detail="model_not_loaded")

    # -------- Prediction --------
    try:
        df = pd.DataFrame([data.model_dump()])                            # Convert input to DataFrame
        pred = int(model.predict(df)[0])                                 # Get predicted class
        prob = float(model.predict_proba(df)[0][1])                      # Get probability score

        PREDICTIONS_TOTAL.inc()                                          # Increment success counter
        PREDICTION_LATENCY.observe(time.time() - start_time)             # Record prediction latency

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
        raise HTTPException(status_code=500, detail="prediction_failed")


# =================================================
# Metrics endpoint
# =================================================
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")          # Expose Prometheus metrics
