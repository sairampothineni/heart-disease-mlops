import pytest                                  # Pytest framework for writing and running tests
from httpx import AsyncClient                  # Async HTTP client for testing FastAPI endpoints
from httpx import ASGITransport                # ASGI transport to run FastAPI app in-memory

from src.api.app import app                    # Import the FastAPI application instance


@pytest.mark.asyncio                           # Marks this test as asynchronous
async def test_predict_success():
    payload = {                                # Valid request payload with all required fields
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1
    }

    transport = ASGITransport(app=app)         # Create in-memory transport using FastAPI app

    async with AsyncClient(                    # Initialize async HTTP client
        transport=transport,
        base_url="http://test"
    ) as client:
        response = await client.post("/predict", json=payload)  # Send POST request to /predict

    assert response.status_code == 200          # Expect successful HTTP response
    body = response.json()                     # Parse JSON response body

    assert "prediction" in body                # Ensure prediction field is present
    assert "confidence" in body                # Ensure confidence field is present
    assert body["prediction"] in [0, 1]        # Prediction must be binary
    assert 0.0 <= body["confidence"] <= 1.0    # Confidence must be a valid probability


@pytest.mark.asyncio                           # Marks this test as asynchronous
async def test_predict_missing_field():
    payload = {                                # Incomplete payload (missing required fields)
        "age": 63,
        "sex": 1
    }

    transport = ASGITransport(app=app)         # Create in-memory ASGI transport

    async with AsyncClient(                    # Initialize async HTTP client
        transport=transport,
        base_url="http://test"
    ) as client:
        response = await client.post("/predict", json=payload)  # Send invalid request

    assert response.status_code == 422          # Expect validation error (Unprocessable Entity)


@pytest.mark.asyncio                           # Marks this test as asynchronous
async def test_predict_out_of_range():
    payload = {                                # Payload with out-of-range value
        "age": 999,  # invalid
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1
    }

    transport = ASGITransport(app=app)         # Create ASGI transport bound to the app

    async with AsyncClient(                    # Initialize async HTTP client
        transport=transport,
        base_url="http://test"
    ) as client:
        response = await client.post("/predict", json=payload)  # Send invalid request

    assert response.status_code == 422          # Expect validation error due to invalid age
