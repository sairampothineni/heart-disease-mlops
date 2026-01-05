import pytest
from httpx import AsyncClient
from httpx import ASGITransport

from src.api.app import app


@pytest.mark.asyncio
async def test_predict_success():
    payload = {
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

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test"
    ) as client:
        response = await client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()

    assert "prediction" in body
    assert "confidence" in body
    assert body["prediction"] in [0, 1]
    assert 0.0 <= body["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_predict_missing_field():
    payload = {
        "age": 63,
        "sex": 1
    }

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test"
    ) as client:
        response = await client.post("/predict", json=payload)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_out_of_range():
    payload = {
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

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test"
    ) as client:
        response = await client.post("/predict", json=payload)

    assert response.status_code == 422
