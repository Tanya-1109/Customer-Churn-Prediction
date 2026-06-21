"""Integration tests for the FastAPI churn prediction service."""

import pytest
from fastapi.testclient import TestClient
from app.api.main import app

SAMPLE_PAYLOAD = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": "255.0",
}


@pytest.fixture(scope="module")
def client():
    """Yields a TestClient with lifespan (startup/shutdown) properly triggered."""
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_returns_valid_probability(client):
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["churn_probability"] <= 1
    assert data["churn_prediction"] in ["High Risk", "Low Risk"]
    assert len(data["top_risk_factors"]) == 3


def test_predict_rejects_invalid_input(client):
    bad_payload = SAMPLE_PAYLOAD.copy()
    bad_payload["InternetService"] = "Fibre optic"  # typo, invalid Literal
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
