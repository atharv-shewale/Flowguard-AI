"""
test_api.py
-----------
Unit tests for FlowGuard AI FastAPI endpoints.

Run with:
  pytest tests/test_api.py -v

These tests use FastAPI's TestClient (no server needed).
"""

import pytest
from fastapi.testclient import TestClient

# ── Patch model loading before importing main ─────────────────────────────────
# This ensures tests run even without trained model files present.
import unittest.mock as mock

# Create a mock model that returns predictable results
def _mock_predict(features):
    return {
        "risk_level": "Medium",
        "risk_score": 50.0,
        "confidence": 0.75,
        "class_probabilities": {"High": 0.10, "Low": 0.15, "Medium": 0.75},
        "insights": [
            {"metric": "Expense Ratio", "status": "Warning", "message": "Expenses are 63% of revenue."},
            {"metric": "Cash Runway", "status": "Warning", "message": "Cash runway is 1.5 months."},
            {"metric": "Payment Delay", "status": "Warning", "message": "Average payment delay is 18 days."},
            {"metric": "Payroll Burden", "status": "Warning", "message": "Payroll is 38% of revenue."},
            {"metric": "Vendor Obligations", "status": "Warning", "message": "Vendor dues are 29% of revenue."},
        ],
        "recommendation": "Moderate risk detected. Prioritise collecting outstanding invoices.",
    }

def _mock_load_model():
    return True

def _mock_get_model_info():
    return {
        "model_name":     "RandomForestClassifier",
        "model_version":  "1.0.0",
        "accuracy":       0.945,
        "f1_score":       0.943,
        "feature_columns": [
            "monthly_revenue", "pending_invoices", "avg_payment_delay",
            "monthly_expenses", "payroll_ratio", "cash_reserve", "vendor_due_amount",
        ],
        "target_classes": ["Low", "Medium", "High"],
        "trained_at":     "2024-01-01T12:00:00",
        "mlflow_run_id":  "abc123",
    }

# Apply mocks
with (
    mock.patch("app.model.load_model", _mock_load_model),
    mock.patch("app.model.predict", _mock_predict),
    mock.patch("app.model.get_model_info", _mock_get_model_info),
    mock.patch("app.model._model_loaded", True),
):
    from app.main import app

client = TestClient(app)


# ─────────────────────────────────────────────────────────────────────────────
# Sample payloads
# ─────────────────────────────────────────────────────────────────────────────

LOW_RISK_PAYLOAD = {
    "monthly_revenue": 130_000,
    "pending_invoices": 8_000,
    "avg_payment_delay": 4,
    "monthly_expenses": 60_000,
    "payroll_ratio": 0.22,
    "cash_reserve": 90_000,
    "vendor_due_amount": 6_000,
    "business_name": "Healthy Corp",
}

HIGH_RISK_PAYLOAD = {
    "monthly_revenue": 28_000,
    "pending_invoices": 55_000,
    "avg_payment_delay": 45,
    "monthly_expenses": 38_000,
    "payroll_ratio": 0.72,
    "cash_reserve": 3_000,
    "vendor_due_amount": 48_000,
    "business_name": "Struggling Co",
}


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthCheck:
    def test_health_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_health_response_structure(self):
        response = client.get("/")
        data = response.json()
        assert "status" in data
        assert "service" in data
        assert "version" in data
        assert "model_loaded" in data

    def test_health_status_is_healthy(self):
        response = client.get("/")
        assert response.json()["status"] == "healthy"


class TestPredictRisk:
    def test_predict_returns_200(self):
        with mock.patch("app.model.predict", _mock_predict):
            response = client.post("/predict-risk", json=LOW_RISK_PAYLOAD)
        assert response.status_code == 200

    def test_predict_response_has_required_fields(self):
        with mock.patch("app.model.predict", _mock_predict):
            response = client.post("/predict-risk", json=LOW_RISK_PAYLOAD)
        data = response.json()
        assert "risk_level" in data
        assert "risk_score" in data
        assert "confidence" in data
        assert "class_probabilities" in data
        assert "insights" in data
        assert "recommendation" in data

    def test_risk_level_is_valid(self):
        with mock.patch("app.model.predict", _mock_predict):
            response = client.post("/predict-risk", json=LOW_RISK_PAYLOAD)
        data = response.json()
        assert data["risk_level"] in ["Low", "Medium", "High"]

    def test_risk_score_in_range(self):
        with mock.patch("app.model.predict", _mock_predict):
            response = client.post("/predict-risk", json=HIGH_RISK_PAYLOAD)
        data = response.json()
        assert 0 <= data["risk_score"] <= 100

    def test_confidence_in_range(self):
        with mock.patch("app.model.predict", _mock_predict):
            response = client.post("/predict-risk", json=LOW_RISK_PAYLOAD)
        data = response.json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_business_name_passed_through(self):
        with mock.patch("app.model.predict", _mock_predict):
            response = client.post("/predict-risk", json=LOW_RISK_PAYLOAD)
        data = response.json()
        assert data.get("business_name") == "Healthy Corp"

    def test_invalid_payroll_ratio_returns_422(self):
        bad_payload = LOW_RISK_PAYLOAD.copy()
        bad_payload["payroll_ratio"] = 1.5   # invalid: > 1.0
        response = client.post("/predict-risk", json=bad_payload)
        assert response.status_code == 422

    def test_missing_required_field_returns_422(self):
        incomplete = {k: v for k, v in LOW_RISK_PAYLOAD.items() if k != "monthly_revenue"}
        response = client.post("/predict-risk", json=incomplete)
        assert response.status_code == 422

    def test_negative_revenue_returns_422(self):
        bad_payload = LOW_RISK_PAYLOAD.copy()
        bad_payload["monthly_revenue"] = -1000
        response = client.post("/predict-risk", json=bad_payload)
        assert response.status_code == 422

    def test_insights_is_list(self):
        with mock.patch("app.model.predict", _mock_predict):
            response = client.post("/predict-risk", json=LOW_RISK_PAYLOAD)
        data = response.json()
        assert isinstance(data["insights"], list)
        assert len(data["insights"]) > 0


class TestBatchPredict:
    def test_batch_predict_returns_200(self):
        payload = {"businesses": [LOW_RISK_PAYLOAD, HIGH_RISK_PAYLOAD]}
        with mock.patch("app.model.predict", _mock_predict):
            response = client.post("/batch-predict", json=payload)
        assert response.status_code == 200

    def test_batch_returns_correct_count(self):
        payload = {"businesses": [LOW_RISK_PAYLOAD, HIGH_RISK_PAYLOAD]}
        with mock.patch("app.model.predict", _mock_predict):
            response = client.post("/batch-predict", json=payload)
        data = response.json()
        assert data["total"] == 2

    def test_batch_response_has_summary(self):
        payload = {"businesses": [LOW_RISK_PAYLOAD]}
        with mock.patch("app.model.predict", _mock_predict):
            response = client.post("/batch-predict", json=payload)
        data = response.json()
        assert "summary" in data


class TestModelInfo:
    def test_model_info_returns_200(self):
        with mock.patch("app.model.get_model_info", _mock_get_model_info):
            response = client.get("/model-info")
        assert response.status_code == 200

    def test_model_info_has_accuracy(self):
        with mock.patch("app.model.get_model_info", _mock_get_model_info):
            response = client.get("/model-info")
        data = response.json()
        assert "accuracy" in data
        assert isinstance(data["accuracy"], float)

    def test_model_info_has_feature_columns(self):
        with mock.patch("app.model.get_model_info", _mock_get_model_info):
            response = client.get("/model-info")
        data = response.json()
        assert "feature_columns" in data
        assert len(data["feature_columns"]) == 7


class TestUIDashboard:
    def test_ui_returns_200_html(self):
        response = client.get("/ui")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "FlowGuard AI" in response.text

