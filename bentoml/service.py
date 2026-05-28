"""
service.py
----------
BentoML Service for FlowGuard AI.

BentoML packages the trained ML model into a self-contained, production-ready
service that can be:
  - Run locally:  bentoml serve bentoml/service.py:svc
  - Built into a Docker image: bentoml build
  - Deployed to BentoCloud or Kubernetes

How it works:
  1. We load the saved model using a BentoML Runner
  2. We define a Service with a /predict endpoint
  3. The service handles HTTP I/O, schema validation, and inference

Run this service:
  cd flowguard-ai
  bentoml serve bentoml/service.py:svc --reload --port 3000
"""

import os
import sys
import pickle
import numpy as np
import bentoml
from bentoml.io import JSON
from pydantic import BaseModel, Field
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Add project root to Python path so we can import from app/
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# Model artifact paths
# ─────────────────────────────────────────────────────────────────────────────
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH   = os.path.join(MODELS_DIR, "saved_model.pkl")
SCALER_PATH  = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

FEATURE_COLS = [
    "monthly_revenue",
    "pending_invoices",
    "avg_payment_delay",
    "monthly_expenses",
    "payroll_ratio",
    "cash_reserve",
    "vendor_due_amount",
]


# ─────────────────────────────────────────────────────────────────────────────
# Register / retrieve the model in BentoML's local model store
# ─────────────────────────────────────────────────────────────────────────────
def register_model_in_bentoml():
    """
    Save the sklearn model into BentoML's model store (only once).
    BentoML stores models in ~/.bentoml/models/ with versioning.
    """
    try:
        # Check if already saved to avoid duplicate registration
        bentoml.sklearn.get("flowguard_risk_model:latest")
        print("ℹ️   Model already registered in BentoML store.")
    except bentoml.exceptions.NotFound:
        print("📦  Registering model in BentoML model store…")
        with open(MODEL_PATH, "rb") as f:
            sklearn_model = pickle.load(f)
        bentoml.sklearn.save_model(
            "flowguard_risk_model",
            sklearn_model,
            metadata={
                "description": "FlowGuard AI – SME Cash Flow Risk Classifier",
                "feature_columns": FEATURE_COLS,
                "target_classes": ["High", "Low", "Medium"],
            },
        )
        print("✅  Model registered as 'flowguard_risk_model:latest'")


# Register the model when module loads
register_model_in_bentoml()

# ─────────────────────────────────────────────────────────────────────────────
# Load supporting artifacts (scaler and encoder) globally
# ─────────────────────────────────────────────────────────────────────────────
with open(SCALER_PATH, "rb") as f:
    _scaler = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    _encoder = pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# BentoML Runner – wraps the model for async/parallel inference
# ─────────────────────────────────────────────────────────────────────────────
# Runners allow BentoML to scale the model independently of the API layer.
flowguard_runner = bentoml.sklearn.get("flowguard_risk_model:latest").to_runner()

# ─────────────────────────────────────────────────────────────────────────────
# BentoML Service definition
# ─────────────────────────────────────────────────────────────────────────────
svc = bentoml.Service(
    name="flowguard_ai_service",
    runners=[flowguard_runner],
)


# ─────────────────────────────────────────────────────────────────────────────
# Input / Output schemas (Pydantic models used inline)
# ─────────────────────────────────────────────────────────────────────────────

class RiskInput(BaseModel):
    monthly_revenue:   float = Field(..., example=85_000.0)
    pending_invoices:  float = Field(..., example=22_000.0)
    avg_payment_delay: float = Field(..., example=18.0)
    monthly_expenses:  float = Field(..., example=54_000.0)
    payroll_ratio:     float = Field(..., example=0.38)
    cash_reserve:      float = Field(..., example=18_000.0)
    vendor_due_amount: float = Field(..., example=25_000.0)
    business_name:     Optional[str] = Field(None, example="Cape Bakery Ltd")


class RiskOutput(BaseModel):
    risk_level:   str
    risk_score:   float
    confidence:   float
    probabilities: dict
    business_name: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# API Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@svc.api(
    input=JSON(pydantic_model=RiskInput),
    output=JSON(pydantic_model=RiskOutput),
    route="/predict",
)
async def predict(input_data: RiskInput) -> RiskOutput:
    """
    BentoML prediction endpoint.

    Accepts financial features of a single business and returns
    the cash-flow risk classification with score and confidence.
    """
    # Build feature array in correct column order
    x = np.array([[
        input_data.monthly_revenue,
        input_data.pending_invoices,
        input_data.avg_payment_delay,
        input_data.monthly_expenses,
        input_data.payroll_ratio,
        input_data.cash_reserve,
        input_data.vendor_due_amount,
    ]])

    # Scale features
    x_scaled = _scaler.transform(x)

    # Run prediction via the BentoML runner (async)
    pred_idx = await flowguard_runner.predict.async_run(x_scaled)
    proba    = await flowguard_runner.predict_proba.async_run(x_scaled)

    pred_idx = int(pred_idx[0])
    proba    = proba[0]

    # Decode label
    classes    = _encoder.classes_  # ['High', 'Low', 'Medium']
    risk_level = classes[pred_idx]
    confidence = float(proba[pred_idx])

    # Build probabilities dict and risk score
    class_probs = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
    WEIGHT      = {"High": 100, "Medium": 50, "Low": 0}
    risk_score  = sum(class_probs[c] * WEIGHT[c] for c in class_probs)

    return RiskOutput(
        risk_level    = risk_level,
        risk_score    = round(risk_score, 2),
        confidence    = round(confidence, 4),
        probabilities = class_probs,
        business_name = input_data.business_name,
    )
