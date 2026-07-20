"""
service.py
----------
BentoML Service for FlowGuard AI (BentoML 1.2+ / 1.4+ Class-based Syntax).

BentoML packages the trained ML model into a self-contained, production-ready
service.

Run this service:
  cd flowguard-ai
  bentoml serve bentoml/service.py:svc --reload --port 3000
"""

import os
import sys
import pickle
import numpy as np
import bentoml
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

# ─────────────────────────────────────────────────────────────────────────────
# Input / Output schemas (Pydantic models)
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
# Modern BentoML Class-Based Service Definition
# ─────────────────────────────────────────────────────────────────────────────

@bentoml.service(
    name="flowguard_ai_service",
)
class FlowGuardAIService:
    def __init__(self):
        print("[BENTO] Initializing FlowGuard AI Service...")
        
        # Load sklearn model directly from pickle
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Run training first.")
        
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        print("[OK] Loaded model successfully.")

        # Load scaler
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)
        print("[OK] Loaded scaler successfully.")

        # Load label encoder
        with open(ENCODER_PATH, "rb") as f:
            self.encoder = pickle.load(f)
        print("[OK] Loaded label encoder successfully.")

    @bentoml.api
    def predict(self, input_data: RiskInput) -> RiskOutput:
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
        x_scaled = self.scaler.transform(x)

        # Run prediction
        pred_idx = self.model.predict(x_scaled)
        proba    = self.model.predict_proba(x_scaled)

        pred_idx = int(pred_idx[0])
        proba    = proba[0]

        # Decode label
        classes    = self.encoder.classes_  # ['High', 'Low', 'Medium']
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

# Export the service object for uvicorn/bentoml serve
svc = FlowGuardAIService
