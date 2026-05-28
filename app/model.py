"""
model.py
--------
Model loading and inference logic for FlowGuard AI.

This module:
  - Loads the trained model, scaler, and label encoder from disk
  - Provides a predict() function used by FastAPI endpoints
  - Generates human-readable insights based on financial values
  - Computes a risk score (0–100) from class probabilities
  - Implements a rule-based FALLBACK if the model file is not found

Design principle: keep all ML logic here, not inside main.py.
"""

import os
import pickle
import datetime
import numpy as np
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(BASE_DIR, "models")
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

# Map encoded label indices to human-readable class names
# LabelEncoder encodes alphabetically: High=0, Low=1, Medium=2
LABEL_MAP = {0: "High", 1: "Low", 2: "Medium"}

# ─────────────────────────────────────────────────────────────────────────────
# Singleton model state (loaded once at startup)
# ─────────────────────────────────────────────────────────────────────────────
_model   = None
_scaler  = None
_encoder = None
_model_loaded = False
_model_meta: Dict = {}


def load_model() -> bool:
    """
    Load model, scaler, and label encoder from disk.
    Returns True if loaded successfully, False otherwise.
    Called once when FastAPI app starts.
    """
    global _model, _scaler, _encoder, _model_loaded, _model_meta

    try:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)

        with open(SCALER_PATH, "rb") as f:
            _scaler = pickle.load(f)

        with open(ENCODER_PATH, "rb") as f:
            _encoder = pickle.load(f)

        _model_loaded = True

        # Attempt to read stored metadata
        meta_path = os.path.join(MODELS_DIR, "model_meta.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                _model_meta = pickle.load(f)
        else:
            _model_meta = {
                "model_name":     type(_model).__name__,
                "model_version":  "1.0.0",
                "accuracy":       0.0,
                "f1_score":       0.0,
                "trained_at":     datetime.datetime.utcnow().isoformat(),
                "mlflow_run_id":  None,
            }

        print(f"✅  Model loaded: {_model_meta.get('model_name', 'Unknown')}")
        return True

    except FileNotFoundError:
        print("⚠️   Model files not found. Using RULE-BASED FALLBACK mode.")
        _model_loaded = False
        return False

    except Exception as e:
        print(f"❌  Error loading model: {e}")
        _model_loaded = False
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Rule-Based Fallback
# ─────────────────────────────────────────────────────────────────────────────

def rule_based_predict(features: dict) -> Tuple[str, float, dict]:
    """
    Simple heuristic fallback when the ML model is not available.
    Uses financial ratios to estimate risk.

    Returns:
        (risk_level, risk_score, class_probabilities)
    """
    rev  = features["monthly_revenue"]
    exp  = features["monthly_expenses"]
    res  = features["cash_reserve"]
    pi   = features["pending_invoices"]
    apd  = features["avg_payment_delay"]
    pr   = features["payroll_ratio"]
    vda  = features["vendor_due_amount"]

    score = 0  # higher = worse

    # Expense ratio > 80% is bad
    if rev > 0 and (exp / rev) > 0.80:
        score += 30
    elif rev > 0 and (exp / rev) > 0.60:
        score += 15

    # Low cash reserve relative to expenses
    if res < exp * 0.5:
        score += 25
    elif res < exp:
        score += 10

    # Long payment delays
    if apd > 30:
        score += 20
    elif apd > 15:
        score += 10

    # High pending invoices relative to revenue
    if rev > 0 and (pi / rev) > 0.5:
        score += 15
    elif rev > 0 and (pi / rev) > 0.2:
        score += 5

    # High payroll burden
    if pr > 0.6:
        score += 10

    # High vendor dues
    if rev > 0 and (vda / rev) > 0.5:
        score += 10

    score = min(score, 100)

    if score <= 30:
        level = "Low"
        probs = {"Low": 0.80, "Medium": 0.15, "High": 0.05}
    elif score <= 60:
        level = "Medium"
        probs = {"Low": 0.20, "Medium": 0.60, "High": 0.20}
    else:
        level = "High"
        probs = {"Low": 0.05, "Medium": 0.20, "High": 0.75}

    return level, float(score), probs


# ─────────────────────────────────────────────────────────────────────────────
# Insight generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_insights(features: dict) -> List[dict]:
    """
    Produce human-readable insights based on financial feature values.
    Each insight has: metric, status, message.
    """
    insights = []
    rev = features["monthly_revenue"]

    # 1. Expense ratio
    expense_ratio = features["monthly_expenses"] / rev if rev > 0 else 1.0
    if expense_ratio > 0.80:
        insights.append({
            "metric": "Expense Ratio",
            "status": "Critical",
            "message": f"Monthly expenses are {expense_ratio:.0%} of revenue — very thin margin.",
        })
    elif expense_ratio > 0.60:
        insights.append({
            "metric": "Expense Ratio",
            "status": "Warning",
            "message": f"Monthly expenses are {expense_ratio:.0%} of revenue — monitor closely.",
        })
    else:
        insights.append({
            "metric": "Expense Ratio",
            "status": "Healthy",
            "message": f"Expense ratio of {expense_ratio:.0%} is within a healthy range.",
        })

    # 2. Cash reserve runway (months)
    monthly_burn = features["monthly_expenses"]
    runway = features["cash_reserve"] / monthly_burn if monthly_burn > 0 else 99
    if runway < 1:
        insights.append({
            "metric": "Cash Runway",
            "status": "Critical",
            "message": f"Cash reserve covers less than 1 month of expenses ({runway:.1f} months).",
        })
    elif runway < 2:
        insights.append({
            "metric": "Cash Runway",
            "status": "Warning",
            "message": f"Cash runway is {runway:.1f} months — consider securing a credit line.",
        })
    else:
        insights.append({
            "metric": "Cash Runway",
            "status": "Healthy",
            "message": f"Cash runway of {runway:.1f} months provides a solid buffer.",
        })

    # 3. Payment delay
    apd = features["avg_payment_delay"]
    if apd > 30:
        insights.append({
            "metric": "Payment Delay",
            "status": "Critical",
            "message": f"Average client payment delay is {apd:.0f} days — severe receivables risk.",
        })
    elif apd > 15:
        insights.append({
            "metric": "Payment Delay",
            "status": "Warning",
            "message": f"Average payment delay of {apd:.0f} days is above the ideal 15-day threshold.",
        })
    else:
        insights.append({
            "metric": "Payment Delay",
            "status": "Healthy",
            "message": f"Payment delay of {apd:.0f} days is excellent — clients pay on time.",
        })

    # 4. Payroll ratio
    pr = features["payroll_ratio"]
    if pr > 0.60:
        insights.append({
            "metric": "Payroll Burden",
            "status": "Critical",
            "message": f"Payroll consumes {pr:.0%} of revenue — unsustainably high.",
        })
    elif pr > 0.40:
        insights.append({
            "metric": "Payroll Burden",
            "status": "Warning",
            "message": f"Payroll is {pr:.0%} of revenue — approaching a risky level.",
        })
    else:
        insights.append({
            "metric": "Payroll Burden",
            "status": "Healthy",
            "message": f"Payroll ratio of {pr:.0%} is well-managed.",
        })

    # 5. Vendor dues vs revenue
    vda_ratio = features["vendor_due_amount"] / rev if rev > 0 else 1.0
    if vda_ratio > 0.50:
        insights.append({
            "metric": "Vendor Obligations",
            "status": "Critical",
            "message": f"Vendor dues are {vda_ratio:.0%} of monthly revenue — payment pressure is high.",
        })
    elif vda_ratio > 0.20:
        insights.append({
            "metric": "Vendor Obligations",
            "status": "Warning",
            "message": f"Vendor dues represent {vda_ratio:.0%} of revenue — negotiate extended terms.",
        })
    else:
        insights.append({
            "metric": "Vendor Obligations",
            "status": "Healthy",
            "message": f"Vendor obligations are manageable at {vda_ratio:.0%} of revenue.",
        })

    return insights


def get_recommendation(risk_level: str) -> str:
    """Return a concise human-readable recommendation for each risk level."""
    recs = {
        "Low": (
            "Business is financially healthy. Continue monitoring cash flow monthly. "
            "Consider investing surplus cash reserve in short-term instruments."
        ),
        "Medium": (
            "Moderate risk detected. Prioritise collecting outstanding invoices, "
            "review vendor payment terms, and ensure a 2-month cash buffer is maintained."
        ),
        "High": (
            "High financial risk — immediate action required! "
            "Urgently reduce expenses, accelerate invoice collection, "
            "explore emergency credit facilities, and consult a financial advisor."
        ),
    }
    return recs.get(risk_level, "No recommendation available.")


# ─────────────────────────────────────────────────────────────────────────────
# Main prediction function
# ─────────────────────────────────────────────────────────────────────────────

def predict(features: dict) -> dict:
    """
    Run cash-flow risk prediction for a single business.

    Args:
        features: dict with keys matching FEATURE_COLS

    Returns:
        dict with risk_level, risk_score, confidence, class_probabilities,
        insights, recommendation
    """

    if _model_loaded and _model is not None:
        # ── ML Model path ─────────────────────────────────────────────────
        x = np.array([[features[col] for col in FEATURE_COLS]])
        x_scaled = _scaler.transform(x)

        # Predict class and probabilities
        pred_idx  = _model.predict(x_scaled)[0]
        proba     = _model.predict_proba(x_scaled)[0]  # shape: (n_classes,)

        # Map to label names
        classes   = _encoder.classes_           # e.g. ['High', 'Low', 'Medium']
        risk_level = classes[pred_idx]

        # Build class_probabilities dict
        class_probs = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}

        # Confidence = probability of predicted class
        confidence = float(proba[pred_idx])

        # Risk score: weighted sum — High=100, Medium=50, Low=0
        WEIGHT = {"High": 100, "Medium": 50, "Low": 0}
        risk_score = sum(class_probs[c] * WEIGHT[c] for c in class_probs)
        risk_score = round(risk_score, 2)

    else:
        # ── Rule-based fallback ───────────────────────────────────────────
        risk_level, risk_score, class_probs = rule_based_predict(features)
        confidence = max(class_probs.values())

    # ── Insights & recommendation ─────────────────────────────────────────
    insights       = generate_insights(features)
    recommendation = get_recommendation(risk_level)

    return {
        "risk_level":          risk_level,
        "risk_score":          risk_score,
        "confidence":          round(confidence, 4),
        "class_probabilities": class_probs,
        "insights":            insights,
        "recommendation":      recommendation,
    }


def get_model_info() -> dict:
    """Return metadata about the currently loaded model."""
    return {
        "model_name":      _model_meta.get("model_name", "Rule-Based Fallback") if _model_loaded else "Rule-Based Fallback",
        "model_version":   _model_meta.get("model_version", "N/A"),
        "accuracy":        _model_meta.get("accuracy", 0.0),
        "f1_score":        _model_meta.get("f1_score", 0.0),
        "feature_columns": FEATURE_COLS,
        "target_classes":  ["Low", "Medium", "High"],
        "trained_at":      _model_meta.get("trained_at", "Not trained yet"),
        "mlflow_run_id":   _model_meta.get("mlflow_run_id", None),
    }
