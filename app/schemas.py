"""
schemas.py
----------
Pydantic data models (schemas) for FlowGuard AI FastAPI endpoints.

Pydantic validates all incoming JSON automatically.
If a field is missing or has the wrong type, FastAPI returns a 422 error
with a clear explanation — no manual validation code needed.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from enum import Enum


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    """Possible risk classification outputs."""
    LOW    = "Low"
    MEDIUM = "Medium"
    HIGH   = "High"


# ─────────────────────────────────────────────────────────────────────────────
# Request Schemas  (what clients SEND to the API)
# ─────────────────────────────────────────────────────────────────────────────

class BusinessFeatures(BaseModel):
    """
    Financial features of a single SME business.
    All monetary values are in ZAR (South African Rand) or any single currency.
    """
    monthly_revenue: float = Field(
        ...,
        gt=0,
        description="Total revenue earned in the current month (ZAR)",
        example=85_000.0,
    )
    pending_invoices: float = Field(
        ...,
        ge=0,
        description="Total value of unpaid invoices outstanding (ZAR)",
        example=22_000.0,
    )
    avg_payment_delay: float = Field(
        ...,
        ge=0,
        description="Average number of days clients delay payment",
        example=18.0,
    )
    monthly_expenses: float = Field(
        ...,
        ge=0,
        description="Total operational expenses for the month (ZAR)",
        example=54_000.0,
    )
    payroll_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Payroll cost as a fraction of monthly revenue (0.0 – 1.0)",
        example=0.38,
    )
    cash_reserve: float = Field(
        ...,
        ge=0,
        description="Total liquid cash available in business accounts (ZAR)",
        example=18_000.0,
    )
    vendor_due_amount: float = Field(
        ...,
        ge=0,
        description="Total amount owed to suppliers / vendors (ZAR)",
        example=25_000.0,
    )

    # Optional business metadata (not used for ML, but useful for reports)
    business_name: Optional[str] = Field(
        None,
        description="Business name for report labelling",
        example="Cape Town Bakery Pty Ltd",
    )
    business_id: Optional[str] = Field(
        None,
        description="Internal business identifier",
        example="BIZ-00142",
    )

    @field_validator("payroll_ratio")
    @classmethod
    def payroll_ratio_check(cls, v):
        if v > 1.0 or v < 0.0:
            raise ValueError("payroll_ratio must be between 0.0 and 1.0")
        return round(v, 4)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Low Risk Example",
                    "value": {
                        "monthly_revenue": 130_000,
                        "pending_invoices": 8_000,
                        "avg_payment_delay": 4,
                        "monthly_expenses": 60_000,
                        "payroll_ratio": 0.22,
                        "cash_reserve": 90_000,
                        "vendor_due_amount": 6_000,
                        "business_name": "Solid Enterprises Ltd",
                    },
                },
                {
                    "summary": "High Risk Example",
                    "value": {
                        "monthly_revenue": 28_000,
                        "pending_invoices": 55_000,
                        "avg_payment_delay": 45,
                        "monthly_expenses": 38_000,
                        "payroll_ratio": 0.72,
                        "cash_reserve": 3_000,
                        "vendor_due_amount": 48_000,
                        "business_name": "Struggling Co",
                    },
                },
            ]
        }
    }


class BatchPredictRequest(BaseModel):
    """Request schema for batch prediction of multiple businesses."""
    businesses: List[BusinessFeatures] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of business records to predict (max 100 per request)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Response Schemas  (what the API RETURNS to clients)
# ─────────────────────────────────────────────────────────────────────────────

class RiskInsight(BaseModel):
    """A single human-readable insight about a financial metric."""
    metric:  str
    status:  str   # e.g. "Healthy", "Warning", "Critical"
    message: str


class PredictResponse(BaseModel):
    """
    Response returned by POST /predict-risk.
    Includes risk category, numeric score, model confidence, and insights.
    """
    risk_level:       RiskLevel
    risk_score:       float = Field(description="Numeric risk score 0–100 (higher = more risky)")
    confidence:       float = Field(description="Model confidence in prediction (0–1)")
    class_probabilities: dict = Field(description="Probability per risk class")
    insights:         List[RiskInsight]
    recommendation:   str
    business_name:    Optional[str] = None
    business_id:      Optional[str] = None

    model_config = {"use_enum_values": True}


class BatchPredictItem(BaseModel):
    """Single result within a batch prediction response."""
    index:        int
    business_name: Optional[str] = None
    risk_level:   RiskLevel
    risk_score:   float
    confidence:   float

    model_config = {"use_enum_values": True}


class BatchPredictResponse(BaseModel):
    """Response for POST /batch-predict."""
    total:   int
    results: List[BatchPredictItem]
    summary: dict   # e.g. {"Low": 5, "Medium": 3, "High": 2}


class ModelInfoResponse(BaseModel):
    """Response for GET /model-info."""
    model_name:     str
    model_version:  str
    accuracy:       float
    f1_score:       float
    feature_columns: List[str]
    target_classes:  List[str]
    trained_at:     str
    mlflow_run_id:  Optional[str] = None

    model_config = {"protected_namespaces": ()}


class HealthResponse(BaseModel):
    """Response for GET /."""
    status:        str
    service:       str
    version:       str
    model_loaded:  bool

    model_config = {"protected_namespaces": ()}
