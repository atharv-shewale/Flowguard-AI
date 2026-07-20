"""
schemas.py
----------
Pydantic v2 data models (schemas) for FlowGuard AI FastAPI endpoints.

Pydantic v2 NOTE:
  - Field(example=...) is DEPRECATED and silently ignored in JSON schema output.
  - Use Field(..., json_schema_extra={"example": value}) instead.
  - model_config["json_schema_extra"] must use "example" (singular) for Swagger UI
    to pre-fill the request body; the "examples" list format is not rendered by Swagger.
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

    ### Example (Medium Risk – Cape Town Bakery):
    ```json
    {
      "monthly_revenue": 85000,
      "pending_invoices": 22000,
      "avg_payment_delay": 18,
      "monthly_expenses": 54000,
      "payroll_ratio": 0.38,
      "cash_reserve": 18000,
      "vendor_due_amount": 25000,
      "business_name": "Cape Town Bakery Pty Ltd",
      "business_id": "BIZ-002"
    }
    ```
    """

    monthly_revenue: float = Field(
        ...,
        gt=0,
        description="Total revenue earned in the current month (ZAR). Must be > 0.",
        json_schema_extra={"example": 85000.0},
    )
    pending_invoices: float = Field(
        ...,
        ge=0,
        description="Total value of unpaid invoices outstanding (ZAR). Use 0 if none.",
        json_schema_extra={"example": 22000.0},
    )
    avg_payment_delay: float = Field(
        ...,
        ge=0,
        description="Average number of days clients delay payment. Use 0 if clients pay on time.",
        json_schema_extra={"example": 18.0},
    )
    monthly_expenses: float = Field(
        ...,
        ge=0,
        description="Total operational expenses for the month (ZAR).",
        json_schema_extra={"example": 54000.0},
    )
    payroll_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Payroll cost as a fraction of monthly revenue (0.0 to 1.0). E.g. 0.38 = 38%.",
        json_schema_extra={"example": 0.38},
    )
    cash_reserve: float = Field(
        ...,
        ge=0,
        description="Total liquid cash available in business accounts (ZAR).",
        json_schema_extra={"example": 18000.0},
    )
    vendor_due_amount: float = Field(
        ...,
        ge=0,
        description="Total amount owed to suppliers / vendors (ZAR).",
        json_schema_extra={"example": 25000.0},
    )

    # Optional business metadata (not used for ML, but useful for labelling reports)
    business_name: Optional[str] = Field(
        default=None,
        description="Business name for report labelling (optional).",
        json_schema_extra={"example": "Cape Town Bakery Pty Ltd"},
    )
    business_id: Optional[str] = Field(
        default=None,
        description="Internal business identifier (optional).",
        json_schema_extra={"example": "BIZ-002"},
    )

    @field_validator("payroll_ratio")
    @classmethod
    def payroll_ratio_check(cls, v):
        if v > 1.0 or v < 0.0:
            raise ValueError("payroll_ratio must be between 0.0 and 1.0")
        return round(v, 4)

    model_config = {
        # This single "example" key is what Swagger UI uses to pre-fill the body.
        # Must be singular "example", NOT a list under "examples".
        "json_schema_extra": {
            "example": {
                "monthly_revenue": 85000,
                "pending_invoices": 22000,
                "avg_payment_delay": 18,
                "monthly_expenses": 54000,
                "payroll_ratio": 0.38,
                "cash_reserve": 18000,
                "vendor_due_amount": 25000,
                "business_name": "Cape Town Bakery Pty Ltd",
                "business_id": "BIZ-002",
            }
        }
    }


class BatchPredictRequest(BaseModel):
    """Request schema for batch prediction of multiple businesses."""
    businesses: List[BusinessFeatures] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of business records to predict (1–100 per request).",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "businesses": [
                    {
                        "monthly_revenue": 130000,
                        "pending_invoices": 8000,
                        "avg_payment_delay": 4,
                        "monthly_expenses": 60000,
                        "payroll_ratio": 0.22,
                        "cash_reserve": 90000,
                        "vendor_due_amount": 6000,
                        "business_name": "Solid Enterprises Ltd",
                        "business_id": "BIZ-001",
                    },
                    {
                        "monthly_revenue": 28000,
                        "pending_invoices": 55000,
                        "avg_payment_delay": 45,
                        "monthly_expenses": 38000,
                        "payroll_ratio": 0.72,
                        "cash_reserve": 3000,
                        "vendor_due_amount": 48000,
                        "business_name": "Struggling Co",
                        "business_id": "BIZ-003",
                    },
                ]
            }
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Response Schemas  (what the API RETURNS to clients)
# ─────────────────────────────────────────────────────────────────────────────

class RiskInsight(BaseModel):
    """A single human-readable insight about a financial metric."""
    metric:  str
    status:  str   # "Healthy" | "Warning" | "Critical"
    message: str


class PredictResponse(BaseModel):
    """
    Response returned by POST /predict-risk.
    Includes risk category, numeric score, model confidence, and insights.
    """
    risk_level:          RiskLevel
    risk_score:          float = Field(description="Numeric risk score 0–100 (higher = more risky)")
    confidence:          float = Field(description="Model confidence in prediction (0.0–1.0)")
    class_probabilities: dict  = Field(description="Probability per risk class: {Low, Medium, High}")
    insights:            List[RiskInsight]
    recommendation:      str
    business_name:       Optional[str] = None
    business_id:         Optional[str] = None

    model_config = {"use_enum_values": True}


class BatchPredictItem(BaseModel):
    """Single result within a batch prediction response."""
    index:         int
    business_name: Optional[str] = None
    risk_level:    RiskLevel
    risk_score:    float
    confidence:    float

    model_config = {"use_enum_values": True}


class BatchPredictResponse(BaseModel):
    """Response for POST /batch-predict."""
    total:   int
    results: List[BatchPredictItem]
    summary: dict   # e.g. {"Low": 5, "Medium": 3, "High": 2}


class ModelInfoResponse(BaseModel):
    """Response for GET /model-info — includes full ML transparency data."""
    model_name:          str
    model_version:       str
    accuracy:            float
    f1_score:            float
    feature_columns:     List[str]
    target_classes:      List[str]
    trained_at:          str
    mlflow_run_id:       Optional[str] = None
    # Transparency fields
    feature_importances: Optional[dict] = None
    model_hyperparams:   Optional[dict] = None
    automl_comparison:   Optional[List[dict]] = None
    model_loaded:        Optional[bool] = None
    training_pipeline:   Optional[List[dict]] = None

    model_config = {"protected_namespaces": ()}


class HealthResponse(BaseModel):
    """Response for GET /."""
    status:       str
    service:      str
    version:      str
    model_loaded: bool

    model_config = {"protected_namespaces": ()}
