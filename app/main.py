"""
main.py
-------
FlowGuard AI – FastAPI Application Entry Point

This is the main REST API server. It exposes:
  GET  /                → Health check
  POST /predict-risk    → Single business risk prediction
  POST /batch-predict   → Batch prediction for multiple businesses
  GET  /model-info      → Model metadata and accuracy

Swagger UI is automatically available at http://127.0.0.1:8000/docs
ReDoc is available at http://127.0.0.1:8000/redoc

Run with:
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import time
import logging
import os

from app.schemas import (
    BusinessFeatures,
    BatchPredictRequest,
    PredictResponse,
    BatchPredictResponse,
    BatchPredictItem,
    ModelInfoResponse,
    HealthResponse,
    RiskInsight,
)
from app import model as ml_model

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("flowguard")


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan event handler (model loading on startup)
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model on startup; clean up on shutdown."""
    logger.info("FlowGuard AI starting up...")
    loaded = ml_model.load_model()
    if loaded:
        logger.info("ML model loaded successfully")
    else:
        logger.warning("Running in FALLBACK (rule-based) mode -- train the model first")
    yield
    logger.info("FlowGuard AI shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI application instance
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FlowGuard AI - SME Cash Flow Risk Prediction API",
    description="""
## FlowGuard AI

A machine learning-powered platform that predicts cash-flow risk for Small and Medium Enterprises (SMEs).

### 🖥️ Interactive Web UI Dashboard
👉 **[Open FlowGuard AI Web UI Dashboard](/ui)** (Runs a complete visual interface with Form Presets, Single Risk Predictor, Portfolio Analyzer, and real-time model diagnostics)

### Risk Categories
| Level  | Meaning                                     |
|--------|---------------------------------------------|
| Low    | Business is financially healthy             |
| Medium | Some risk factors present — monitor closely |
| High   | Urgent action required to avoid cash crisis |

### How to Use
1. Call `POST /predict-risk` with your business's financial data
2. Receive a risk classification, score (0–100), confidence, and actionable insights
3. Use `GET /model-info` to verify the active model version

### Sample Inputs
Try the pre-filled examples in the Swagger UI below.
""",
    version="1.0.0",
    lifespan=lifespan,
    contact={
        "name": "FlowGuard AI Team",
        "email": "support@flowguard.ai",
    },
    license_info={
        "name": "MIT",
    },
    docs_url="/docs",        # Swagger UI
    redoc_url="/redoc",      # ReDoc
    openapi_url="/openapi.json",
)

# ─────────────────────────────────────────────────────────────────────────────
# CORS Middleware (allow all origins for local demo)
# ─────────────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Request timing middleware
# ─────────────────────────────────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(duration)
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({duration}ms)")
    return response


# (startup is handled by lifespan above)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

# ── GET /ui ───────────────────────────────────────────────────────────────────
@app.get(
    "/ui",
    response_class=HTMLResponse,
    summary="Interactive Web UI Dashboard",
    tags=["Interface"],
)
async def serve_ui():
    """
    Serves the beautiful interactive Single Page Application (SPA) dashboard.
    Enables single SME cash-flow predictions, batch portfolio analyzer, and real-time model diagnostics.
    """
    ui_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "templates", "dashboard.html"
    )
    if not os.path.exists(ui_path):
        raise HTTPException(status_code=404, detail="Dashboard UI template file not found")
    with open(ui_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


# ── GET / ─────────────────────────────────────────────────────────────────────
@app.api_route(
    "/",
    methods=["GET", "HEAD"],
    response_model=HealthResponse,
    summary="Health Check",
    tags=["System"],
)
async def health_check():
    """
    Returns the service health status and whether the ML model is loaded.
    Use this to verify the API is running before making predictions.
    """
    return HealthResponse(
        status="healthy",
        service="FlowGuard AI – SME Cash Flow Risk Prediction",
        version="1.0.0",
        model_loaded=ml_model._model_loaded,
    )


# ── POST /predict-risk ────────────────────────────────────────────────────────
@app.post(
    "/predict-risk",
    response_model=PredictResponse,
    summary="Predict Cash Flow Risk",
    tags=["Prediction"],
    responses={
        200: {"description": "Successful prediction"},
        422: {"description": "Validation error – check input field types and ranges"},
        500: {"description": "Internal server error"},
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "Low Risk – Solid Enterprises": {
                            "summary": "Low Risk – financially healthy business",
                            "value": {
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
                        },
                        "Medium Risk – Cape Town Bakery": {
                            "summary": "Medium Risk – some concerns, monitoring needed",
                            "value": {
                                "monthly_revenue": 85000,
                                "pending_invoices": 22000,
                                "avg_payment_delay": 18,
                                "monthly_expenses": 54000,
                                "payroll_ratio": 0.38,
                                "cash_reserve": 18000,
                                "vendor_due_amount": 25000,
                                "business_name": "Cape Town Bakery Pty Ltd",
                                "business_id": "BIZ-002",
                            },
                        },
                        "High Risk – Struggling Co": {
                            "summary": "High Risk – urgent action required",
                            "value": {
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
                        },
                    }
                }
            }
        }
    },
)
async def predict_risk(business: BusinessFeatures):

    """
    Predict the cash-flow risk level for a single SME business.

    ### Input
    Provide the 7 key financial metrics for the business.

    ### Output
    - **risk_level**: Low | Medium | High
    - **risk_score**: 0–100 (higher = more risk)
    - **confidence**: model confidence in this prediction (0–1)
    - **class_probabilities**: probability for each risk class
    - **insights**: list of financial health observations
    - **recommendation**: suggested action based on risk level

    ### Sample Inputs
    Use the example values pre-filled in the Swagger UI.
    """
    try:
        # Convert Pydantic model to plain dict for the inference engine
        features = {
            "monthly_revenue":   business.monthly_revenue,
            "pending_invoices":  business.pending_invoices,
            "avg_payment_delay": business.avg_payment_delay,
            "monthly_expenses":  business.monthly_expenses,
            "payroll_ratio":     business.payroll_ratio,
            "cash_reserve":      business.cash_reserve,
            "vendor_due_amount": business.vendor_due_amount,
        }

        result = ml_model.predict(features)

        # Convert raw insight dicts → Pydantic RiskInsight objects
        insights = [RiskInsight(**i) for i in result["insights"]]

        return PredictResponse(
            risk_level           = result["risk_level"],
            risk_score           = result["risk_score"],
            confidence           = result["confidence"],
            class_probabilities  = result["class_probabilities"],
            insights             = insights,
            recommendation       = result["recommendation"],
            business_name        = business.business_name,
            business_id          = business.business_id,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ── POST /batch-predict ────────────────────────────────────────────────────────
@app.post(
    "/batch-predict",
    response_model=BatchPredictResponse,
    summary="Batch Predict – Multiple Businesses",
    tags=["Prediction"],
)
async def batch_predict(request: BatchPredictRequest):
    """
    Predict cash-flow risk for a batch of up to **100 businesses** in a single request.

    Returns a summary count of risk levels alongside individual results.
    Useful for portfolio risk analysis or bulk report generation.
    """
    results  = []
    summary  = {"Low": 0, "Medium": 0, "High": 0}

    for idx, business in enumerate(request.businesses):
        try:
            features = {
                "monthly_revenue":   business.monthly_revenue,
                "pending_invoices":  business.pending_invoices,
                "avg_payment_delay": business.avg_payment_delay,
                "monthly_expenses":  business.monthly_expenses,
                "payroll_ratio":     business.payroll_ratio,
                "cash_reserve":      business.cash_reserve,
                "vendor_due_amount": business.vendor_due_amount,
            }
            result = ml_model.predict(features)

            item = BatchPredictItem(
                index         = idx,
                business_name = business.business_name,
                risk_level    = result["risk_level"],
                risk_score    = result["risk_score"],
                confidence    = result["confidence"],
            )
            results.append(item)
            summary[result["risk_level"]] = summary.get(result["risk_level"], 0) + 1

        except Exception as e:
            logger.error(f"Batch item {idx} failed: {e}")
            # Skip failed items and continue

    return BatchPredictResponse(
        total   = len(results),
        results = results,
        summary = summary,
    )


# ── GET /model-info ────────────────────────────────────────────────────────────
@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model Metadata",
    tags=["System"],
)
async def model_info():
    """
    Returns metadata about the currently active ML model:
    - Model name and version
    - Accuracy and F1 score on the test set
    - Feature column names
    - MLflow run ID (if available)
    """
    info = ml_model.get_model_info()
    return ModelInfoResponse(**info)


# ─────────────────────────────────────────────────────────────────────────────
# Custom error handlers
# ─────────────────────────────────────────────────────────────────────────────
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "available_docs": "/docs"},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """
    Custom handler for 422 Validation Errors.
    Returns a clean, human-readable error message listing the missing/invalid fields.
    """
    errors = []
    for err in exc.errors():
        field = " -> ".join(str(loc) for loc in err["loc"])
        errors.append({
            "field": field,
            "issue": err["msg"],
            "type":  err["type"],
        })
    return JSONResponse(
        status_code=422,
        content={
            "error": "Request validation failed. Check all required fields are present and correctly typed.",
            "hint": "Use the 'Try it out' button in Swagger UI (/docs) and select an example from the dropdown.",
            "fields": errors,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run directly (development mode)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
