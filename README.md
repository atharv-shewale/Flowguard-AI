# FlowGuard AI – SME Cash Flow Risk Prediction Platform

> **A machine learning system that predicts cash-flow risk for Small and Medium Enterprises.**  
> College MVP | FastAPI · Scikit-learn · PyCaret AutoML · MLflow · BentoML · Jenkins

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Project Structure](#-project-structure)
3. [Tech Stack](#-tech-stack)
4. [Quick Start (5 Steps)](#-quick-start)
5. [Step-by-Step Guide](#-step-by-step-guide)
   - [Step 1 – Install Dependencies](#step-1--install-dependencies)
   - [Step 2 – Generate Dataset](#step-2--generate-dataset)
   - [Step 3 – Train the Model](#step-3--train-the-model-baseline)
   - [Step 4 – Run AutoML](#step-4--run-automl-pycaret)
   - [Step 5 – Start MLflow UI](#step-5--start-mlflow-ui)
   - [Step 6 – Run FastAPI](#step-6--run-fastapi-server)
   - [Step 7 – Run BentoML](#step-7--run-bentoml-service)
6. [API Reference](#-api-reference)
7. [Sample JSON Inputs](#-sample-json-inputs)
8. [MLflow Guide](#-mlflow-guide)
9. [BentoML Guide](#-bentoml-guide)
10. [Jenkins CI/CD Design](#-jenkins-cicd-design)
11. [Risk Score Explained](#-risk-score-explained)
12. [Running Tests](#-running-tests)
13. [Architecture Diagram](#-architecture-diagram)

---

## 🎯 Project Overview

FlowGuard AI solves a critical problem: **SMEs often don't know they're heading into a cash-flow crisis until it's too late.**

This system takes 7 key financial metrics and predicts the risk level:

| Input Feature        | Description                                  |
|----------------------|----------------------------------------------|
| `monthly_revenue`    | Total revenue earned this month (ZAR)        |
| `pending_invoices`   | Unpaid invoice value outstanding (ZAR)       |
| `avg_payment_delay`  | Average days clients delay payment           |
| `monthly_expenses`   | Total operating expenses (ZAR)               |
| `payroll_ratio`      | Payroll as fraction of revenue (0.0–1.0)     |
| `cash_reserve`       | Liquid cash in business accounts (ZAR)       |
| `vendor_due_amount`  | Amount owed to suppliers (ZAR)               |

**Output:**
- ✅ Risk Level: **Low / Medium / High**
- 📊 Risk Score: **0–100** (higher = more risk)
- 🎯 Confidence: Model certainty (0–1)
- 💡 Insights: 5 actionable financial health observations
- 📝 Recommendation: What the business should do

---

## 📁 Project Structure

```
flowguard-ai/
│
├── app/                        # FastAPI Application
│   ├── __init__.py
│   ├── main.py                 # API routes, middleware, startup
│   ├── schemas.py              # Pydantic request/response models
│   └── model.py                # ML inference + rule-based fallback
│
├── ml/                         # Machine Learning
│   ├── generate_dataset.py     # Synthetic dataset generator
│   ├── train.py                # Baseline Random Forest + MLflow
│   ├── automl.py               # PyCaret AutoML pipeline
│   └── dataset.csv             # Generated after running generate_dataset.py
│
├── bentoml/                    # BentoML Deployment
│   ├── service.py              # BentoML service definition
│   └── bentofile.yaml          # Build + Docker configuration
│
├── jenkins/
│   └── Jenkinsfile             # CI/CD pipeline (6 stages)
│
├── models/                     # Saved model artifacts (created after training)
│   ├── saved_model.pkl         # Trained scikit-learn model
│   ├── scaler.pkl              # Feature scaler
│   └── label_encoder.pkl       # Label encoder
│
├── tests/
│   ├── __init__.py
│   └── test_api.py             # pytest unit tests
│
├── requirements.txt
└── README.md
```

---

## 🛠 Tech Stack

| Component    | Technology          | Purpose                              |
|--------------|---------------------|--------------------------------------|
| API          | **FastAPI**         | REST API with Swagger UI             |
| Validation   | **Pydantic v2**     | Input/output data validation         |
| ML Baseline  | **Scikit-learn**    | Random Forest classifier             |
| AutoML       | **PyCaret**         | Compare 15+ models, select best      |
| MLOps        | **MLflow**          | Experiment tracking, metrics logging |
| Serving      | **BentoML**         | Model packaging and REST serving     |
| CI/CD        | **Jenkins**         | Automated build/test/deploy pipeline |
| Testing      | **pytest**          | Unit and integration tests           |

---

## ⚡ Quick Start

```bash
# 1. Clone and enter directory
cd flowguard-ai

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Generate dataset + train model
python ml/generate_dataset.py
python ml/train.py

# 4. Start MLflow UI (new terminal)
mlflow ui --port 5000

# 5. Start FastAPI (new terminal)
uvicorn app.main:app --reload --port 8000
# Open: http://localhost:8000/docs
```

---

## 📖 Step-by-Step Guide

### Step 1 – Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install all packages
pip install -r requirements.txt
```

> ⏱ This may take 3–5 minutes as PyCaret installs many ML libraries.

---

### Step 2 – Generate Dataset

The synthetic dataset simulates 1,500 SME financial records (500 per risk class).

```bash
python ml/generate_dataset.py
```

**Expected output:**
```
✅  Dataset saved → ml/dataset.csv  (1500 rows)
Low      500
Medium   500
High     500
```

---

### Step 3 – Train the Model (Baseline)

> ⚠️ Start your MLflow server first (Step 5) so metrics are logged properly.  
> If MLflow is not running, the script still trains and saves the model locally.

```bash
python ml/train.py
```

**What happens:**
1. Loads `ml/dataset.csv`
2. Splits 80% train / 20% test
3. Scales features with `StandardScaler`
4. Trains a `RandomForestClassifier` with 200 trees
5. Logs accuracy, precision, recall, F1 to **MLflow**
6. Saves model artifacts to `models/`

**Expected output:**
```
=======================================================
  FlowGuard AI – Baseline Training Results
=======================================================
  Accuracy : 0.9533
  Precision: 0.9536
  Recall   : 0.9533
  F1 Score : 0.9534

              precision    recall  f1-score
  High           0.96      0.95      0.96
  Low            0.97      0.97      0.97
  Medium         0.93      0.94      0.94
```

---

### Step 4 – Run AutoML (PyCaret)

```bash
python ml/automl.py
```

**What happens:**
1. Initialises PyCaret environment (handles scaling, CV, encoding)
2. Compares 15+ algorithms (Random Forest, XGBoost, LightGBM, SVM, etc.)
3. Selects the best model by Accuracy
4. Tunes hyperparameters with random search (20 iterations)
5. Finalises model on 100% of data
6. Saves best model to `models/pycaret_best_model.pkl`
7. Logs everything to **MLflow** experiment `FlowGuard-AI-AutoML`

> ⏱ AutoML takes 3–10 minutes depending on your hardware.

---

### Step 5 – Start MLflow UI

**Open a separate terminal** and run:

```bash
# From inside the flowguard-ai directory
mlflow ui --host 127.0.0.1 --port 5000
```

Then open your browser at: **http://127.0.0.1:5000**

You will see two experiments:
- `FlowGuard-AI-Baseline` – from `train.py`
- `FlowGuard-AI-AutoML`   – from `automl.py`

Each experiment shows: accuracy, precision, recall, F1, hyperparameters, and model artifacts.

---

### Step 6 – Run FastAPI Server

**Open another terminal:**

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

| URL                             | What it is              |
|---------------------------------|-------------------------|
| http://localhost:8000/docs      | **Swagger UI** (interactive) |
| http://localhost:8000/redoc     | ReDoc documentation     |
| http://localhost:8000/          | Health check JSON        |
| http://localhost:8000/model-info| Model metadata           |

---

### Step 7 – Run BentoML Service

> Run this **after** training the model (Step 3).

```bash
# Register model in BentoML store and start service
cd flowguard-ai
bentoml serve bentoml/service.py:svc --reload --port 3000
```

Test the BentoML endpoint:
```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "monthly_revenue": 85000,
    "pending_invoices": 22000,
    "avg_payment_delay": 18,
    "monthly_expenses": 54000,
    "payroll_ratio": 0.38,
    "cash_reserve": 18000,
    "vendor_due_amount": 25000,
    "business_name": "Demo Corp"
  }'
```

---

## 📡 API Reference

### `GET /`
**Health check**

Response:
```json
{
  "status": "healthy",
  "service": "FlowGuard AI – SME Cash Flow Risk Prediction",
  "version": "1.0.0",
  "model_loaded": true
}
```

---

### `POST /predict-risk`
**Predict cash-flow risk for a single business**

Request body: `BusinessFeatures` (see schemas.py)

Response:
```json
{
  "risk_level": "Medium",
  "risk_score": 52.3,
  "confidence": 0.71,
  "class_probabilities": {
    "High": 0.15,
    "Low": 0.14,
    "Medium": 0.71
  },
  "insights": [
    {"metric": "Expense Ratio", "status": "Warning", "message": "..."},
    {"metric": "Cash Runway", "status": "Warning", "message": "..."}
  ],
  "recommendation": "Moderate risk detected. Prioritise collecting outstanding invoices...",
  "business_name": "Cape Town Bakery Pty Ltd"
}
```

---

### `POST /batch-predict`
**Predict for up to 100 businesses at once**

```json
{
  "businesses": [
    { ...business1 },
    { ...business2 }
  ]
}
```

Response:
```json
{
  "total": 2,
  "results": [
    {"index": 0, "business_name": "...", "risk_level": "Low", "risk_score": 12.0, "confidence": 0.91},
    {"index": 1, "business_name": "...", "risk_level": "High", "risk_score": 88.5, "confidence": 0.87}
  ],
  "summary": {"Low": 1, "Medium": 0, "High": 1}
}
```

---

### `GET /model-info`
**Active model metadata**

```json
{
  "model_name": "RandomForestClassifier",
  "model_version": "1.0.0",
  "accuracy": 0.9533,
  "f1_score": 0.9534,
  "feature_columns": ["monthly_revenue", "pending_invoices", ...],
  "target_classes": ["Low", "Medium", "High"],
  "trained_at": "2024-01-15T10:30:00",
  "mlflow_run_id": "abc123def456"
}
```

---

## 🧪 Sample JSON Inputs

Copy these into Swagger UI's `POST /predict-risk` → **Try it out**:

### ✅ Low Risk Business
```json
{
  "monthly_revenue": 130000,
  "pending_invoices": 8000,
  "avg_payment_delay": 4,
  "monthly_expenses": 60000,
  "payroll_ratio": 0.22,
  "cash_reserve": 90000,
  "vendor_due_amount": 6000,
  "business_name": "Solid Enterprises Ltd"
}
```

### ⚠️ Medium Risk Business
```json
{
  "monthly_revenue": 75000,
  "pending_invoices": 24000,
  "avg_payment_delay": 20,
  "monthly_expenses": 52000,
  "payroll_ratio": 0.40,
  "cash_reserve": 22000,
  "vendor_due_amount": 18000,
  "business_name": "Growing Pains Co"
}
```

### 🚨 High Risk Business
```json
{
  "monthly_revenue": 28000,
  "pending_invoices": 55000,
  "avg_payment_delay": 45,
  "monthly_expenses": 38000,
  "payroll_ratio": 0.72,
  "cash_reserve": 3000,
  "vendor_due_amount": 48000,
  "business_name": "Struggling Co"
}
```

---

## 📈 MLflow Guide

### What is MLflow?
MLflow tracks ML experiments so you can compare runs, view metrics, and download model artifacts.

### Viewing Experiments

1. Start: `mlflow ui --port 5000`
2. Open: http://127.0.0.1:5000
3. Click **FlowGuard-AI-Baseline** or **FlowGuard-AI-AutoML**
4. Click any run to see metrics, parameters, and artifacts

### Logged Information

| Experiment            | Logged Metrics                     |
|-----------------------|------------------------------------|
| Baseline (train.py)   | accuracy, precision, recall, f1    |
| AutoML (automl.py)    | cv_accuracy, cv_precision, cv_f1   |

Both experiments log: hyperparameters, feature columns, model artifacts.

---

## 🚀 BentoML Guide

### What is BentoML?
BentoML packages your ML model into a production-ready REST service, similar to deploying a mini Flask/FastAPI app but purpose-built for ML.

### Commands

```bash
# List all registered models
bentoml models list

# List all built Bentos
bentoml list

# Build a Bento package
bentoml build -f bentoml/bentofile.yaml

# Serve the latest Bento
bentoml serve flowguard_ai_service:latest --port 3000

# (Optional) Containerize with Docker
bentoml containerize flowguard_ai_service:latest
docker run -p 3000:3000 flowguard_ai_service:latest
```

---

## 🔧 Jenkins CI/CD Design

The `jenkins/Jenkinsfile` defines a 6-stage pipeline:

```
[1] Checkout → [2] Setup Env → [3] Tests → [4] Generate Data
       ↓
[5] Train Model (Baseline ∥ AutoML in parallel) → [6] Deploy (main branch only)
```

| Stage         | What it does                                      |
|---------------|---------------------------------------------------|
| Checkout      | Pulls code from Git                               |
| Setup Env     | Creates venv, installs requirements               |
| Test          | Runs pytest, publishes JUnit XML report           |
| Generate Data | Runs `generate_dataset.py`                        |
| Train         | Runs `train.py` + `automl.py` **in parallel**     |
| Deploy        | Builds Bento, starts BentoML service              |

> This file shows a real-world CI/CD pattern without requiring an actual Jenkins server for the demo.

---

## 📊 Risk Score Explained

The **risk score** (0–100) is a weighted sum of class probabilities:

```
risk_score = P(High) × 100 + P(Medium) × 50 + P(Low) × 0
```

| Score Range | Interpretation |
|-------------|----------------|
| 0 – 30      | Low Risk       |
| 31 – 65     | Medium Risk    |
| 66 – 100    | High Risk      |

The **rule-based fallback** (used if the model isn't trained) calculates risk from financial ratios:
- Expense ratio > 80% → +30 points
- Cash runway < 1 month → +25 points
- Payment delay > 30 days → +20 points
- Pending invoices > 50% of revenue → +15 points

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=app --cov-report=term-missing

# Run a specific test class
pytest tests/test_api.py::TestPredictRisk -v
```

Expected output:
```
tests/test_api.py::TestHealthCheck::test_health_returns_200 PASSED
tests/test_api.py::TestPredictRisk::test_predict_returns_200 PASSED
tests/test_api.py::TestPredictRisk::test_risk_level_is_valid PASSED
...
25 passed in 2.31s
```

---

## 🏗 Architecture Diagram

```
                     ┌─────────────────────────────────────────┐
                     │              DATA LAYER                  │
                     │   ml/generate_dataset.py → dataset.csv  │
                     └─────────────────┬───────────────────────┘
                                       │
                     ┌─────────────────▼───────────────────────┐
                     │              ML LAYER                    │
                     │  ┌──────────────┐  ┌──────────────────┐ │
                     │  │  train.py    │  │   automl.py      │ │
                     │  │ (Scikit-learn│  │   (PyCaret)      │ │
                     │  │  Random      │  │  15+ algorithms  │ │
                     │  │  Forest)     │  │  auto-select     │ │
                     │  └──────┬───────┘  └────────┬─────────┘ │
                     └─────────┼──────────────────-┼───────────┘
                               │   MLflow Logging  │
                     ┌─────────▼───────────────────▼───────────┐
                     │            MLOPS LAYER                   │
                     │         MLflow Tracking Server           │
                     │      http://localhost:5000               │
                     └─────────────────┬───────────────────────┘
                                       │  Saved model (.pkl)
                     ┌─────────────────▼───────────────────────┐
                     │           SERVING LAYER                  │
                     │  ┌─────────────────┐  ┌──────────────┐  │
                     │  │   FastAPI       │  │   BentoML    │  │
                     │  │ :8000/docs      │  │  :3000       │  │
                     │  │ /predict-risk   │  │  /predict    │  │
                     │  │ /batch-predict  │  │              │  │
                     │  │ /model-info     │  │              │  │
                     │  └─────────────────┘  └──────────────┘  │
                     └─────────────────────────────────────────┘
                                       │
                     ┌─────────────────▼───────────────────────┐
                     │           CI/CD LAYER                    │
                     │     Jenkins Pipeline (Jenkinsfile)       │
                     │  Checkout → Setup → Test → Train →      │
                     │  Deploy (automated on git push)          │
                     └─────────────────────────────────────────┘
```

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'app'` | Run uvicorn from inside the `flowguard-ai/` directory |
| `FileNotFoundError: saved_model.pkl` | Run `python ml/train.py` first |
| MLflow logs not appearing | Start MLflow server first: `mlflow ui --port 5000` |
| PyCaret installation error | Try `pip install pycaret` without `[full]` first |
| BentoML model not found | Run train.py first, then start service.py |
| Port 8000 in use | Kill existing process: `taskkill /f /im python.exe` (Windows) |

---

## 👥 Team

FlowGuard AI MVP — Built for college POE demonstration.

**Components demonstrated:**
- ✅ FastAPI REST API with Swagger UI
- ✅ Pydantic data validation
- ✅ Scikit-learn baseline model
- ✅ PyCaret AutoML
- ✅ MLflow experiment tracking
- ✅ BentoML model serving
- ✅ Jenkins CI/CD pipeline design
- ✅ Rule-based fallback system
- ✅ Feature-based insight generation
- ✅ pytest unit tests

---

*FlowGuard AI — Predict the risk before it becomes a crisis.*
