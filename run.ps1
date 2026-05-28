# run.ps1
# ─────────────────────────────────────────────────────────────────────────────
# FlowGuard AI - Windows Quick Launcher Script
#
# This script sets up the environment and runs each component.
# Usage: Right-click -> Run with PowerShell
#        OR: .\run.ps1 [command]
#
# Commands:
#   .\run.ps1 install    - Install all Python dependencies
#   .\run.ps1 data       - Generate synthetic dataset
#   .\run.ps1 train      - Train baseline model
#   .\run.ps1 automl     - Run PyCaret AutoML
#   .\run.ps1 mlflow     - Start MLflow UI (http://localhost:5000)
#   .\run.ps1 api        - Start FastAPI server (http://localhost:8000/docs)
#   .\run.ps1 bento      - Start BentoML service (http://localhost:3000)
#   .\run.ps1 test       - Run all unit tests
#   .\run.ps1 all        - Full pipeline: install + data + train + api
# ─────────────────────────────────────────────────────────────────────────────

# Force UTF-8 output for emoji/unicode in Python scripts
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$command = $args[0]

switch ($command) {
    "install" {
        Write-Host "Installing dependencies..." -ForegroundColor Cyan
        python -m pip install -r requirements.txt
    }
    "data" {
        Write-Host "Generating synthetic dataset..." -ForegroundColor Cyan
        python ml/generate_dataset.py
    }
    "train" {
        Write-Host "Training baseline model..." -ForegroundColor Cyan
        python ml/train.py
    }
    "automl" {
        Write-Host "Running PyCaret AutoML..." -ForegroundColor Cyan
        python ml/automl.py
    }
    "mlflow" {
        Write-Host "Starting MLflow UI at http://localhost:5000 ..." -ForegroundColor Green
        Write-Host "Press Ctrl+C to stop." -ForegroundColor Yellow
        mlflow ui --host 127.0.0.1 --port 5000
    }
    "api" {
        Write-Host "Starting FastAPI at http://localhost:8000/docs ..." -ForegroundColor Green
        Write-Host "Press Ctrl+C to stop." -ForegroundColor Yellow
        uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    }
    "bento" {
        Write-Host "Starting BentoML service at http://localhost:3000 ..." -ForegroundColor Green
        Write-Host "Press Ctrl+C to stop." -ForegroundColor Yellow
        bentoml serve bentoml/service.py:svc --reload --port 3000
    }
    "test" {
        Write-Host "Running unit tests..." -ForegroundColor Cyan
        pytest tests/ -v --tb=short
    }
    "all" {
        Write-Host "=== FlowGuard AI - Full Setup ===" -ForegroundColor Magenta
        Write-Host "Step 1/4: Installing dependencies..."
        python -m pip install -r requirements.txt
        Write-Host "Step 2/4: Generating dataset..."
        python ml/generate_dataset.py
        Write-Host "Step 3/4: Training model..."
        python ml/train.py
        Write-Host "Step 4/4: Starting FastAPI server..."
        Write-Host ""
        Write-Host "FastAPI running at: http://localhost:8000/docs" -ForegroundColor Green
        Write-Host "Remember to start MLflow in another terminal: .\run.ps1 mlflow" -ForegroundColor Yellow
        uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    }
    default {
        Write-Host ""
        Write-Host "FlowGuard AI - Windows Launcher" -ForegroundColor Magenta
        Write-Host "===============================" -ForegroundColor Magenta
        Write-Host ""
        Write-Host "Usage: .\run.ps1 [command]" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Commands:" -ForegroundColor Yellow
        Write-Host "  install  - Install all Python dependencies"
        Write-Host "  data     - Generate synthetic SME dataset"
        Write-Host "  train    - Train Random Forest baseline model"
        Write-Host "  automl   - Run PyCaret AutoML (picks best model)"
        Write-Host "  mlflow   - Start MLflow UI (http://localhost:5000)"
        Write-Host "  api      - Start FastAPI (http://localhost:8000/docs)"
        Write-Host "  bento    - Start BentoML service (http://localhost:3000)"
        Write-Host "  test     - Run unit tests with pytest"
        Write-Host "  all      - Full setup: install + data + train + api"
        Write-Host ""
        Write-Host "Recommended demo order:" -ForegroundColor Green
        Write-Host "  Terminal 1: .\run.ps1 mlflow"
        Write-Host "  Terminal 2: .\run.ps1 data"
        Write-Host "              .\run.ps1 train"
        Write-Host "  Terminal 3: .\run.ps1 api"
        Write-Host "  Terminal 4: .\run.ps1 bento"
    }
}
