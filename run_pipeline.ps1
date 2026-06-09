$ErrorActionPreference = 'Stop'

Write-Host "================================================" -ForegroundColor Cyan
Write-Host " Advanced Web Development Lab Setup & Execution " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

Write-Host "`n[1/4] Training the ML model (FLAML) and tracking with MLflow..." -ForegroundColor Yellow
python train.py

Write-Host "`n[2/4] Registering model to BentoML store..." -ForegroundColor Yellow
python save_bento_model.py

Write-Host "`n[3/4] Building Bento container..." -ForegroundColor Yellow
python -m bentoml build

Write-Host "`n[4/4] Starting Web Services..." -ForegroundColor Yellow
$projectPath = (Get-Location).Path

Write-Host " -> Launching Flask UI (Port 5000) in a new window" -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit -Command `"cd '$projectPath\flask_app'; python -m flask run --port=5000`""

Write-Host " -> Launching FastAPI Backend (Port 8000) in a new window" -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit -Command `"cd '$projectPath\fastapi_app'; python -m uvicorn main:app --port=8000 --reload`""

Write-Host "`nDone! The pipeline is complete. Your servers are starting up in separate windows." -ForegroundColor Cyan
Write-Host "- Flask UI: http://127.0.0.1:5000"
Write-Host "- FastAPI Swagger UI: http://127.0.0.1:8000/docs"
