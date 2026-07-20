# Project Setup Guide

This guide will walk you through setting up and running the Advanced Web Development Lab Machine Learning application on a new PC.

## Prerequisites
Ensure the new PC has the following installed:
1. **Python 3.9-3.11**: Make sure Python is added to the system PATH.
2. **Git**: To clone the repository.
3. **MySQL Database**: E.g., through XAMPP or a standalone MySQL Server installation.

## 1. Clone the Repository
Open your terminal (or PowerShell) and clone the project:
```bash
git clone https://github.com/atharv-shewale/External-AWDL.git
cd External-AWDL
```

## 2. Database Setup
Ensure your MySQL server is running. Create the database and required tables using the provided `schema.sql` file:
```bash
mysql -u root -p < schema.sql
```
> [!NOTE]
> By default, the app expects MySQL to be running on `localhost` with user `root` and an empty password. If your MySQL setup differs, you will need to set the following Environment Variables before running the applications:
> - `DB_HOST` (default: `localhost`)
> - `DB_USER` (default: `root`)
> - `DB_PASSWORD` (default: empty string)
> - `DB_NAME` (default: `diabetes_db`)

## 3. Python Environment Setup
It is highly recommended to use a virtual environment to avoid conflicts with other python packages:
```bash
# Create the virtual environment
python -m venv venv

# Activate the virtual environment (Windows)
venv\Scripts\activate

# If using Mac/Linux, activate via: source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

## 4. Run the Pipeline
If you are on Windows, you can easily run the entire pipeline (training, BentoML packaging, and starting web servers) using the provided PowerShell script:

```bash
.\run_pipeline.ps1
```

> [!TIP]
> If you face execution policy issues running the `.ps1` script, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` beforehand.

### Manual Step-by-Step Execution
If you prefer running the steps manually (or are on Linux/Mac), execute the following commands in order:

1. **Train the ML Model:**
   ```bash
   python train.py
   ```
2. **Register Model with BentoML:**
   ```bash
   python save_bento_model.py
   ```
3. **Build the BentoML Container:**
   ```bash
   bentoml build
   ```
4. **Start the Flask UI:**
   ```bash
   cd flask_app
   python -m flask run --port=5000
   ```
5. **Start the FastAPI Backend (in a new terminal):**
   ```bash
   # Don't forget to activate your venv in the new terminal!
   cd fastapi_app
   python -m uvicorn main:app --port=8000 --reload
   ```

## 5. Usage
- Access the **Flask UI** at: `http://127.0.0.1:5000`
- Access the **FastAPI Swagger UI** at: `http://127.0.0.1:8000/docs`

## 6. Deployment Checklist

1. Copy env template and set DB/app ports:
   ```bash
   cp .env.example .env
   ```
2. Build and package model artifacts:
   ```bash
   PYTHON_BIN=python3.11 bash scripts/deploy_build.sh
   ```
3. Deploy/run services on your target host:
   - Flask UI: `python -m flask run --port=${FLASK_PORT:-5000}`
   - FastAPI API: `python -m uvicorn main:app --port=${FASTAPI_PORT:-8000} --host 0.0.0.0`
   - Bento service: `bentoml serve service:DiabetesService --port ${BENTO_PORT:-3000}`
4. Run post-deploy smoke tests:
   ```bash
   bash scripts/smoke_test.sh
   ```
