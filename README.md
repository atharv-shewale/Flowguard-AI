# Advanced Web Development Lab - ML Application

This project is a full-stack Machine Learning application built using Flask, FastAPI, and MySQL, with MLOps tools integrated (Jenkins, BentoML, MLflow, and FLAML for AutoML). It predicts disease progression based on the scikit-learn `load_diabetes()` dataset.

## System Design

1. **User Interface (Flask frontend)**: HTML forms styled with modern CSS (glassmorphism) accept 10 clinical features from the user.
2. **Web Applications**:
   - **Flask App**: Serves the UI, handles form submissions, predicts the progression using a trained `.pkl` model, logs data to MySQL, and renders the result.
   - **FastAPI App**: Provides a backend API (`/predict`) taking JSON inputs and returning predictions. It also logs data to MySQL and provides an interactive Swagger UI.
3. **Database (MySQL)**: Stores every prediction request, recording the 10 features, the prediction output, and the source (flask_ui vs fastapi).
4. **MLOps**:
   - **FLAML (AutoML)**: Automatically searches for the best regression model.
   - **MLflow**: Tracks training runs, metrics, and parameters.
   - **BentoML**: Packages the trained model into a production-ready container format.
   - **Jenkins**: A `Jenkinsfile` orchestrates the CI/CD pipeline (lint, test, train, package).

## Setup & Running Instructions

### 1. Database Setup
Ensure you have MySQL running locally (e.g., via XAMPP). Create the database and table:
```bash
mysql -u root -p < schema.sql
```
*(By default, the apps connect to `localhost` with user `root` and no password. You can set `DB_HOST`, `DB_USER`, `DB_PASSWORD` env vars if needed).*

### 2. Python Environment Setup
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
# source venv/bin/activate

pip install -r requirements.txt
```

### 3. Model Training
Run the training script to train the model using FLAML, track it with MLflow, and export `model.pkl`:
```bash
python train.py
```

### 4. Running the Flask App
```bash
# Using Flask CLI
cd flask_app
flask run --port=5000
```
Visit `http://127.0.0.1:5000` to interact with the UI.

### 5. Running the FastAPI App
```bash
cd fastapi_app
uvicorn main:app --reload --port=8000
```
Visit `http://127.0.0.1:8000/docs` to test the API via Swagger UI.

### 6. Packaging with BentoML
```bash
# Register the model to local BentoML store
python save_bento_model.py

# Build the Bento
bentoml build
```
