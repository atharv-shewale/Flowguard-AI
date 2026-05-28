"""
train.py
--------
Baseline ML training script for FlowGuard AI.

What this script does:
  1. Loads the synthetic SME dataset
  2. Preprocesses features (scaling)
  3. Trains a Random Forest classifier (scikit-learn baseline)
  4. Logs all metrics and artifacts to MLflow
  5. Saves the trained model as a .pkl file

This is the BASELINE model. For AutoML, see automl.py.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

# Allow importing save_meta from the same ml/ package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_meta import save_model_meta

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "ml", "dataset.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH  = os.path.join(MODELS_DIR, "saved_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "monthly_revenue",
    "pending_invoices",
    "avg_payment_delay",
    "monthly_expenses",
    "payroll_ratio",
    "cash_reserve",
    "vendor_due_amount",
]
TARGET_COL = "cash_flow_risk"


def load_data():
    """Load dataset and return X (features) and y (encoded labels)."""
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS].values

    # Encode target: Low=0, Medium=1, High=2 (alphabetical by LabelEncoder)
    le = LabelEncoder()
    y  = le.fit_transform(df[TARGET_COL])

    print(f"[DATA] Dataset shape: {df.shape}")
    print(f"       Classes: {le.classes_}")
    return X, y, le


def preprocess(X_train, X_test):
    """Scale features using StandardScaler (fit on train, transform both)."""
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    return X_train, X_test, scaler


def train():
    """Main training function with MLflow tracking."""

    # ── 1. Load data ─────────────────────────────────────────────────────────
    X, y, le = load_data()

    # ── 2. Train/test split (80/20 stratified) ────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 3. Preprocess ────────────────────────────────────────────────────────
    X_train, X_test, scaler = preprocess(X_train, X_test)

    # ── 4. Hyperparameters ───────────────────────────────────────────────────
    hyperparams = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "random_state": 42,
        "class_weight": "balanced",
    }

    # ── 5. MLflow experiment setup ───────────────────────────────────────────
    # Using local file-based tracking (no server needed).
    # Data is stored in ./mlruns/ -- start the UI afterwards with: mlflow ui
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("FlowGuard-AI-Baseline")

    with mlflow.start_run(run_name="RandomForest-Baseline"):

        # ── 6. Train model ───────────────────────────────────────────────────
        clf = RandomForestClassifier(**hyperparams)
        clf.fit(X_train, y_train)

        # ── 7. Evaluate ──────────────────────────────────────────────────────
        y_pred  = clf.predict(X_test)
        acc     = accuracy_score(y_test, y_pred)
        prec    = precision_score(y_test, y_pred, average="weighted")
        rec     = recall_score(y_test, y_pred, average="weighted")
        f1      = f1_score(y_test, y_pred, average="weighted")

        print("\n" + "=" * 55)
        print("  FlowGuard AI - Baseline Training Results")
        print("=" * 55)
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1 Score : {f1:.4f}")
        print("\n" + classification_report(y_test, y_pred, target_names=le.classes_))

        # ── 8. Log to MLflow ─────────────────────────────────────────────────
        mlflow.log_params(hyperparams)
        mlflow.log_metrics({
            "accuracy":  acc,
            "precision": prec,
            "recall":    rec,
            "f1_score":  f1,
        })
        mlflow.log_param("feature_columns", str(FEATURE_COLS))
        mlflow.log_param("target_classes",  str(list(le.classes_)))
        mlflow.log_param("train_samples",   X_train.shape[0])
        mlflow.log_param("test_samples",    X_test.shape[0])

        # Log the scikit-learn model artifact to MLflow
        mlflow.sklearn.log_model(clf, "random_forest_model")

        run_id = mlflow.active_run().info.run_id
        print(f"\n[OK] MLflow run logged. Run ID: {run_id}")

    # ── 9. Save artifacts locally ────────────────────────────────────────────
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    # ── 10. Save model metadata for FastAPI /model-info ──────────────────────
    save_model_meta(
        model_name    = type(clf).__name__,
        accuracy      = acc,
        f1_score      = f1,
        mlflow_run_id = run_id,
        model_version = "1.0.0",
    )

    print(f"\n[SAVED] Model   -> {MODEL_PATH}")
    print(f"[SAVED] Scaler  -> {SCALER_PATH}")
    print(f"[SAVED] Encoder -> {ENCODER_PATH}")

    return clf, scaler, le, acc


if __name__ == "__main__":
    # First generate the dataset if it doesn't exist
    if not os.path.exists(DATA_PATH):
        print("[WARN] Dataset not found. Generating synthetic data first...")
        import subprocess
        subprocess.run([sys.executable, os.path.join(BASE_DIR, "ml", "generate_dataset.py")], check=True)

    train()
