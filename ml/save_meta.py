"""
save_meta.py
------------
Saves model metadata (accuracy, version, etc.) after training so that
FastAPI's /model-info endpoint can return real values.

Run this automatically by train.py, or call it manually.
"""

import os
import pickle
import datetime

def save_model_meta(
    model_name: str,
    accuracy: float,
    f1_score: float,
    mlflow_run_id: str = None,
    model_version: str = "1.0.0",
):
    """Save a metadata dictionary alongside the model artifacts."""
    BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    os.makedirs(MODELS_DIR, exist_ok=True)

    meta = {
        "model_name":    model_name,
        "model_version": model_version,
        "accuracy":      round(accuracy, 4),
        "f1_score":      round(f1_score, 4),
        "trained_at":    datetime.datetime.utcnow().isoformat(),
        "mlflow_run_id": mlflow_run_id,
    }

    meta_path = os.path.join(MODELS_DIR, "model_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    print(f"[OK] Model metadata saved -> {meta_path}")
    return meta
