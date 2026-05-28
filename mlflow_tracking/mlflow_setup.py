"""
mlflow_setup.py
---------------
MLflow local tracking server setup helper for FlowGuard AI.

This script:
  1. Creates the local MLflow tracking directory
  2. Configures experiment names
  3. Prints the commands to start the UI

MLflow stores all experiment data locally in ./mlruns/ by default.
No external server or cloud account is needed for the MVP demo.

Usage:
  python mlflow/mlflow_setup.py
"""

import os
import mlflow
import mlflow.sklearn

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Local file tracking URI – no server needed; data stored in ./mlruns/
TRACKING_URI = "file:./mlruns"

# Experiment names used across train.py and automl.py
EXPERIMENTS = [
    "FlowGuard-AI-Baseline",
    "FlowGuard-AI-AutoML",
]


def setup_mlflow():
    """
    Initialise MLflow experiments locally.
    Creates the experiment entries so they appear in the UI even before training.
    """
    print("=" * 55)
    print("  FlowGuard AI – MLflow Setup")
    print("=" * 55)

    # Set tracking URI (connect to local server)
    mlflow.set_tracking_uri(TRACKING_URI)
    print(f"\n📡  Tracking URI : {TRACKING_URI}")
    print(f"📂  Local store  : ./mlruns/  (created automatically)")

    # Create each experiment if it doesn't exist
    for exp_name in EXPERIMENTS:
        exp = mlflow.get_experiment_by_name(exp_name)
        if exp is None:
            exp_id = mlflow.create_experiment(
                exp_name,
                tags={"project": "FlowGuard-AI", "env": "development"},
            )
            print(f"✅  Created experiment: '{exp_name}' (ID: {exp_id})")
        else:
            print(f"ℹ️   Experiment already exists: '{exp_name}' (ID: {exp.experiment_id})")

    print("\n" + "=" * 55)
    print("  HOW TO USE MLFLOW")
    print("=" * 55)
    print("""
1. START THE MLFLOW SERVER (in a separate terminal):
   ─────────────────────────────────────────────────
   mlflow ui --host 127.0.0.1 --port 5000

2. OPEN THE UI IN YOUR BROWSER:
   ─────────────────────────────────────────────────
   http://127.0.0.1:5000

3. TRAIN MODELS (they will appear in the UI):
   ─────────────────────────────────────────────────
   python ml/train.py       ← Baseline (Random Forest)
   python ml/automl.py      ← AutoML (PyCaret)

4. WHAT YOU CAN DO IN THE UI:
   ─────────────────────────────────────────────────
   • Compare model accuracy across runs
   • View hyperparameters used in each run
   • Download trained model artifacts
   • See metrics charts over multiple runs
   • Register a model in the Model Registry

5. PROGRAMMATIC ACCESS (in Python):
   ─────────────────────────────────────────────────
   import mlflow
   mlflow.set_tracking_uri("http://127.0.0.1:5000")
   runs = mlflow.search_runs(experiment_names=["FlowGuard-AI-Baseline"])
   print(runs[["run_id", "metrics.accuracy", "params.n_estimators"]])
""")


if __name__ == "__main__":
    setup_mlflow()
