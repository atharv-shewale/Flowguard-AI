"""
automl.py
---------
AutoML pipeline for FlowGuard AI using PyCaret.

What this script does:
  1. Loads the SME financial dataset
  2. Initialises a PyCaret classification environment
  3. Compares MULTIPLE models automatically (Random Forest, XGBoost, LightGBM, etc.)
  4. Selects the BEST performing model based on Accuracy
  5. Fine-tunes (tunes) the best model with PyCaret's hyperparameter search
  6. Evaluates the tuned model and logs everything to MLflow
  7. Saves the best model for FastAPI and BentoML to consume

PyCaret abstracts away boilerplate so you can compare 15+ algorithms in seconds.
This is the production-quality model selection step.
"""

import os
import pickle
import pandas as pd
import mlflow

# ─────────────────────────────────────────────────────────────────────────────
# PyCaret imports – lazy-imported inside functions to avoid slow startup
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(BASE_DIR, "ml", "dataset.csv")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_automl_model.pkl")

os.makedirs(MODELS_DIR, exist_ok=True)

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


def run_automl():
    """
    Full AutoML pipeline using PyCaret.
    Returns the best trained model pipeline.
    """
    from pycaret.classification import (
        setup,
        compare_models,
        tune_model,
        evaluate_model,
        finalize_model,
        save_model,
        pull,
        get_config,
    )

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("[DATA] Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df_model = df[FEATURE_COLS + [TARGET_COL]].copy()
    print(f"    Shape: {df_model.shape}")
    print(f"    Target distribution:\n{df_model[TARGET_COL].value_counts()}\n")

    # ── 2. PyCaret Setup ─────────────────────────────────────────────────────
    # setup() handles:
    #   • Train/test split
    #   • Feature scaling
    #   • Encoding categorical features (target label)
    #   • Cross-validation configuration
    print("[SETUP] Initialising PyCaret environment...")
    clf_setup = setup(
        data             = df_model,
        target           = TARGET_COL,
        session_id       = 42,            # reproducibility seed
        train_size       = 0.8,           # 80% train / 20% test
        fold             = 5,             # 5-fold cross-validation
        normalize        = True,          # StandardScaler
        feature_selection= False,         # keep all 7 features (MVP)
        verbose          = False,         # suppress repetitive output
        log_experiment   = False,         # we log manually via MLflow below
    )

    # ── 3. Compare all available models ──────────────────────────────────────
    # PyCaret trains and cross-validates 15+ classifiers in one call.
    # We sort by Accuracy and return the top-3 for inspection.
    print("[AUTOML] Comparing all models (this may take 1–3 minutes)...\n")
    best_models = compare_models(
        sort          = "Accuracy",
        n_select      = 3,        # keep top 3
        exclude       = ["catboost"],   # CatBoost can be slow on some systems
        verbose       = True,
    )

    # compare_models() returns a list when n_select > 1
    best_model = best_models[0] if isinstance(best_models, list) else best_models
    comparison_df = pull()  # grab the comparison results dataframe

    print(f"\n[BEST] Best model: {type(best_model).__name__}")
    print(comparison_df.head(5))

    # ── 4. Tune the best model's hyperparameters ─────────────────────────────
    print("\n[TUNE] Tuning best model hyperparameters...")
    tuned_model = tune_model(
        best_model,
        optimize       = "Accuracy",
        n_iter         = 20,       # number of random search iterations
        verbose        = True,
    )
    tuned_results = pull()

    # ── 5. Finalise (retrain on full data) ────────────────────────────────
    # finalize_model() retrains on ALL data (train + test) so we use 100%
    # of the data for the production model.
    print("\n[FINALIZE] Finalising model on full dataset...")
    final_model = finalize_model(tuned_model)

    # ── 6. Save with PyCaret's save_model (includes full pipeline) ───────────
    pycaret_save_path = os.path.join(MODELS_DIR, "pycaret_best_model")
    save_model(final_model, pycaret_save_path)
    print(f"[SAVED] PyCaret pipeline saved -> {pycaret_save_path}.pkl")

    # ── 7. Also save as raw pickle for FastAPI compatibility ─────────────────
    with open(BEST_MODEL_PATH, "wb") as f:
        pickle.dump(final_model, f)
    print(f"[SAVED] Raw pickle saved -> {BEST_MODEL_PATH}")

    # ── 8. Log everything to MLflow ──────────────────────────────────────────
    mlflow.set_tracking_uri("file:./mlruns")   # local file-based (no server needed)
    mlflow.set_experiment("FlowGuard-AI-AutoML")

    with mlflow.start_run(run_name=f"AutoML-Best-{type(best_model).__name__}"):
        # Log best model type
        mlflow.log_param("best_model_type", type(best_model).__name__)
        mlflow.log_param("tuned_model_type", type(tuned_model).__name__)
        mlflow.log_param("pycaret_fold", 5)
        mlflow.log_param("features", str(FEATURE_COLS))

        # Log metrics from tuned model results
        if not tuned_results.empty:
            best_row = tuned_results.iloc[0]
            mlflow.log_metrics({
                "cv_accuracy":  float(best_row.get("Accuracy", 0)),
                "cv_precision": float(best_row.get("Precision", best_row.get("Prec.", 0))),
                "cv_recall":    float(best_row.get("Recall", 0)),
                "cv_f1":        float(best_row.get("F1", 0)),
            })

        # Log comparison table as artifact
        comp_path = os.path.join(MODELS_DIR, "model_comparison.csv")
        comparison_df.to_csv(comp_path, index=False)
        mlflow.log_artifact(comp_path)

        # Log the best model artifact
        mlflow.sklearn.log_model(final_model, "automl_best_model")

        run_id = mlflow.active_run().info.run_id
        print(f"\n[OK] MLflow AutoML run logged. Run ID: {run_id}")

    print("\n[OK] AutoML complete! Best model is ready for deployment.")
    return final_model, comparison_df


if __name__ == "__main__":
    # Generate dataset if needed
    if not os.path.exists(DATA_PATH):
        print("[WARN] Dataset not found. Generating...")
        import subprocess, sys
        gen_script = os.path.join(BASE_DIR, "ml", "generate_dataset.py")
        subprocess.run([sys.executable, gen_script], check=True)

    run_automl()
