# MLflow Setup and Tracking

MLflow is integrated into this project to track the parameters, metrics, and models during the AutoML training process.

## Viewing MLflow Dashboard

After running `python train.py`, the training script logs an experiment run into a local `mlruns` directory. To visualize the results in a web browser:

1. Open a terminal.
2. Ensure your virtual environment is active.
3. Start the MLflow UI server by running:
   ```bash
   mlflow ui
   ```
4. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```
*(Note: Since Flask also runs on port 5000, you may want to start MLflow on a different port if both are running simultaneously: `mlflow ui --port 5001`)*

## What is Tracked?
- **Parameters**: `time_budget` (the time given to FLAML to search for models), `best_estimator` (the algorithm chosen by FLAML, e.g., lgbm, xgboost, rf).
- **Metrics**: `test_r2` (the R-squared score on the hold-out test set).
- **Artifacts**: The final `model.pkl` file is logged as an artifact tied to the run.
