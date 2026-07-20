from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from flaml import AutoML
import mlflow
import joblib

def main():
    # Set MLflow experiment
    mlflow.set_experiment("Diabetes_Progression_AutoML")
    
    with mlflow.start_run():
        # Load dataset
        print("Loading dataset...")
        diabetes = load_diabetes()
        X, y = diabetes.data, diabetes.target
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Configure AutoML
        automl = AutoML()
        automl_settings = {
            "time_budget": 15,  # 15 seconds budget for quick demonstration
            "metric": 'r2',
            "task": 'regression',
            "estimator_list": ["lgbm", "rf", "extra_tree"],
            "log_file_name": 'diabetes_automl.log',
            "seed": 42
        }
        
        print("Starting AutoML training...")
        # Train model
        automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
        
        # Get best model info
        print(f"Best ML learner: {automl.best_estimator}")
        print(f"Best R2 score on validation data: {1 - automl.best_loss}")
        
        # Evaluate on test set
        test_r2 = automl.score(X_test, y_test)
        print(f"Test R2 score: {test_r2}")
        
        # Log metrics and params to MLflow
        mlflow.log_param("time_budget", automl_settings["time_budget"])
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_param("best_estimator", automl.best_estimator)
        
        # Export model
        model_path = "model.pkl"
        joblib.dump(automl, model_path)
        print(f"Model saved to {model_path}")
        
        # Log the model artifact to MLflow
        mlflow.log_artifact(model_path)

if __name__ == "__main__":
    main()
