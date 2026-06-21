"""Train final production model using best hyperparameters from tuning."""

import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import yaml
import os
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from src.models.pipeline import build_pipeline
from src.models.tune import run_tuning


def train_final(tune: bool = False, config_path: str = "config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load RAW data, split fresh — pipeline handles all preprocessing
    raw = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    X = raw.drop(columns=["Churn"])
    y = raw["Churn"].map({"Yes": 1, "No": 0})

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri(
        "sqlite:///D:/Customer Churn Prediction/Customer-Churn-Prediction/mlflow.db"
    )

    if tune:
        print("Running hyperparameter tuning first...")
        best_params, _ = run_tuning(n_trials=50)
    else:
        best_params = config["model"]

    with mlflow.start_run(run_name="final_production_model"):
        pipeline = build_pipeline(model_params=best_params)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "roc_auc": roc_auc_score(y_test, y_proba),
            "f1": f1_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
        }

        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "pipeline")

        # Save locally for API
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, "models/churn_pipeline.pkl")
        print(f"Pipeline saved to models/churn_pipeline.pkl")

        print("\nFinal model metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    return pipeline


if __name__ == "__main__":
    train_final(tune=False)  # set tune=True to run Optuna first
