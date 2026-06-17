"""Training script for churn prediction models."""
import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import yaml
import joblib
import os


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }


def train(config_path="config.yaml"):
    config = load_config(config_path)

    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    mlflow.set_tracking_uri("sqlite:///D:/Customer Churn Prediction/Customer-Churn-Prediction/notebooks/mlflow.db")
    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run(run_name="xgboost_production"):
        params = config["model"]
        model = xgb.XGBClassifier(**params, eval_metric="logloss")
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "model")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/churn_model.pkl")

        print("Training complete. Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    return model, metrics


if __name__ == "__main__":
    train()