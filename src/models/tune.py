"""Hyperparameter tuning with Optuna."""
import optuna
import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from src.models.pipeline import build_pipeline


def objective(trial, X_train, y_train):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "random_state": 42,
        "eval_metric": "logloss",
        "use_label_encoder": False
    }

    pipeline = build_pipeline(model_params=params)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv, scoring='roc_auc', n_jobs=-1
    )
    return scores.mean()


def run_tuning(n_trials: int = 50):
    # Load RAW data — pipeline does all preprocessing internally
    raw = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    X = raw.drop(columns=['Churn'])
    y = raw['Churn'].map({'Yes': 1, 'No': 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri(
        "sqlite:///D:/Customer Churn Prediction/Customer-Churn-Prediction/mlflow.db"
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        study_name="churn-xgb-tuning",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    print(f"Starting Optuna tuning — {n_trials} trials...")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=True
    )

    best_params = study.best_params
    print(f"\nBest ROC-AUC (CV): {study.best_value:.4f}")
    print(f"Best params: {best_params}")

    with mlflow.start_run(run_name="optuna_best_xgboost"):
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_roc_auc", study.best_value)

    return best_params, study


if __name__ == "__main__":
    best_params, study = run_tuning(n_trials=50)