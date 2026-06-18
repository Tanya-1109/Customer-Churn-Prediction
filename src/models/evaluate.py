"""Model evaluation and explainability utilities."""
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix
)


def get_metrics(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def find_optimal_threshold(model, X_test, y_test, min_recall=0.75) -> float:
    """Find the threshold maximizing precision while satisfying a minimum recall constraint."""
    y_proba = model.predict_proba(X_test)[:, 1]
    best_t, best_precision = 0.5, 0

    for t in np.arange(0.1, 0.9, 0.01):
        y_pred_t = (y_proba >= t).astype(int)
        r = recall_score(y_test, y_pred_t)
        p = precision_score(y_test, y_pred_t, zero_division=0)
        if r >= min_recall and p > best_precision:
            best_t, best_precision = t, p

    return best_t


def get_shap_values(model, X_sample):
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X_sample), explainer


if __name__ == "__main__":
    import joblib
    model = joblib.load("models/churn_model.pkl")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    metrics = get_metrics(model, X_test, y_test)
    print("Test metrics:", metrics)

    threshold = find_optimal_threshold(model, X_test, y_test)
    print(f"Recommended threshold: {threshold:.2f}")