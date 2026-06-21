"""Production pipeline: preprocessing + model in a single sklearn object."""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb

# ── Custom transformer for feature engineering ───────────────
# Wraps your Phase 3 logic so it fits into sklearn's Pipeline API

SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer that adds domain-driven features."""

    def fit(self, X, y=None):
        return self  # stateless transformer — nothing to learn

    def transform(self, X):
        X = X.copy()

        # Coerce types
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0)

        # Engineered features from EDA (Phase 2 & 3)
        X["is_new_customer"] = (X["tenure"] <= 12).astype(int)
        X["total_services"] = (X[SERVICE_COLS] == "Yes").sum(axis=1)
        X["avg_monthly_spend"] = X["TotalCharges"] / (X["tenure"] + 1)
        X["is_month_to_month"] = (X["Contract"] == "Month-to-month").astype(int)
        X["is_electronic_check"] = (X["PaymentMethod"] == "Electronic check").astype(
            int
        )
        X["has_internet"] = (X["InternetService"] != "No").astype(int)

        # Simplify redundant category values
        for col in SERVICE_COLS + ["MultipleLines"]:
            X[col] = X[col].replace(["No internet service", "No phone service"], "No")

        return X


# ── Column groups after feature engineering ──────────────────
BINARY_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "MultipleLines",
    "SeniorCitizen",
] + SERVICE_COLS

MULTICLASS_COLS = ["InternetService", "Contract", "PaymentMethod"]

NUM_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "avg_monthly_spend",
    "total_services",
    "is_new_customer",
    "is_month_to_month",
    "is_electronic_check",
    "has_internet",
]


def build_pipeline(model_params: dict = None) -> Pipeline:
    """
    Build end-to-end pipeline.
    model_params: XGBoost hyperparameters (from tuning or config).
    """
    if model_params is None:
        model_params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "eval_metric": "logloss",
            "use_label_encoder": False,
        }

    # ColumnTransformer applies different transforms to different column groups
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("cat_binary", OrdinalEncoder(), BINARY_COLS),
            ("cat_multi", OrdinalEncoder(), MULTICLASS_COLS),
        ],
        remainder="drop",
    )  # drops customerID and other unused cols

    pipeline = Pipeline(
        steps=[
            ("feature_engineering", ChurnFeatureEngineer()),
            ("preprocessor", preprocessor),
            ("classifier", xgb.XGBClassifier(**model_params)),
        ]
    )

    return pipeline
