"""Feature engineering pipeline for churn prediction."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

BINARY_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "MultipleLines",
] + SERVICE_COLS

MULTICLASS_COLS = ["InternetService", "Contract", "PaymentMethod", "tenure_group"]

NUM_COLS_TO_SCALE = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: types, target encoding, drop ID."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df.drop(columns=["customerID"], inplace=True, errors="ignore")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features derived from EDA insights."""
    df = df.copy()

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"],
    )
    df["is_new_customer"] = (df["tenure"] <= 12).astype(int)
    df["total_services"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    df["has_internet"] = (df["InternetService"] != "No").astype(int)

    for col in SERVICE_COLS + ["MultipleLines"]:
        df[col] = df[col].replace(["No internet service", "No phone service"], "No")

    return df


def encode_and_scale(
    df: pd.DataFrame,
    scaler: StandardScaler = None,
    label_encoders: dict = None,
    fit: bool = True,
):
    """Encode categoricals and scale numericals. Returns df + fitted transformers."""
    df = df.copy()

    if label_encoders is None:
        label_encoders = {}

    for col in BINARY_COLS:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        else:
            df[col] = label_encoders[col].transform(df[col])

    df = pd.get_dummies(df, columns=MULTICLASS_COLS, drop_first=True)

    if scaler is None:
        scaler = StandardScaler()

    if fit:
        df[NUM_COLS_TO_SCALE] = scaler.fit_transform(df[NUM_COLS_TO_SCALE])
    else:
        df[NUM_COLS_TO_SCALE] = scaler.transform(df[NUM_COLS_TO_SCALE])

    return df, scaler, label_encoders


def run_pipeline(raw_path: str):
    """Full feature engineering pipeline — used by train.py and predict.py."""
    df = pd.read_csv(raw_path)
    df = clean_data(df)
    df = engineer_features(df)
    df, scaler, encoders = encode_and_scale(df, fit=True)
    return df, scaler, encoders


if __name__ == "__main__":
    df, scaler, encoders = run_pipeline("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print(f"Processed shape: {df.shape}")
    print(df.head())
