import pandas as pd
import pytest
from src.models.pipeline import build_pipeline


SAMPLE = pd.DataFrame([{
    'customerID': 'T-001', 'gender': 'Male', 'SeniorCitizen': 0,
    'Partner': 'No', 'Dependents': 'No', 'tenure': 5,
    'PhoneService': 'Yes', 'MultipleLines': 'No',
    'InternetService': 'Fiber optic', 'OnlineSecurity': 'No',
    'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': 'No',
    'StreamingTV': 'Yes', 'StreamingMovies': 'No',
    'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 70.0, 'TotalCharges': '350.0'
}])


def test_pipeline_output_shape():
    """Pipeline should return a probability between 0 and 1."""
    import pandas as pd
    raw = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    X = raw.drop(columns=['Churn'])
    y = raw['Churn'].map({'Yes': 1, 'No': 0})

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    proba = pipeline.predict_proba(SAMPLE)[:, 1]
    assert len(proba) == 1
    assert 0 <= proba[0] <= 1


if __name__ == "__main__":
    test_pipeline_output_shape()
    print("Pipeline test passed!")