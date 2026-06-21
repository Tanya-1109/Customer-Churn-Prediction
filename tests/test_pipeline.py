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

import pandas as pd
import pytest
from src.models.pipeline import build_pipeline


def make_fake_raw_data(n=50):
    """Generate a small synthetic dataset matching the raw schema, for testing only."""
    import numpy as np
    np.random.seed(42)
    return pd.DataFrame({
        'customerID': [f'C{i:04d}' for i in range(n)],
        'gender': np.random.choice(['Male', 'Female'], n),
        'SeniorCitizen': np.random.choice([0, 1], n),
        'Partner': np.random.choice(['Yes', 'No'], n),
        'Dependents': np.random.choice(['Yes', 'No'], n),
        'tenure': np.random.randint(0, 72, n),
        'PhoneService': np.random.choice(['Yes', 'No'], n),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n),
        'PaymentMethod': np.random.choice([
            'Electronic check', 'Mailed check',
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ], n),
        'MonthlyCharges': np.random.uniform(20, 120, n).round(2),
        'TotalCharges': np.random.uniform(20, 8000, n).round(2).astype(str),
        'Churn': np.random.choice(['Yes', 'No'], n)
    })


def test_pipeline_output_shape():
    """Pipeline should return a probability between 0 and 1, using synthetic data."""
    raw = make_fake_raw_data(n=50)
    X = raw.drop(columns=['Churn'])
    y = raw['Churn'].map({'Yes': 1, 'No': 0})

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    sample = X.iloc[[0]]
    proba = pipeline.predict_proba(sample)[:, 1]
    assert len(proba) == 1
    assert 0 <= proba[0] <= 1


if __name__ == "__main__":
    test_pipeline_output_shape()
    print("Pipeline test passed!")