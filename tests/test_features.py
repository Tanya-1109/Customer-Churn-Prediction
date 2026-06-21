import pandas as pd
import sys

sys.path.append("..")
from src.features.build_features import clean_data, engineer_features


def test_clean_data_converts_total_charges():
    df = pd.DataFrame(
        {
            "customerID": ["1", "2"],
            "TotalCharges": ["100.5", " "],
            "Churn": ["Yes", "No"],
            "tenure": [5, 0],
        }
    )
    cleaned = clean_data(df)
    assert cleaned["TotalCharges"].dtype == float
    assert cleaned["TotalCharges"].iloc[1] == 0
    assert cleaned["Churn"].tolist() == [1, 0]
    assert "customerID" not in cleaned.columns


def test_engineer_features_creates_new_columns():
    df = pd.DataFrame(
        {
            "tenure": [5, 20],
            "TotalCharges": [100, 1000],
            "MonthlyCharges": [50, 60],
            "Contract": ["Month-to-month", "Two year"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "InternetService": ["Fiber optic", "No"],
            "OnlineSecurity": ["No", "No internet service"],
            "OnlineBackup": ["No", "No internet service"],
            "DeviceProtection": ["No", "No internet service"],
            "TechSupport": ["No", "No internet service"],
            "StreamingTV": ["No", "No internet service"],
            "StreamingMovies": ["No", "No internet service"],
            "MultipleLines": ["No", "No phone service"],
        }
    )
    result = engineer_features(df)
    assert "is_new_customer" in result.columns
    assert result["is_new_customer"].tolist() == [1, 0]
    assert result["is_month_to_month"].tolist() == [1, 0]


if __name__ == "__main__":
    test_clean_data_converts_total_charges()
    test_engineer_features_creates_new_columns()
    print("All tests passed!")
