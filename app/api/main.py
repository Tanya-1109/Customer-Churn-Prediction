"""FastAPI inference service for churn prediction."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import shap
import numpy as np
from contextlib import asynccontextmanager

from app.api.schemas import CustomerInput, ChurnPrediction

MODEL_PATH = "models/churn_pipeline.pkl"
RISK_THRESHOLD = 0.35  # from Phase 5 threshold tuning

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup, not on every request."""
    ml_models["pipeline"] = joblib.load(MODEL_PATH)
    # Build a SHAP explainer against the trained classifier step
    classifier = ml_models["pipeline"].named_steps["classifier"]
    ml_models["explainer"] = shap.TreeExplainer(classifier)
    yield
    ml_models.clear()


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts customer churn probability from account and service data.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in real production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "Churn Prediction API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": "pipeline" in ml_models}


@app.post("/predict", response_model=ChurnPrediction)
def predict_churn(customer: CustomerInput):
    try:
        pipeline = ml_models["pipeline"]
        input_df = pd.DataFrame([customer.model_dump()])

        proba = pipeline.predict_proba(input_df)[:, 1][0]
        prediction = "High Risk" if proba >= RISK_THRESHOLD else "Low Risk"

        # Get top SHAP-driven risk factors for this prediction
        preprocessed = pipeline.named_steps["feature_engineering"].transform(input_df)
        transformed = pipeline.named_steps["preprocessor"].transform(preprocessed)
        shap_values = ml_models["explainer"].shap_values(transformed)

        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        shap_row = shap_values[0]
        top_idx = np.argsort(np.abs(shap_row))[-3:][::-1]
        top_factors = [feature_names[i] for i in top_idx]

        return ChurnPrediction(
            churn_probability=round(float(proba), 4),
            churn_prediction=prediction,
            risk_threshold_used=RISK_THRESHOLD,
            top_risk_factors=top_factors,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
