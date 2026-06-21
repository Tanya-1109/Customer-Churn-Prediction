"""Interactive Streamlit demo for churn prediction."""
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Churn Prediction Demo", page_icon="📊", layout="wide")


@st.cache_resource
def load_pipeline():
    return joblib.load("models/churn_pipeline.pkl")


pipeline = load_pipeline()
RISK_THRESHOLD = 0.35

st.title("📊 Customer Churn Prediction")
st.markdown("Enter customer details to predict their likelihood of churning.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)

with col2:
    st.subheader("Services")
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

with col3:
    st.subheader("Account & Billing")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.slider("Monthly Charges ($)", 0.0, 150.0, 70.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly_charges))

st.markdown("---")

if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
    input_data = pd.DataFrame([{
        'gender': gender, 'SeniorCitizen': senior, 'Partner': partner,
        'Dependents': dependents, 'tenure': tenure, 'PhoneService': phone,
        'MultipleLines': multiple_lines, 'InternetService': internet,
        'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
        'DeviceProtection': device_protection, 'TechSupport': tech_support,
        'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
        'Contract': contract, 'PaperlessBilling': paperless,
        'PaymentMethod': payment, 'MonthlyCharges': monthly_charges,
        'TotalCharges': str(total_charges)
    }])

    proba = pipeline.predict_proba(input_data)[:, 1][0]
    is_high_risk = proba >= RISK_THRESHOLD

    result_col1, result_col2 = st.columns([1, 2])

    with result_col1:
        if is_high_risk:
            st.error(f"### ⚠️ HIGH RISK\n**{proba:.1%}** churn probability")
        else:
            st.success(f"### ✅ LOW RISK\n**{proba:.1%}** churn probability")

    with result_col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            title={'text': "Churn Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#E8634C" if is_high_risk else "#4C9BE8"},
                'steps': [
                    {'range': [0, 35], 'color': "#E8F4EA"},
                    {'range': [35, 100], 'color': "#FCE8E5"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'value': RISK_THRESHOLD * 100
                }
            }
        ))
        fig.update_layout(height=250, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    if is_high_risk:
        st.markdown("### 💡 Suggested retention actions")
        if contract == "Month-to-month":
            st.write("- Offer a discount for switching to an annual contract")
        if online_security == "No" or tech_support == "No":
            st.write("- Offer a free trial of security/support add-ons")
        if payment == "Electronic check":
            st.write("- Encourage switching to automatic payment methods")