import streamlit as st
import pandas as pd
import joblib

# ---------------- Load Files ----------------
model = joblib.load("XGBOOST_MODEL.pkl")
scaler = joblib.load("StandardScale.pkl")
encoders = joblib.load("LabelEncoder.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn or not.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )
    
    online_backup = st.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

with col2:
    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=72,
        value=12
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        value=1000.0
    )
    
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )

# ---------------- Prediction ----------------

if st.button("Predict Churn", use_container_width=True):

    data = pd.DataFrame({
    "gender":[gender],
    "SeniorCitizen":[senior],
    "Partner":[partner],
    "Dependents":[dependents],
    "tenure":[tenure],
    "PhoneService":[phone_service],
    "MultipleLines":[multiple_lines],
    "InternetService":[internet_service],
    "OnlineSecurity":[online_security],
    "OnlineBackup":[online_backup],
    "DeviceProtection":[device_protection],
    "TechSupport":[tech_support],
    "StreamingTV":[streaming_tv],
    "StreamingMovies":[streaming_movies],
    "Contract":[contract],
    "PaperlessBilling":[paperless_billing],
    "PaymentMethod":[payment_method],
    "MonthlyCharges":[monthly_charges],
    "TotalCharges":[total_charges]
    })

    # Encode categorical columns
    for col in data.select_dtypes(include="object").columns:
        data[col] = encoders[col].transform(data[col])

    # Scale numerical columns
    data[["tenure","MonthlyCharges","TotalCharges"]] = scaler.transform(
        data[["tenure","MonthlyCharges","TotalCharges"]]
    )

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)

    if prediction == 1:
        st.error(
            f"❌ Customer is likely to churn\n\nProbability: {probability[0][1]*100:.2f}%"
        )
    else:
        st.success(
            f"✅ Customer is likely to stay\n\nProbability: {probability[0][0]*100:.2f}%"
        )