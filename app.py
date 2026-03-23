# app.py — PredictPal Loan Default Predictor

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="PredictPal", layout="centered")

# Load model pipeline
@st.cache_resource
def load_model():
    return joblib.load("model_pipeline.joblib")

pipe = load_model()

st.title("🏦 PredictPal — Loan Default Predictor")
st.write("Predict whether a borrower is likely to default based on their loan details.")

# Extract column names
preprocessor = pipe.named_steps['preprocessor']
numeric_cols = preprocessor.transformers_[0][2]
categorical_cols = preprocessor.transformers_[1][2] if len(preprocessor.transformers_) > 1 else []

st.sidebar.header("Enter Applicant Details")
input_data = {}

# Numeric inputs
for col in numeric_cols:
    input_data[col] = st.sidebar.number_input(col, value=0.0)

# Categorical inputs
for col in categorical_cols:
    input_data[col] = st.sidebar.text_input(col, value="")

# When button is clicked
if st.sidebar.button("Predict Default Probability"):
    X_new = pd.DataFrame([input_data])
    prob = pipe.predict_proba(X_new)[0][1]
    pred = pipe.predict(X_new)[0]

    st.subheader("📊 Prediction Result")
    st.metric("Probability of Default", f"{prob:.2%}")
    st.write("**Decision:**", "🔴 High Risk (Default likely)" if pred == 1 else "🟢 Low Risk (Likely to repay)")

    # Feature importance note
    st.caption("This prediction is generated using your trained ML model.")