# app.py
import streamlit as st
import pandas as pd
import joblib

# ---------------------------
# Load trained pipeline
# ---------------------------
@st.cache_resource
def load_model():
    return joblib.load("stroke_logreg_pipeline.pkl")  # or stroke_logreg_pipeline.pkl

pipe = load_model()

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Stroke Risk Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("Stroke Risk Prediction App")
st.write(
    "This app uses a Logistic Regression model trained on the "
    "[Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) "
    "to estimate the risk of stroke.\n\n"
    "**Important:** This is an educational demo and **not** medical advice."
)

st.markdown("---")

# ---------------------------
# Input form
# ---------------------------
st.header("Enter Patient Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=50)
    hypertension = st.selectbox("Hypertension (0 = No, 1 = Yes)", [0, 1])
    heart_disease = st.selectbox("Heart disease (0 = No, 1 = Yes)", [0, 1])
    ever_married = st.selectbox("Ever married?", ["Yes", "No"])

with col2:
    work_type = st.selectbox(
        "Work type",
        ["children", "Govt_job", "Never_worked", "Private", "Self-employed"]
    )
    Residence_type = st.selectbox("Residence type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input(
        "Average glucose level",
        min_value=0.0,
        value=100.0,
        step=0.1
    )
    bmi = st.number_input(
        "Body Mass Index (BMI)",
        min_value=0.0,
        value=25.0,
        step=0.1
    )
    smoking_status = st.selectbox(
        "Smoking status",
        ["formerly smoked", "never smoked", "smokes", "Unknown"]
    )

# Build input DataFrame with same columns as training
input_df = pd.DataFrame({
    "gender": [gender],
    "age": [age],
    "hypertension": [hypertension],
    "heart_disease": [heart_disease],
    "ever_married": [ever_married],
    "work_type": [work_type],
    "Residence_type": [Residence_type],
    "avg_glucose_level": [avg_glucose_level],
    "bmi": [bmi],
    "smoking_status": [smoking_status]
})

st.markdown("---")

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Stroke Risk"):
    proba = pipe.predict_proba(input_df)[0, 1]
    pred = pipe.predict(input_df)[0]

    st.subheader("Prediction Result")
    st.write(f"Estimated probability of stroke: **{proba:.3f}**")

    if pred == 1:
        st.error("Model prediction: **HIGH RISK (1)**")
    else:
        st.success("Model prediction: **LOW RISK (0)**")

    st.caption(
        "Note: This model is for learning and demonstration only. "
        "It must not be used for real medical decisions."
    )