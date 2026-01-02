import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# -------------------------------------------------------------
# Train pipeline ONCE and cache it (no .pkl, no joblib)
# -------------------------------------------------------------
@st.cache_resource
def train_pipeline():
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")

    # Basic cleaning
    df = df.copy()
    df.drop(columns=["id"], inplace=True)
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    X = df.drop("stroke", axis=1)
    y = df["stroke"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_features = ["age", "avg_glucose_level", "bmi"]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    log_reg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
    )

    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", log_reg)])
    pipe.fit(X_train, y_train)
    return pipe


pipe = train_pipeline()  # model is trained & cached here

# -------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------
st.set_page_config(
    page_title="Stroke Risk Prediction",
    page_icon="🩺",
    layout="centered",
)

st.title("Stroke Risk Prediction App")
st.write(
    "This app uses a Logistic Regression model trained on the "
    "[Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) "
    "to estimate stroke risk.\n\n"
    "**Important:** This is an educational demo and **not** medical advice."
)

st.markdown("---")
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
        ["children", "Govt_job", "Never_worked", "Private", "Self-employed"],
    )
    Residence_type = st.selectbox("Residence type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input(
        "Average glucose level", min_value=0.0, value=100.0, step=0.1
    )
    bmi = st.number_input(
        "Body Mass Index (BMI)", min_value=0.0, value=25.0, step=0.1
    )
    smoking_status = st.selectbox(
        "Smoking status",
        ["formerly smoked", "never smoked", "smokes", "Unknown"],
    )

input_df = pd.DataFrame(
    {
        "gender": [gender],
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "ever_married": [ever_married],
        "work_type": [work_type],
        "Residence_type": [Residence_type],
        "avg_glucose_level": [avg_glucose_level],
        "bmi": [bmi],
        "smoking_status": [smoking_status],
    }
)

st.markdown("---")

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
