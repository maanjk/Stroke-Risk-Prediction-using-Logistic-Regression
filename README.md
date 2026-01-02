# Stroke Risk Prediction using Logistic Regression and Streamlit

An end‑to‑end machine‑learning project that predicts stroke risk from basic clinical and demographic information using the **Stroke Prediction Dataset** from Kaggle.  

The pipeline includes:

- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Training a **Logistic Regression** classifier
- Model evaluation with multiple metrics
- Exporting the trained pipeline
- A **Streamlit** web app for interactive predictions

> ⚠️ **Disclaimer:** This project is for educational purposes only and must **not** be used for real medical diagnosis or treatment decisions.

---

## 1. Dataset

- **Name:** Stroke Prediction Dataset  
- **Source:** Kaggle – [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)  
- **Rows:** 5,110  
- **Columns:** 12 (including `id` and `stroke`)

**Main features**

- `gender`
- `age`
- `hypertension`
- `heart_disease`
- `ever_married`
- `work_type`
- `Residence_type`
- `avg_glucose_level`
- `bmi`
- `smoking_status`

**Target**

- `stroke` – binary label (0 = no stroke, 1 = stroke)

---

## 2. Problem Statement

Given a patient’s demographic and clinical attributes, predict whether they are at risk of stroke.

This is a **binary classification** problem:

- **Input:** Patient features (age, hypertension, heart disease, etc.)
- **Output:** Probability of stroke and a final class prediction (0 or 1)

---

## 3. Project Structure

Example structure of this repository:

```text
.
├── app.py                         # Streamlit web app
├── stroke_logreg_pipeline.pkl     # Trained sklearn Pipeline (preprocess + model)
├── notebook.ipynb                 # Jupyter/Kaggle notebook with EDA + training
├── requirements.txt               # Python dependencies for the app
└── README.md                      # This file
