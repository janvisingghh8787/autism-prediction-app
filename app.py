import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- 1. Define Categories and Features (must match model training) ---
ALL_ETHNICITIES = ["White-European", "Black", "Latino", "Asian", "Middle Eastern", "Others"]

# Preferred final features (for retrained models)
FINAL_FEATURES = [
    'age', 'gender', 'jaundice', 'family_history',
    'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
    'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
    'ethnicity_Asian', 'ethnicity_Black', 'ethnicity_Latino',
    'ethnicity_Middle Eastern', 'ethnicity_Others', 'ethnicity_White-European'
]

# Old features used in previous training versions
OLD_FEATURES = [
    'ID', 'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
    'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
    'age', 'gender', 'ethnicity', 'jaundice', 'austim',
    'contry_of_res', 'used_app_before', 'result', 'age_desc', 'relation'
]

# --- 2. Load the trained model ---
model = None
try:
    with open("autism_prediction_model.pkl", "rb") as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"FATAL ERROR: Could not load model file. Error details: {e}")

# --- 3. Streamlit App Logic ---
if model and hasattr(model, 'predict'):
    st.set_page_config(page_title="Autism Prediction", page_icon="🧠", layout="centered")
    st.title("🧩 Autism Spectrum Disorder Prediction")
    st.markdown("### Answer the following MCQs to predict the likelihood of Autism")

    def user_input():
        st.subheader("👤 Personal Details")
        age = st.number_input("1️⃣ What is your age?", min_value=1, max_value=100, value=25)
        gender = st.radio("2️⃣ What is your gender?", ["Male", "Female"])
        ethnicity = st.selectbox("3️⃣ What is your ethnicity?", ALL_ETHNICITIES)
        jaundice = st.radio("4️⃣ Did you have jaundice at birth?", ["Yes", "No"])
        family_history = st.radio("5️⃣ Is there a family member with ASD?", ["Yes", "No"])

        st.subheader("🧠 Behavioral Screening Questions (A6-A10)")
        q6 = st.radio("6️⃣ Does the person often notice small sounds others don’t?", ["Yes", "No"])
        q7 = st.radio("7️⃣ Does the person concentrate more on the whole picture than details?", ["Yes", "No"])
        q8 = st.radio("8️⃣ Does the person find it easy to do more than one thing at once?", ["Yes", "No"])
        q9 = st.radio("9️⃣ Does the person find social situations easy?", ["Yes", "No"])
        q10 = st.radio("🔟 Does the person prefer to go out with others rather than stay home alone?", ["Yes", "No"])

        data = {
            "age": age,
            "gender": 1 if gender == "Male" else 0,
            "jaundice": 1 if jaundice == "Yes" else 0,
            "family_history": 1 if family_history == "Yes" else 0,
            "ethnicity": ethnicity,
            "A1_Score": 0, "A2_Score": 0, "A3_Score": 0, "A4_Score": 0, "A5_Score": 0,
            "A6_Score": 1 if q6 == "Yes" else 0,
            "A7_Score": 1 if q7 == "Yes" else 0,
            "A8_Score": 1 if q8 == "Yes" else 0,
            "A9_Score": 1 if q9 == "Yes" else 0,
            "A10_Score": 1 if q10 == "Yes" else 0
        }

        input_df = pd.DataFrame([data])

        # --- Encoding & Compatibility Fix ---
        # One-hot encode ethnicity
        ethnicity_ohe = pd.get_dummies(input_df['ethnicity'], prefix='ethnicity')
        input_df = pd.concat([input_df.drop('ethnicity', axis=1), ethnicity_ohe], axis=1)

        # Ensure all expected ethnicity columns exist
        for ethnic_group in ALL_ETHNICITIES:
            col_name = f'ethnicity_{ethnic_group}'
            if col_name not in input_df.columns:
                input_df[col_name] = 0

        # --- AUTO ALIGNMENT TO MODEL FEATURE NAMES ---
        try:
            model_features = model.feature_names_in_  # Available in sklearn >=1.0
        except AttributeError:
            # Fallback if not stored
            model_features = FINAL_FEATURES if set(FINAL_FEATURES).issubset(input_df.columns) else OLD_FEATURES

        # Add any missing columns expected by model
        for col in model_features:
            if col not in input_df.columns:
                input_df[col] = 0

        # Keep only expected columns in correct order
        input_df = input_df[model_features]

        return input_df

    input_df = user_input()

    if st.button("🔍 Predict") and input_df is not None:
        try:
            prediction = model.predict(input_df)[0]
            if hasattr(model, "predict_proba"):
                prediction_proba = model.predict_proba(input_df)[:, 1][0] * 100
            else:
                prediction_proba = 50  # Fallback if probability not supported

            if prediction == 1:
                st.error(f"Prediction: High Likelihood of ASD ({prediction_proba:.2f}% Probability)")
            else:
                st.success(f"Prediction: Low Likelihood of ASD ({100 - prediction_proba:.2f}% Probability)")

            st.warning("⚠️ Disclaimer: This prediction is not a medical diagnosis. Please consult a healthcare professional.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
