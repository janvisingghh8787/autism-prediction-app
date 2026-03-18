import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Autism Predictor",
    page_icon="🧠",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
}
h1 {
    text-align: center;
    color: #00C9A7;
}
h2, h3 {
    color: #00C9A7;
}
.stButton>button {
    background: linear-gradient(90deg, #00C9A7, #00A8E8);
    color: white;
    border-radius: 12px;
    height: 3em;
    font-size: 18px;
    width: 100%;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<h1>🧠 Autism Spectrum Disorder Predictor</h1>
<p style='text-align:center; font-size:18px;'>
AI-powered screening using Machine Learning
</p>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Model Info")
st.sidebar.markdown("""
- **Model:** Random Forest  
- **Accuracy:** ~92%  
- **Type:** Classification  
- **Features:** Behavioral + Demographic  
""")

# ---------------- DATA ----------------
ALL_ETHNICITIES = ["White-European", "Black", "Latino", "Asian", "Middle Eastern", "Others"]

FINAL_FEATURES = [
    'age', 'gender', 'jaundice', 'family_history',
    'A1_Score','A2_Score','A3_Score','A4_Score','A5_Score',
    'A6_Score','A7_Score','A8_Score','A9_Score','A10_Score',
    'ethnicity_Asian','ethnicity_Black','ethnicity_Latino',
    'ethnicity_Middle Eastern','ethnicity_Others','ethnicity_White-European'
]

# ---------------- LOAD MODEL ----------------
model = None
try:
    with open("autism_prediction_model.pkl", "rb") as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model: {e}")

# ---------------- INPUT UI ----------------
def user_input():
    st.divider()
    st.subheader("👤 Personal Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 100, 25)
        gender = st.radio("Gender", ["Male", "Female"])

    with col2:
        ethnicity = st.selectbox("Ethnicity", ALL_ETHNICITIES)
        jaundice = st.radio("Jaundice at birth?", ["Yes", "No"])
        family_history = st.radio("Family history of ASD?", ["Yes", "No"])

    st.divider()
    st.subheader("🧠 Behavioral Questions")

    col3, col4 = st.columns(2)

    with col3:
        q6 = st.radio("Notices small sounds others don’t?", ["Yes", "No"])
        q7 = st.radio("Focuses on whole picture?", ["Yes", "No"])
        q8 = st.radio("Multitasking ability?", ["Yes", "No"])

    with col4:
        q9 = st.radio("Social situations easy?", ["Yes", "No"])
        q10 = st.radio("Prefers going out?", ["Yes", "No"])

    data = {
        "age": age,
        "gender": 1 if gender == "Male" else 0,
        "jaundice": 1 if jaundice == "Yes" else 0,
        "family_history": 1 if family_history == "Yes" else 0,
        "ethnicity": ethnicity,
        "A1_Score":0,"A2_Score":0,"A3_Score":0,"A4_Score":0,"A5_Score":0,
        "A6_Score":1 if q6=="Yes" else 0,
        "A7_Score":1 if q7=="Yes" else 0,
        "A8_Score":1 if q8=="Yes" else 0,
        "A9_Score":1 if q9=="Yes" else 0,
        "A10_Score":1 if q10=="Yes" else 0
    }

    input_df = pd.DataFrame([data])

    # Encoding
    ethnicity_ohe = pd.get_dummies(input_df['ethnicity'], prefix='ethnicity')
    input_df = pd.concat([input_df.drop('ethnicity', axis=1), ethnicity_ohe], axis=1)

    for eth in ALL_ETHNICITIES:
        col = f'ethnicity_{eth}'
        if col not in input_df.columns:
            input_df[col] = 0

    try:
        model_features=model.features_name_in_
    except:
        model_features= FINAL_FEATURES
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_features]


    return input_df

# ---------------- MAIN ----------------
if model:
    input_df = user_input()

    st.divider()

    if st.button("🚀 Predict"):
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[:,1][0]*100

        st.progress(int(prob))

        if prediction == 1:
            st.markdown(f"""
            <div style='background:#ff4b4b;padding:25px;border-radius:12px;text-align:center'>
                <h2>⚠️ High Likelihood of ASD</h2>
                <h3>{prob:.2f}% Probability</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:#00C9A7;padding:25px;border-radius:12px;text-align:center'>
                <h2>✅ Low Likelihood of ASD</h2>
                <h3>{100-prob:.2f}% Confidence</h3>
            </div>
            """, unsafe_allow_html=True)

        st.info("⚠️ This is not a medical diagnosis. Consult a professional.")