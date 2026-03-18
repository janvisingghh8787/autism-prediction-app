import streamlit as st
import pandas as pd
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Autism Predictor", page_icon="🧠", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
h1 {text-align:center; color:#00C9A7;}
.stButton>button {
    background: linear-gradient(90deg, #00C9A7, #00A8E8);
    color: white;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<h1>🧠 Autism Spectrum Disorder Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered screening using Machine Learning</p>", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Model Info")
st.sidebar.write("""
- Model: Random Forest  
- Accuracy: ~92%  
- Type: Classification  
""")

# ---------------- LOAD MODEL ----------------
try:
    model = pickle.load(open("autism_prediction_model.pkl", "rb"))
except:
    st.error("❌ Model not found")
    st.stop()

# ---------------- GET MODEL FEATURES ----------------
try:
    model_features = list(model.feature_names_in_)
except:
    st.error("❌ Model missing feature names. Retrain with sklearn >=1.0")
    st.stop()

# ---------------- INPUT ----------------
ALL_ETHNICITIES = ["White-European", "Black", "Latino", "Asian", "Middle Eastern", "Others"]

st.subheader("👤 Personal Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 100, 25)
    gender = st.radio("Gender", ["Male", "Female"])

with col2:
    ethnicity = st.selectbox("Ethnicity", ALL_ETHNICITIES)
    jaundice = st.radio("Jaundice at birth?", ["Yes", "No"])
    family_history = st.radio("Family history of ASD?", ["Yes", "No"])

st.subheader("🧠 Behavioral Questions")

q6 = st.radio("Notices small sounds?", ["Yes", "No"])
q7 = st.radio("Focuses on whole picture?", ["Yes", "No"])
q8 = st.radio("Multitasking ability?", ["Yes", "No"])
q9 = st.radio("Social situations easy?", ["Yes", "No"])
q10 = st.radio("Prefers going out?", ["Yes", "No"])

# ---------------- BUILD INPUT ----------------
data = {
    "age": age,
    "gender": 1 if gender == "Male" else 0,
    "jaundice": 1 if jaundice == "Yes" else 0,
    "family_history": 1 if family_history == "Yes" else 0,
    "ethnicity": ethnicity,
    "A1_Score": 0,
    "A2_Score": 0,
    "A3_Score": 0,
    "A4_Score": 0,
    "A5_Score": 0,
    "A6_Score": 1 if q6 == "Yes" else 0,
    "A7_Score": 1 if q7 == "Yes" else 0,
    "A8_Score": 1 if q8 == "Yes" else 0,
    "A9_Score": 1 if q9 == "Yes" else 0,
    "A10_Score": 1 if q10 == "Yes" else 0
}

input_df = pd.DataFrame([data])

# ---------------- ENCODING ----------------
ethnicity_ohe = pd.get_dummies(input_df['ethnicity'], prefix='ethnicity')
input_df = pd.concat([input_df.drop('ethnicity', axis=1), ethnicity_ohe], axis=1)

# ---------------- ALIGN FEATURES (KEY FIX) ----------------
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[model_features]

# Convert to numeric
input_df = input_df.astype(float)

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict"):
    try:
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[:, 1][0] * 100

        st.progress(int(prob))

        if prediction == 1:
            st.markdown(f"""
            <div style='background:#ff4b4b;padding:20px;border-radius:10px;text-align:center'>
                <h2>⚠️ High Likelihood of ASD</h2>
                <h3>{prob:.2f}% Probability</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:#00C9A7;padding:20px;border-radius:10px;text-align:center'>
                <h2>✅ Low Likelihood of ASD</h2>
                <h3>{100-prob:.2f}% Confidence</h3>
            </div>
            """, unsafe_allow_html=True)

        st.info("⚠️ This is not a medical diagnosis. Consult a professional.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")