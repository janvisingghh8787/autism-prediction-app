import streamlit as st
import pandas as pd
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Autism Predictor", page_icon="🧠", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
h1 {text-align:center; color:#00E5FF;}
.stButton>button {
    background: linear-gradient(90deg, #00E5FF, #00C9A7);
    color: black;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<h1>🧠 Autism Spectrum Disorder Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered screening + Doctor Recommendations</p>", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("autism_prediction_model.pkl", "rb"))
model_features = list(model.feature_names_in_)

ALL_ETHNICITIES = ["White-European", "Black", "Latino", "Asian", "Middle Eastern", "Others"]

# ---------------- LOCATION ----------------
st.subheader("📍 Enter Your City")
city = st.text_input("City (Mumbai, Delhi, Pune)")

# ---------------- PERSONAL DETAILS ----------------
st.subheader("👤 Personal Details")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 100, 25)
    gender = st.radio("Gender", ["Male", "Female"])

with col2:
    ethnicity = st.selectbox("Ethnicity", ALL_ETHNICITIES)
    jaundice = st.radio("Jaundice at birth?", ["Yes", "No"])
    family_history = st.radio("Family history of ASD?", ["Yes", "No"])

# ---------------- BEHAVIOR ----------------
st.subheader("🧠 Behavioral Questions")

col3, col4 = st.columns(2)

with col3:
    q6 = st.radio("Notices small sounds?", ["Yes", "No"])
    q7 = st.radio("Focuses on whole picture?", ["Yes", "No"])
    q8 = st.radio("Multitasking ability?", ["Yes", "No"])

with col4:
    q9 = st.radio("Social situations easy?", ["Yes", "No"])
    q10 = st.radio("Prefers going out?", ["Yes", "No"])

# ---------------- INPUT DATA ----------------
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

# ---------------- ENCODING ----------------
ethnicity_ohe = pd.get_dummies(input_df['ethnicity'], prefix='ethnicity')
input_df = pd.concat([input_df.drop('ethnicity', axis=1), ethnicity_ohe], axis=1)

# ---------------- ALIGN FEATURES ----------------
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[model_features]
input_df = input_df.astype(float)

# ---------------- DOCTOR DATABASE ----------------
doctor_db = {
    "mumbai": [
        {"name":"Dr. Anjali Mehta","specialization":"Pediatric Neurologist","address":"Lilavati Hospital, Bandra West","phone":"+91 9876543210"},
        {"name":"Dr. Rajiv Sharma","specialization":"Child Psychiatrist","address":"Fortis Hospital, Mulund","phone":"+91 9123456780"},
        {"name":"Dr. Neha Kapoor","specialization":"ASD Specialist","address":"Kokilaben Hospital, Andheri West","phone":"+91 9988776655"}
    ],
    "delhi": [
        {"name":"Dr. Amit Verma","specialization":"Autism Specialist","address":"AIIMS, New Delhi","phone":"+91 9012345678"},
        {"name":"Dr. Ritu Singh","specialization":"Child Psychologist","address":"Max Hospital, Saket","phone":"+91 9871234560"},
        {"name":"Dr. Kunal Gupta","specialization":"Neurologist","address":"Apollo Hospital, Sarita Vihar","phone":"+91 9090909090"}
    ],
    "pune": [
        {"name":"Dr. Sneha Joshi","specialization":"Pediatrician","address":"Ruby Hall Clinic","phone":"+91 9898989898"},
        {"name":"Dr. Abhishek Kulkarni","specialization":"Psychiatrist","address":"Jehangir Hospital","phone":"+91 9765432101"},
        {"name":"Dr. Meera Patil","specialization":"ASD Therapist","address":"Deenanath Mangeshkar Hospital","phone":"+91 9345678901"}
    ]
}

# ---------------- PREDICT ----------------
if st.button("🚀 Predict"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[:,1][0]*100

    st.progress(int(prob))

    if pred == 1:
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

    # ---------------- DOCTOR DISPLAY ----------------
    if city:
        city_lower = city.lower()

        if city_lower in doctor_db:
            st.subheader("🏥 Recommended Doctors Near You")

            for doc in doctor_db[city_lower]:
                st.markdown(f"""
                <div style='background:#1e293b;padding:15px;border-radius:10px;margin-bottom:10px'>
                    <h4 style='color:#00E5FF;'>{doc['name']}</h4>
                    <p>🩺 {doc['specialization']}</p>
                    <p>📍 {doc['address']}</p>
                    <p>📞 {doc['phone']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No doctors found for this city. Try Mumbai, Delhi, or Pune.")

    st.info("⚠️ This is not a medical diagnosis. Consult a healthcare professional.")