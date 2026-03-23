import streamlit as st
import pandas as pd
import pickle
import requests

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Autism Predictor", page_icon="🧠", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0A0F1E;
    color: #E8EAF0;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #0D1B2A 0%, #1B2A4A 50%, #0D1B2A 100%);
    border: 1px solid rgba(100, 180, 255, 0.15);
    border-radius: 20px;
    padding: 40px;
    text-align: center;
    margin-bottom: 30px;
    position: relative;
    overflow: hidden;
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center, rgba(0, 200, 255, 0.06) 0%, transparent 60%);
    pointer-events: none;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #64B4FF, #00E5C8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}

.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    font-size: 1.05rem;
    color: rgba(200, 215, 240, 0.65);
    margin-top: 10px;
    letter-spacing: 0.04em;
}

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #64B4FF;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 30px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(100, 180, 255, 0.2);
}

/* Input card panels */
.input-panel {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(100,180,255,0.1);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}

/* Doctor card */
.doctor-card {
    background: linear-gradient(135deg, rgba(13,27,42,0.9), rgba(27,42,74,0.7));
    border: 1px solid rgba(100, 180, 255, 0.2);
    border-radius: 16px;
    padding: 22px 26px;
    margin-bottom: 16px;
    transition: border-color 0.2s;
    position: relative;
    overflow: hidden;
}

.doctor-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #64B4FF, #00E5C8);
    border-radius: 16px 16px 0 0;
}

.doctor-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #64B4FF;
    margin: 0 0 6px 0;
}

.doctor-spec {
    font-size: 0.85rem;
    font-weight: 500;
    color: #00E5C8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 10px;
}

.doctor-detail {
    font-size: 0.92rem;
    color: rgba(200, 215, 240, 0.75);
    margin: 4px 0;
    display: flex;
    align-items: flex-start;
    gap: 6px;
}

.star-rating {
    color: #FFD700;
    font-size: 0.95rem;
    margin: 8px 0;
}

/* Result cards */
.result-positive {
    background: linear-gradient(135deg, rgba(255,75,75,0.15), rgba(200,30,30,0.08));
    border: 1px solid rgba(255,75,75,0.35);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    margin: 20px 0;
}

.result-negative {
    background: linear-gradient(135deg, rgba(0,201,167,0.15), rgba(0,140,120,0.08));
    border: 1px solid rgba(0,201,167,0.35);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    margin: 20px 0;
}

.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    margin: 0;
}

.result-prob {
    font-family: 'DM Sans', sans-serif;
    font-size: 2.4rem;
    font-weight: 300;
    margin: 8px 0 0 0;
}

/* Predict button */
.stButton > button {
    background: linear-gradient(90deg, #64B4FF, #00E5C8) !important;
    color: #0A0F1E !important;
    border: none !important;
    border-radius: 12px !important;
    height: 3.2em !important;
    font-size: 1.05rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover {
    opacity: 0.88 !important;
}

/* Streamlit default overrides */
.stSelectbox label, .stNumberInput label,
.stRadio label, .stTextInput label {
    color: rgba(200, 215, 240, 0.75) !important;
    font-size: 0.9rem !important;
    font-weight: 400 !important;
}

[data-testid="stRadio"] p {
    color: rgba(200, 215, 240, 0.85) !important;
}

.stProgress > div > div {
    background: linear-gradient(90deg, #64B4FF, #00E5C8) !important;
}

.disclaimer {
    background: rgba(255,200,0,0.07);
    border: 1px solid rgba(255,200,0,0.25);
    border-radius: 10px;
    padding: 14px 18px;
    color: rgba(255,220,100,0.85);
    font-size: 0.88rem;
    margin-top: 24px;
    text-align: center;
}

div[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(100,180,255,0.1) !important;
    border-radius: 12px !important;
}

.api-key-note {
    background: rgba(100,180,255,0.06);
    border: 1px dashed rgba(100,180,255,0.3);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: rgba(180,210,255,0.8);
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HERO HEADER ----------------
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">🧠 ASD Screening Tool</p>
    <p class="hero-sub">AI-powered Autism Spectrum Disorder screening · Find specialists near you</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# GOOGLE PLACES DOCTOR FINDER
# ============================================================

def get_stars(rating):
    if not rating:
        return "No rating"
    full = int(rating)
    half = 1 if (rating - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + ("½" if half else "") + "☆" * empty + f"  {rating}/5"


def fetch_doctors_google(city: str, api_key: str):
    """
    Search for autism/child neurology specialists using multiple fallback queries.
    Removes the restrictive 'type' filter which causes empty results for niche searches.
    """
    # Ordered from most specific → broadest fallback
    queries = [
        f"autism specialist {city}",
        f"child neurologist autism {city}",
        f"child psychiatrist {city}",
        f"pediatric neurologist {city}",
        f"developmental pediatrician {city}",
    ]

    text_search_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    raw_results = []
    seen_ids = set()

    for query in queries:
        if len(raw_results) >= 5:
            break
        params = {
            "query": query,
            "key": api_key,
            "language": "en",
            # NO 'type' filter — it clashes with text search and kills results
        }
        try:
            resp = requests.get(text_search_url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            # Surface API errors (INVALID_REQUEST, REQUEST_DENIED, etc.)
            status = data.get("status", "")
            if status not in ("OK", "ZERO_RESULTS"):
                return None, f"Google API error: {status} — {data.get('error_message', '')}"

            for place in data.get("results", []):
                pid = place.get("place_id")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    raw_results.append(place)
                if len(raw_results) >= 5:
                    break
        except Exception as e:
            return None, str(e)

    if not raw_results:
        return [], None

    # Fetch detailed info for each place
    doctors = []
    details_url = "https://maps.googleapis.com/maps/api/place/details/json"
    for place in raw_results[:5]:
        place_id = place.get("place_id")
        det_params = {
            "place_id": place_id,
            "fields": "name,formatted_address,formatted_phone_number,rating,user_ratings_total,editorial_summary,opening_hours",
            "key": api_key,
        }
        try:
            det_resp = requests.get(details_url, params=det_params, timeout=10)
            det_resp.raise_for_status()
            detail = det_resp.json().get("result", {})
        except Exception:
            detail = {}

        doctors.append({
            "name":          detail.get("name") or place.get("name", "Unknown"),
            "address":       detail.get("formatted_address") or place.get("formatted_address", "N/A"),
            "phone":         detail.get("formatted_phone_number") or "Not listed",
            "rating":        detail.get("rating") or place.get("rating"),
            "total_ratings": detail.get("user_ratings_total") or place.get("user_ratings_total", 0),
            "summary":       (detail.get("editorial_summary") or {}).get("overview", ""),
            "open_now":      (detail.get("opening_hours") or {}).get("open_now"),
        })

        if len(doctors) == 3:
            break

    return doctors, None


# ============================================================
# LOAD MODEL
# ============================================================
try:
    model = pickle.load(open("autism_prediction_model.pkl", "rb"))
    model_features = list(model.feature_names_in_)
    model_loaded = True
except Exception:
    model_loaded = False
    model_features = []

ALL_ETHNICITIES = ["White-European", "Black", "Latino", "Asian", "Middle Eastern", "Others"]

# ============================================================
# SIDEBAR — API KEY
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    api_key = st.text_input(
        "Google Places API Key",
        type="password",
        placeholder="Paste your API key here",
        help="Get one free at console.cloud.google.com → Enable Places API"
    )
    st.markdown("""
    <div style='font-size:0.78rem;color:rgba(180,210,255,0.65);margin-top:8px;line-height:1.6'>
    <b>How to get a free key:</b><br>
    1. Go to console.cloud.google.com<br>
    2. New Project → Enable <b>Places API</b><br>
    3. APIs & Services → Credentials → Create API Key<br>
    4. $200 free credit/month (more than enough)
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    if not model_loaded:
        st.warning("⚠️ Model file not found. Place `autism_prediction_model.pkl` in the app directory.")

# ============================================================
# MAIN LAYOUT — two columns
# ============================================================
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    # ---- LOCATION ----
    st.markdown('<p class="section-header">📍 Location</p>', unsafe_allow_html=True)

    INDIAN_CITIES = [
        "Agartala","Agra","Ahmedabad","Aizawl","Ajmer","Aligarh","Allahabad",
        "Amravati","Amritsar","Aurangabad","Bangalore","Bareilly","Bhopal",
        "Bhubaneswar","Chandigarh","Chennai","Coimbatore","Cuttack","Dehradun",
        "Delhi","Dhanbad","Durgapur","Erode","Faridabad","Ghaziabad","Gorakhpur",
        "Guwahati","Gwalior","Howrah","Hubli","Hyderabad","Imphal","Indore",
        "Itanagar","Jabalpur","Jaipur","Jalandhar","Jammu","Jamnagar","Jamshedpur",
        "Jodhpur","Kakinada","Kanpur","Kochi","Kohima","Kolkata","Kota",
        "Kozhikode","Lucknow","Ludhiana","Madurai","Mangalore","Meerut","Mumbai",
        "Mysore","Nagpur","Nashik","Navi Mumbai","Noida","Patna","Pondicherry",
        "Pune","Raipur","Rajkot","Ranchi","Salem","Shillong","Shimla","Siliguri",
        "Srinagar","Surat","Thane","Thiruvananthapuram","Tiruchirappalli","Udaipur",
        "Vadodara","Varanasi","Vijayawada","Visakhapatnam","Warangal"
    ]

    city = st.selectbox("Select your city", sorted(INDIAN_CITIES))

    # ---- PERSONAL DETAILS ----
    st.markdown('<p class="section-header">👤 Personal Details</p>', unsafe_allow_html=True)

    age = st.number_input("Age", min_value=1, max_value=100, value=25)
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    ethnicity = st.selectbox("Ethnicity", ALL_ETHNICITIES)
    jaundice = st.radio("Jaundice at birth?", ["Yes", "No"], horizontal=True)
    family_history = st.radio("Family history of ASD?", ["Yes", "No"], horizontal=True)

with right_col:
    # ---- BEHAVIORAL ----
    st.markdown('<p class="section-header">🧠 Behavioral Screening (A6–A10)</p>', unsafe_allow_html=True)

    q6  = st.radio("Notices small sounds others miss?", ["Yes", "No"], horizontal=True)
    q7  = st.radio("Focuses on the whole picture rather than details?", ["Yes", "No"], horizontal=True)
    q8  = st.radio("Can multitask easily?", ["Yes", "No"], horizontal=True)
    q9  = st.radio("Finds social situations easy?", ["Yes", "No"], horizontal=True)
    q10 = st.radio("Prefers going out and meeting people?", ["Yes", "No"], horizontal=True)

    st.markdown("")
    predict_btn = st.button("🚀 Run Screening")

# ============================================================
# PREDICTION
# ============================================================
if predict_btn:
    if not model_loaded:
        st.error("Model not loaded. Please add `autism_prediction_model.pkl` to the app directory.")
    else:
        data = {
            "age": age,
            "gender": 1 if gender == "Male" else 0,
            "jaundice": 1 if jaundice == "Yes" else 0,
            "family_history": 1 if family_history == "Yes" else 0,
            "ethnicity": ethnicity,
            "A1_Score": 0, "A2_Score": 0, "A3_Score": 0, "A4_Score": 0, "A5_Score": 0,
            "A6_Score":  1 if q6 == "Yes" else 0,
            "A7_Score":  1 if q7 == "Yes" else 0,
            "A8_Score":  1 if q8 == "Yes" else 0,
            "A9_Score":  1 if q9 == "Yes" else 0,
            "A10_Score": 1 if q10 == "Yes" else 0,
        }

        input_df = pd.DataFrame([data])
        ethnicity_ohe = pd.get_dummies(input_df["ethnicity"], prefix="ethnicity")
        input_df = pd.concat([input_df.drop("ethnicity", axis=1), ethnicity_ohe], axis=1)
        for col in model_features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model_features].astype(float)

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[:, 1][0] * 100

        # Result display
        result_col, _ = st.columns([2, 1])
        with result_col:
            st.progress(int(prob))

            if pred == 1:
                st.markdown(f"""
                <div class="result-positive">
                    <p class="result-title" style="color:#FF6B6B">⚠️ High Likelihood of ASD</p>
                    <p class="result-prob" style="color:#FF9999">{prob:.1f}%<span style="font-size:1rem;color:rgba(255,150,150,0.6)"> probability</span></p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                **Indicators detected:**
                - Elevated behavioral screening score  
                - Possible difficulty in social interaction  
                - **Recommendation:** Please consult a specialist below
                """)
            else:
                st.markdown(f"""
                <div class="result-negative">
                    <p class="result-title" style="color:#00E5C8">✅ Low Likelihood of ASD</p>
                    <p class="result-prob" style="color:#80FFE8">{100-prob:.1f}%<span style="font-size:1rem;color:rgba(100,255,220,0.6)"> confidence</span></p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                **Screening notes:**
                - Low behavioral indicators detected  
                - Normal interaction patterns observed  
                - No immediate concern — continue monitoring if needed
                """)

        # ============================================================
        # DOCTOR FINDER
        # ============================================================
        st.markdown('<p class="section-header">🏥 Autism Specialists Near You</p>', unsafe_allow_html=True)

        if not api_key:
            st.markdown("""
            <div class="api-key-note">
            🔑 <b>Add your Google Places API key</b> in the sidebar to find real doctors near you.<br>
            It's free to set up and takes about 5 minutes.
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner(f"Finding top autism specialists in {city}..."):
                doctors, error = fetch_doctors_google(city, api_key)

            if error:
                st.error(f"API Error: {error}")
                st.info("Fixes: Check API key is correct, Places API is enabled in Google Cloud, and billing is active (free tier works).")
            elif not doctors:
                st.warning(f"No specialists found for {city}. Try a nearby metro like Mumbai, Delhi, Pune or Bangalore.")
            else:
                for i, doc in enumerate(doctors):
                    rating_display = get_stars(doc["rating"]) if doc["rating"] else "No rating yet"
                    open_status = ""
                    if doc["open_now"] is True:
                        open_status = "<span style='color:#00E5C8;font-size:0.8rem;'>● Open now</span>"
                    elif doc["open_now"] is False:
                        open_status = "<span style='color:#FF6B6B;font-size:0.8rem;'>● Closed now</span>"

                    summary_html = f"<p class='doctor-detail'>📝 {doc['summary']}</p>" if doc["summary"] else ""
                    total_html = f" <span style='color:rgba(200,215,240,0.45);font-size:0.8rem'>({doc['total_ratings']} reviews)</span>" if doc["total_ratings"] else ""

                    st.markdown(f"""
                    <div class="doctor-card">
                        <p class="doctor-name">#{i+1} &nbsp; {doc['name']}</p>
                        <p class="doctor-spec">Autism · Neurodevelopmental Specialist</p>
                        <p class="star-rating">{rating_display}{total_html} &nbsp; {open_status}</p>
                        <p class="doctor-detail">📍 {doc['address']}</p>
                        <p class="doctor-detail">📞 {doc['phone']}</p>
                        {summary_html}
                    </div>
                    """, unsafe_allow_html=True)

        # Disclaimer
        st.markdown("""
        <div class="disclaimer">
        ⚠️ This tool is for <b>educational and screening purposes only</b> — it is not a clinical diagnosis.
        Always consult a licensed medical professional for a proper evaluation.
        </div>
        """, unsafe_allow_html=True)
