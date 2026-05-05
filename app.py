# ============================================================
# AI-Powered Heart Disease Prediction System
# Professional Tabbed Edition
# Designed & Developed by Vinit Gill
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Heart Predictor | Vinit Gill",
    page_icon="⚡",
    layout="wide",
)

# ── Global CSS / Neobrutalism UI ───────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;700;900&display=swap');

/* Main Body */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #F8F9FA !important;
    font-family: 'Public Sans', sans-serif;
    color: #000;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 4px solid #000;
    padding-top: 2rem;
}

/* Sidebar Radio Buttons (Menu) */
[data-testid="stSidebarNav"] { display: none; } /* Hide default */

.nav-item {
    padding: 12px 20px;
    font-weight: 900;
    text-transform: uppercase;
    border: 3px solid #000;
    box-shadow: 4px 4px 0px 0px #000;
    margin-bottom: 15px;
    cursor: pointer;
    text-align: center;
    transition: 0.1s;
}

/* Custom Cards */
.neo-card {
    background: #FFF;
    border: 4px solid #000;
    box-shadow: 10px 10px 0px 0px #000;
    padding: 30px;
    margin-bottom: 30px;
}

.neo-header {
    background: #FFD600;
    border: 4px solid #000;
    box-shadow: 10px 10px 0px 0px #000;
    padding: 20px;
    text-align: center;
    margin-bottom: 40px;
}

/* Buttons */
.stButton > button {
    background: #FF4B4B !important;
    color: white !important;
    border: 4px solid #000 !important;
    box-shadow: 6px 6px 0px 0px #000 !important;
    font-weight: 900 !important;
    text-transform: uppercase !important;
    padding: 20px !important;
    width: 100% !important;
    border-radius: 0px !important;
}

/* Anti-Gravity Animations */
@keyframes floatUp {
  0%, 100% { transform: translateY(0px) rotate(-1deg); }
  50% { transform: translateY(-30px) rotate(1deg); }
}

@keyframes shake {
  0%, 100% { transform: translateX(0); }
  25% { transform: translateX(-5px); }
  75% { transform: translateX(5px); }
}

.anim-low {
    animation: floatUp 3s ease-in-out infinite;
    display: inline-block;
    font-size: 80px;
    border: 4px solid #000;
    padding: 20px;
    background: #00FF41;
    box-shadow: 10px 10px 0px 0px #000;
}

.anim-high {
    animation: shake 0.5s infinite;
    display: inline-block;
    font-size: 80px;
    border: 4px solid #000;
    padding: 20px;
    background: #FF0000;
    box-shadow: 10px 10px 0px 0px #000;
}

.footer-bar {
    margin-top: 50px;
    padding: 20px;
    border-top: 4px solid #000;
    text-align: center;
    font-weight: 900;
    background: #00E5FF;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────
ARTIFACTS = ["best_model.pkl", "scaler.pkl", "features.pkl"]

@st.cache_resource
def load_artifacts():
    try:
        model = pickle.load(open("best_model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        features = pickle.load(open("features.pkl", "rb"))
        return model, scaler, features
    except:
        return None, None, None

# ── Navigation ────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='text-align:center; font-weight:900;'>⚡ MENU</h2>", unsafe_allow_html=True)
    choice = st.radio("", ["🏠 PREDICTOR", "📊 ANALYTICS", "📖 HEALTH GUIDE", "👤 ABOUT"], label_visibility="collapsed")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='border:3px solid #000; padding:10px; background:#FFD600; box-shadow:4px 4px 0px 0px #000;'>
        <p style='font-weight:900; margin:0;'>VINIT GILL</p>
        <p style='font-size:0.8rem; margin:0;'>ML ENGINEER</p>
    </div>
    """, unsafe_allow_html=True)

# Load Data
model, scaler, features = load_artifacts()

# ── Main Content Logic ────────────────────────────────────

if choice == "🏠 PREDICTOR":
    st.markdown("<div class='neo-header'><h1>⚡ HEART DISEASE PREDICTOR</h1></div>", unsafe_allow_html=True)
    
    if not model:
        st.warning("Model files not found! Please run the training script or upload `best_model.pkl`.")
        st.stop()

    st.markdown("<div class='neo-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-weight:900;'>🧬 ENTER PATIENT VITALS</h3>", unsafe_allow_html=True)
    
    # Grid for inputs
    col1, col2, col3 = st.columns(3)
    user_vals = {}
    
    # Categorical Map
    cat_map = {
        "sex": {"label": "SEX", "opts": {0: "FEMALE", 1: "MALE"}},
        "cp": {"label": "CHEST PAIN", "opts": {0: "TYPICAL", 1: "ATYPICAL", 2: "NON-ANGINAL", 3: "ASYMPTOMATIC"}},
        "fbs": {"label": "BLOOD SUGAR > 120", "opts": {0: "NO", 1: "YES"}},
        "restecg": {"label": "RESTING ECG", "opts": {0: "NORMAL", 1: "ST-T", 2: "LVH"}},
        "exang": {"label": "EXERCISE ANGINA", "opts": {0: "NO", 1: "YES"}},
        "slope": {"label": "ST SLOPE", "opts": {0: "UP", 1: "FLAT", 2: "DOWN"}},
        "ca": {"label": "VESSELS (0-4)", "opts": {0: "0", 1: "1", 2: "2", 3: "3", 4: "4"}},
        "thal": {"label": "THALASSEMIA", "opts": {0: "NULL", 1: "FIXED", 2: "REVERSIBLE", 3: "TYPE 3"}},
    }

    for i, feat in enumerate(features):
        target_col = [col1, col2, col3][i % 3]
        with target_col:
            if feat in cat_map:
                m = cat_map[feat]
                user_vals[feat] = st.selectbox(m["label"], options=list(m["opts"].keys()), format_func=lambda k, m=m: m["opts"][k])
            elif feat == "age": user_vals[feat] = st.number_input("AGE", 1, 100, 45)
            elif feat == "trestbps": user_vals[feat] = st.slider("RESTING BP", 80, 200, 120)
            elif feat == "chol": user_vals[feat] = st.slider("CHOLESTEROL", 100, 500, 200)
            elif feat == "thalach": user_vals[feat] = st.slider("MAX HEART RATE", 60, 220, 150)
            else: user_vals[feat] = st.number_input(feat.upper(), value=1.0)

    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("🚀 ANALYZE RISK"):
        input_data = pd.DataFrame([[user_vals[f] for f in features]], columns=features)
        scaled_data = scaler.transform(input_data)
        
        with st.spinner("AI IS THINKING..."):
            time.sleep(1)
            prediction = model.predict(scaled_data)[0]
            prob = model.predict_proba(scaled_data)[0][1]

        # Results Display
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            if prediction == 1:
                st.markdown("<div class='anim-high'>💀</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='anim-low'>🎈</div>", unsafe_allow_html=True)
        
        with res_col2:
            color = "#FF0000" if prediction == 1 else "#00FF41"
            st.markdown(f"""
            <div style='border:5px solid #000; padding:20px; background:{color}22; box-shadow:8px 8px 0px 0px #000;'>
                <h1 style='font-weight:900; margin:0;'>{'HIGH RISK' if prediction == 1 else 'LOW RISK'}</h1>
                <p style='font-weight:700; font-size:1.5rem;'>RISK PROBABILITY: {prob*100:.1f}%</p>
                <p>{'Please consult a doctor immediately.' if prediction == 1 else 'Your stats look great! Keep it up.'}</p>
            </div>
            """, unsafe_allow_html=True)

elif choice == "📊 ANALYTICS":
    st.markdown("<div class='neo-header'><h1>📊 MODEL INSIGHTS</h1></div>", unsafe_allow_html=True)
    
    st.markdown("<div class='neo-card'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h3 style='font-weight:900;'>FEATURE IMPORTANCE</h3>", unsafe_allow_html=True)
        if os.path.exists("feature_importance.png"):
            st.image("feature_importance.png")
        else:
            st.info("Feature importance chart not found.")
    
    with col2:
        st.markdown("<h3 style='font-weight:900;'>ROC CURVE</h3>", unsafe_allow_html=True)
        if os.path.exists("roc_curves.png"):
            st.image("roc_curves.png")
        else:
            st.info("ROC Curve not found.")
    st.markdown("</div>", unsafe_allow_html=True)

elif choice == "📖 HEALTH GUIDE":
    st.markdown("<div class='neo-header'><h1>📖 HEALTH GUIDE</h1></div>", unsafe_allow_html=True)
    
    st.markdown("<div class='neo-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-weight:900;'>⚡ TOP TIPS FOR HEART HEALTH</h3>", unsafe_allow_html=True)
    st.write("1. **Balanced Diet:** Rich in fiber and low in saturated fats.")
    st.write("2. **Active Lifestyle:** Aim for at least 150 minutes of moderate exercise per week.")
    st.write("3. **Weight Management:** Maintain a healthy BMI (18.5 - 24.9).")
    st.write("4. **Sleep:** Ensure 7-9 hours of quality sleep every night.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='neo-card' style='background:#00E5FF11;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-weight:900;'>💡 GLOSSARY</h3>", unsafe_allow_html=True)
    st.markdown("""
    - **Trestbps:** Resting blood pressure (in mm Hg). Normal is ~120.
    - **Chol:** Serum cholesterol in mg/dl. High levels increase risk.
    - **Thalach:** Maximum heart rate achieved during stress test.
    - **Oldpeak:** ST depression induced by exercise relative to rest.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

elif choice == "👤 ABOUT":
    st.markdown("<div class='neo-header'><h1>👤 ABOUT THE SYSTEM</h1></div>", unsafe_allow_html=True)
    st.markdown("<div class='neo-card'>", unsafe_allow_html=True)
    st.write("This AI-powered Heart Disease Prediction System utilizes state-of-the-art machine learning algorithms to analyze clinical parameters and estimate cardiovascular risk.")
    st.markdown("<h4 style='font-weight:900;'>DEVELOPER</h4>", unsafe_allow_html=True)
    st.write("**Vinit Gill**")
    st.write("Specialist in Machine Learning & AI Application Design.")
    st.markdown("<p style='background:#FFD600; padding:10px; border:2px solid #000; font-weight:900; display:inline-block;'>GIVE A ⭐ ON GITHUB</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────
st.markdown("<div class='footer-bar'>DESIGNED & DEVELOPED BY VINIT GILL</div>", unsafe_allow_html=True)
