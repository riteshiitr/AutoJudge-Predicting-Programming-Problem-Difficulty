import streamlit as st
import joblib
import re
import numpy as np
import nltk
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack

# --- Resource Initialization ---
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        pass

download_nltk_data()

# --- Page Configuration ---
st.set_page_config(page_title="AutoJudge AI", page_icon="⚖️", layout="centered")

# --- Custom Styling ---
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

/* Glass card with neon border */
.glass-card {
    background: rgba(20, 20, 30, 0.85);
    border-radius: 22px;
    padding: 35px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 0 25px rgba(0, 255, 255, 0.08);
}

/* Title with glow */
.main-title {
    font-size: 3.6rem;
    font-weight: 900;
    text-align: center;
    color: #e0f7ff;
    text-shadow: 0 0 15px rgba(75,175,255,0.8);
}

/* Textareas */
.stTextArea textarea {
    background: #0e1624 !important;
    color: #e6f1ff !important;
    border-radius: 14px !important;
    border: 1px solid rgba(75,175,255,0.3) !important;
}

/* Button */
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg,#36d1dc,#5b86e5) !important;
    color: #fff !important;
    font-weight: 700 !important;
    border-radius: 14px !important;
    box-shadow: 0 0 15px rgba(91,134,229,0.6);
}

/* Prediction card */
.prediction-container {
    text-align: center;
    padding: 45px;
    border-radius: 30px;
    background: radial-gradient(circle at top, rgba(255,255,255,0.12), rgba(0,0,0,0.9));
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 0 40px rgba(0,0,0,0.6);
}
</style>
""", unsafe_allow_html=True)


# --- Load Models ---
@st.cache_resource
def load_assets():
    return (
        joblib.load("Trained_Models/difficulty_classifier.pkl"),
        joblib.load("Trained_Models/difficulty_regressor.pkl"),
        joblib.load("Trained_Models/tfidf_vectorizer.pkl")
    )

clf, reg, tfidf = load_assets()

# --- Helpers ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s\^\*]', ' ', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    lem = WordNetLemmatizer()
    return " ".join(lem.lemmatize(w) for w in words if w not in stop_words)

def get_complexity_signal(text):
    matches = re.findall(r'(?:10(?:\^|\*\*|e)|1000)\s*(\d+)', text)
    return max([int(x) for x in matches], default=0)

# --- UI ---
st.markdown('<h1 class="main-title">AutoJudge</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#888;">AI-based Competitive Programming Difficulty Predictor</p>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        prob_desc = st.text_area("Problem Description", height=220)

    with col2:
        input_desc = st.text_area("Input Description", height=220)

    with col3:
        output_desc = st.text_area("Output Description", height=220)


    predict_btn = st.button("🔍 Predict Difficulty")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Prediction ---
if predict_btn and prob_desc and input_desc and output_desc:
    combined_text = f"{prob_desc} {input_desc} {output_desc}"
    cleaned = clean_text(combined_text)
    tfidf_feat = tfidf.transform([cleaned])

    text_len = len(cleaned)
    math_symbols = len(re.findall(r'[+\-*/%=<>!^]', combined_text))
    max_constraint = get_complexity_signal(combined_text)

    keywords = ['graph','dp','tree','segment','dijkstra','shortest','query','array',
                'string','recursion','complexity','optimal','greedy','bitwise','modulo',
                'combinatorics','probability','geometry']
    kw_feats = [1 if k in cleaned else 0 for k in keywords]

    X = hstack([tfidf_feat, np.array([[text_len, math_symbols, max_constraint] + kw_feats])])

    class_idx = clf.predict(X)[0]
    score = float(reg.predict(X)[0])

    labels = {0:"Easy",1:"Medium",2:"Hard"}
    color = "#4bffab" if class_idx==0 else "#ff9b4b" if class_idx==1 else "#ff4b4b"

    st.markdown(f"""
<div class="prediction-container">
    <p style="color:#9aa4b2; letter-spacing:2px;">AI DIFFICULTY ASSESSMENT</p>
    <h1 style="color:{color}; font-size:4rem; margin:10px 0;">
        {labels[class_idx]}
    </h1>
    <hr style="width:60%; border:1px solid rgba(255,255,255,0.2); margin:20px auto;">
    <p style="color:#9aa4b2;">Difficulty Score</p>
    <h2 style="color:#36d1dc; font-size:2.4rem;">
        {score:.2f} / 10
    </h2>
</div>
""", unsafe_allow_html=True)


    st.progress(min(score/10, 1.0))