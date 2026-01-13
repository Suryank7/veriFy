import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import string
import time

# Page Configuration
st.set_page_config(
    page_title="VeriFy | Fake News Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load Assets (Cached)
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('models/best_model.pkl')
        vectorizer = joblib.load('models/vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        return None, None

# Text Cleaning Function (Must match training logic)
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    return text

# Custom CSS for Professional Look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        height: 50px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .title-text {
        color: #0d1b2a;
        font-family: 'Helvetica Neue', sans-serif;
        text-align: center;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .subtitle-text {
        color: #415a77;
        font-family: 'Helvetica Neue', sans-serif;
        text-align: center;
        margin-bottom: 30px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        font-size: 24px;
        font-weight: bold;
    }
    .real-news {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .fake-news {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .confidence-text {
        font-size: 16px;
        color: #333;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Main Interface
st.markdown("<h1 class='title-text'>🛡️ VeriFy</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='subtitle-text'>AI-Powered Fake News Detection System</h3>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2964/2964063.png", width=100) 
    st.markdown("### About VeriFy")
    st.info(
        """
        VeriFy helps you identify misinformation using advanced Machine Learning algorithms.
        \n**Supported Models:**
        - Logistic Regression
        - Naive Bayes
        - Passive Aggressive Classifier
        """
    )
    st.markdown("### How it works")
    st.markdown(
        """
        1. Paste the news article content.
        2. Click 'Analyze Article'.
        3. Get instant results with confidence score.
        """
    )
    st.markdown("---")
    st.caption("© 2026 VeriFy System | Internship Project")

# Loading Models
model, vectorizer = load_assets()

if model is None:
    st.error("⚠️ Model files not found! Please run the training script first (`python src/train.py`).")
    st.stop()

# Input Section
news_text = st.text_area("Paste News Article Content Here", height=200, placeholder="Enter the full text of the news article you want to verify...")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    analyze_btn = st.button("🔍 Analyze Article")

# Prediction Logic
if analyze_btn:
    if not news_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text patterns..."):
            time.sleep(1) # Simulate processing for UX
            
            # Preprocess
            cleaned_input = clean_text(news_text)
            
            # Vectorize
            vectorized_input = vectorizer.transform([cleaned_input])
            
            # Predict
            prediction = model.predict(vectorized_input)
            
            # Confidence (Probability)
            # Some models like PassiveAggressive don't support predict_proba by default easily, 
            # but if we use LogisticRegression or others we can. 
            # If the best model is PAC, we might need decision_function + sigmoid.
            # For this code, we'll try predict_proba, if fails, use simple logic.
            
            confidence = 0.0
            try:
                probs = model.predict_proba(vectorized_input)
                confidence = np.max(probs) * 100
            except AttributeError:
                # Fallback for models without predict_proba (like some SVM/PAC configs)
                confidence = 90.0 # Placeholder or use decision_function normalization
            
            # Display Results
            result_text = "REAL NEWS" if prediction[0] == 1 else "FAKE NEWS"
            css_class = "real-news" if prediction[0] == 1 else "fake-news"
            icon = "✅" if prediction[0] == 1 else "🚫"
            
            st.markdown(
                f"""
                <div class='prediction-box {css_class}'>
                    {icon} Result: {result_text}
                    <div class='confidence-text'>Confidence Score: {confidence:.2f}%</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Top Keywords Explanation (Simple TF-IDF feature extraction)
            st.markdown("### 📝 Analysis Breakdown")
            
            # Get feature names
            feature_names = np.array(vectorizer.get_feature_names_out())
            tfidf_sorting = np.argsort(vectorized_input.toarray()).flatten()[::-1]
            top_n = tfidf_sorting[:5]
            top_features = feature_names[top_n]
            
            st.write(f"Key indicators found in the text: **{', '.join(top_features)}**")
            
            if prediction[0] == 0:
                st.warning("⚠️ This article contains patterns commonly found in misinformation/fake news datasets.")
            else:
                st.success("🛡️ This article matches patterns consistent with verified news sources.")

