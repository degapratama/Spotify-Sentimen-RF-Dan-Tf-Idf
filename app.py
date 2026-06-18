import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

# Set page configuration
st.set_page_config(
    page_title="Sentiment Aplikasi Spotify",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    [data-testid="stSidebar"] {
        background-color: #0d0d0d;
    }
    .main {
        background-color: #1a1a1a;
    }
    
    /* Input area styling */
    textarea {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
        border: 1px solid #444 !important;
        border-radius: 8px !important;
    }
    
    /* Title styling */
    .app-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .header-top {
        font-size: 0.85rem;
        color: #888888;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }
    
    .header-title {
        font-size: 2rem;
        font-weight: bold;
        color: #ffffff;
        margin: 0;
        padding: 0;
    }
    
    .header-subtitle {
        font-size: 0.9rem;
        color: #888888;
        margin-top: 0.3rem;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #00d84d !important;
        color: #000000 !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.5rem !important;
        font-size: 0.9rem !important;
        border: none !important;
    }
    
    .stButton > button:hover {
        background-color: #00cc44 !important;
        transform: scale(1.02);
    }
    
    /* Input label */
    .input-label {
        color: #888888;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 0.8rem;
        font-weight: 600;
    }
    
    /* Result area styling */
    .result-container {
        background-color: #2d2d2d;
        border: 1px solid #444;
        border-radius: 8px;
        padding: 2rem;
        margin-top: 2rem;
    }
    
    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #444;
    }
    
    .result-title {
        font-size: 0.95rem;
        color: #ffffff;
        font-weight: 600;
    }
    
    .sentiment-tag {
        display: inline-block;
        background-color: #00d84d;
        color: #000000;
        padding: 0.25rem 0.8rem;
        border-radius: 20px;
        font-size: 0.65rem;
        font-weight: bold;
        text-transform: uppercase;
    }
    
    .sentiment-result {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 2rem;
    }
    
    .sentiment-left {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .sentiment-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 50px;
        height: 50px;
        border: 2px solid #00d84d;
        border-radius: 50%;
        font-size: 1.8rem;
    }
    
    .sentiment-label {
        font-size: 1.3rem;
        color: #00d84d;
        font-weight: bold;
    }
    
    .sentiment-negative {
        color: #ff4444;
    }
    
    .sentiment-right {
        text-align: right;
    }
    
    .accuracy-label {
        font-size: 0.75rem;
        color: #888888;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }
    
    .confidence-score {
        font-size: 2.8rem;
        font-weight: bold;
        color: #00d84d;
    }
    
    .progress-bar {
        width: 100%;
        height: 3px;
        background-color: #444;
        border-radius: 2px;
        margin-top: 1rem;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        background-color: #00d84d;
        border-radius: 2px;
    }
    </style>
""", unsafe_allow_html=True)

# Define identity tokenizer (required for loading pickled TF-IDF vectorizer)
def identity_tokenizer(x):
    return x

# Load models
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load('Model/rf_model.pkl')
        tfidf_vectorizer = joblib.load('Model/tfidf_vectorizer.pkl')
        return rf_model, tfidf_vectorizer
    except FileNotFoundError as e:
        st.error(f"❌ Error: Model tidak ditemukan - {e}")
        st.stop()

# Preprocess text function
def preprocess_text(text):
    """Preprocess text: lowercase and tokenize"""
    text = text.lower()
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    # Remove empty tokens
    tokens = [token for token in tokens if token]
    return tokens

# Get TF-IDF features
def get_tfidf_features(tokens, vectorizer):
    """Convert tokens to TF-IDF features"""
    return vectorizer.transform([tokens]).toarray()[0]

# Main app
def main():
    # Header
    st.markdown("""
        <div class="app-header">
            <div class="header-top">Sentiment Aplikasi</div>
            <div class="header-title">Analisis Sentimen TF-IDF</div>
            <div class="header-subtitle">Klasifikasi Ulasan Aplikasi</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Load models
    rf_model, tfidf_vectorizer = load_models()
    
    # Input section
    st.markdown('<div class="input-label">INPUT ULASAN PENGGUNA</div>', unsafe_allow_html=True)
    user_review = st.text_area(
        "Masukkan ulasan:",
        placeholder="Deskripsikan Aplikasi atau Bangnya",
        height=110,
        label_visibility="collapsed"
    )
    
    # Classify button - positioned to right
    col1, col2 = st.columns([3, 1])
    with col2:
        classify_button = st.button("🎯 Klasifikasi")
    
    # Process when button is clicked
    if classify_button:
        if not user_review.strip():
            st.warning("⚠️ Silakan masukkan ulasan terlebih dahulu!")
        else:
            # Preprocess
            tokens = preprocess_text(user_review)
            
            if not tokens:
                st.warning("⚠️ Review tidak dapat diproses. Silakan masukkan teks yang valid!")
            else:
                # Get TF-IDF features
                vector = get_tfidf_features(tokens, tfidf_vectorizer).reshape(1, -1)
                
                # Get probabilities
                probabilities = rf_model.predict_proba(vector)[0]
                prediction = rf_model.predict(vector)[0]
                
                # Map labels
                label_map = {0: 'negative', 1: 'positive'}
                sentiment = label_map[prediction]
                confidence = probabilities[prediction] * 100
                
                # Display result
                st.markdown("""
                    <div class="result-container">
                        <div class="result-header">
                            <span class="result-title">Hasil Analisis</span>
                            <span class="sentiment-tag">SENTIMEN UTAMA</span>
                        </div>
                """, unsafe_allow_html=True)
                
                # Sentiment result with icon
                if sentiment == 'positive':
                    icon = "✓"
                    color = "#00d84d"
                    sentiment_text = "Positif"
                else:
                    icon = "✗"
                    color = "#ff4444"
                    sentiment_text = "Negatif"
                
                st.markdown(f"""
                    <div class="sentiment-result">
                        <div class="sentiment-left">
                            <div class="sentiment-icon" style="border-color: {color}; color: {color};">
                                {icon}
                            </div>
                            <div class="sentiment-label" style="color: {color};">
                                {sentiment_text}
                            </div>
                        </div>
                        <div class="sentiment-right">
                            <div class="accuracy-label">Tingkat Akurasi Model</div>
                            <div class="confidence-score">{confidence:.1f}%</div>
                        </div>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {confidence}%;"></div>
                    </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Additional details (expandable)
                with st.expander("📊 Detail Analisis Lengkap"):
                    st.write(f"**Review:** {user_review}")
                    st.write(f"**Tokens:** {', '.join(tokens)}")
                    st.write(f"**Confidence Positif:** {probabilities[1]*100:.2f}%")
                    st.write(f"**Confidence Negatif:** {probabilities[0]*100:.2f}%")

if __name__ == "__main__":
    main()
