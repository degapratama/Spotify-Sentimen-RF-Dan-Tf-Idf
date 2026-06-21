import streamlit as st
import numpy as np
import joblib
import os
import re

# ─────────────────────────────────────────────
# Konfigurasi halaman
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis Sentimen",
    page_icon="💬",
    layout="centered",
)

# ─────────────────────────────────────────────
# Tokenizer identitas (dibutuhkan untuk meload TF-IDF vectorizer yang sudah di-pickle)
# ─────────────────────────────────────────────
def identity_tokenizer(x):
    return x

# ─────────────────────────────────────────────
# Load model (di-cache agar tidak reload terus)
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    rf_path = "Model/rf_model.pkl"
    tfidf_path = "Model/tfidf_vectorizer.pkl"

    if not os.path.exists(rf_path) or not os.path.exists(tfidf_path):
        return None, None

    rf_model = joblib.load(rf_path)
    tfidf_vectorizer = joblib.load(tfidf_path)
    return rf_model, tfidf_vectorizer

# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────
def preprocess_text(text: str) -> list:
    """Lowercase + hapus karakter khusus + tokenisasi sederhana."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    return [t for t in tokens if t]


def get_tfidf_features(tokens: list, vectorizer) -> np.ndarray:
    """Ubah token menjadi vektor TF-IDF."""
    return vectorizer.transform([tokens]).toarray()[0]


LABEL_MAP = {0: "Negatif 😠", 1: "Positif 😊"}
COLOR_MAP = {0: "#FF4B4B", 1: "#21C55D"}

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("💬 Analisis Sentimen Review Aplikasi")
st.caption("Model: TF-IDF + Random Forest (SMOTE | K-Fold)")

st.divider()

# Load model
rf_model, tfidf_vectorizer = load_models()

if rf_model is None:
    st.error(
        "⚠️ **Model tidak ditemukan.**\n\n"
        "Pastikan file berikut sudah tersedia:\n"
        "- `Model/rf_model.pkl`\n"
        "- `Model/tfidf_vectorizer.pkl`"
    )
    st.stop()

# ── Input teks ──────────────────────────────
st.subheader("🔍 Prediksi Sentimen")
user_input = st.text_area(
    label="Masukkan teks review:",
    placeholder="Contoh: aplikasinya bagus banget, koleksi musik lengkap dan tidak ada iklan...",
    height=140,
)

predict_btn = st.button("Analisis Sentimen", type="primary", use_container_width=True)

if predict_btn:
    text = user_input.strip()
    if not text:
        st.warning("Teks review tidak boleh kosong.")
    else:
        tokens = preprocess_text(text)

        if not tokens:
            st.warning("Review tidak dapat diproses. Silakan masukkan teks yang valid!")
        else:
            vector = get_tfidf_features(tokens, tfidf_vectorizer).reshape(1, -1)

            prediction = rf_model.predict(vector)[0]
            probabilities = rf_model.predict_proba(vector)[0]

            label = LABEL_MAP[prediction]
            color = COLOR_MAP[prediction]
            conf = probabilities[prediction] * 100

            st.divider()
            st.subheader("📊 Hasil Analisis")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"<div style='text-align:center; padding:20px; border-radius:12px; "
                    f"background-color:{color}22; border:2px solid {color};'>"
                    f"<h2 style='color:{color}; margin:0'>{label}</h2>"
                    f"<p style='color:gray; margin:4px 0 0'>Sentimen Terdeteksi</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with col2:
                st.metric("Confidence", f"{conf:.1f}%")
                neg_conf = probabilities[0] * 100
                pos_conf = probabilities[1] * 100
                st.caption(f"Negatif: {neg_conf:.1f}% | Positif: {pos_conf:.1f}%")

            # Progress bar probabilitas
            st.markdown("**Distribusi Probabilitas:**")
            st.progress(int(pos_conf), text=f"Positif {pos_conf:.1f}%")

            # Tampilkan tokens
            with st.expander("🔎 Detail Tokenisasi"):
                st.write(f"**Jumlah token:** {len(tokens)}")
                st.write(tokens)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.caption("Model: TF-IDF + RandomForest (n_estimators=300) + SMOTE + 10-Fold CV")