"""
app.py — Skin Disease Detection Application
Built with Streamlit + TensorFlow/Keras (MobileNetV2 Transfer Learning)

Author  : [Your Name]
College : [Your College Name]
Project : Final Year B.Tech CSE Project
"""

import os
import io
import numpy as np
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DermaScan AI — Disease Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* Hero header */
    .hero-header {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
    }
    .hero-header h1 {
        font-family: 'Poppins', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero-header p {
        color: #a0aec0;
        font-size: 1.05rem;
        margin-top: 0;
    }

    /* Cards */
    .card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }

    /* Result box */
    .result-box {
        background: linear-gradient(135deg, rgba(167,139,250,0.15), rgba(96,165,250,0.15));
        border: 1px solid rgba(167,139,250,0.4);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-disease {
        font-family: 'Poppins', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #a78bfa;
    }
    .result-confidence {
        font-size: 1.1rem;
        color: #60a5fa;
        margin-top: 0.3rem;
    }

    /* Severity badge */
    .severity-mild     { background: #22c55e33; color: #4ade80; border: 1px solid #4ade8055; }
    .severity-moderate { background: #f59e0b33; color: #fbbf24; border: 1px solid #fbbf2455; }
    .severity-severe   { background: #ef444433; color: #f87171; border: 1px solid #f8717155; }
    .severity-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }

    /* Remedy list */
    .remedy-item {
        background: rgba(255,255,255,0.04);
        border-left: 3px solid #a78bfa;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        border-radius: 0 8px 8px 0;
        color: #e2e8f0;
        font-size: 0.92rem;
    }

    /* Symptom tag */
    .symptom-tag {
        display: inline-block;
        background: rgba(96,165,250,0.15);
        border: 1px solid rgba(96,165,250,0.3);
        color: #93c5fd;
        padding: 0.25rem 0.75rem;
        border-radius: 50px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }

    /* Progress bar label */
    .prob-label {
        color: #cbd5e1;
        font-size: 0.85rem;
        margin-bottom: 0.1rem;
    }

    /* Warning box */
    .warning-box {
        background: rgba(239,68,68,0.1);
        border: 1px solid rgba(239,68,68,0.4);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: #fca5a5;
        font-size: 0.9rem;
        margin-top: 1rem;
    }

    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.8) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #2563eb);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(124,58,237,0.4);
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(167,139,250,0.4) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }

    /* Section headers */
    .section-title {
        font-family: 'Poppins', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        border-bottom: 2px solid rgba(167,139,250,0.3);
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }

    div[data-testid="stMarkdownContainer"] p { color: #cbd5e1; }
    h3 { color: #e2e8f0 !important; }
    .stSelectbox label { color: #a0aec0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# IMPORTS (after page config)
# ─────────────────────────────────────────────────────────────
from utils.preprocess import preprocess_image, validate_image
from utils.disease_info import DISEASE_INFO, get_disease_details

# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────
MODEL_PATH      = "model/skin_disease_model.h5"
CLASS_NAMES_PATH = "model/class_names.txt"

@st.cache_resource
def load_model_and_classes():
    """Load trained Keras model and class names (cached)."""
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(CLASS_NAMES_PATH, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        return model, class_names, True
    except Exception as e:
        return None, list(DISEASE_INFO.keys()), False


def predict(model, class_names: list, image: Image.Image) -> list[tuple]:
    """
    Run inference and return sorted (disease, confidence) list.
    Falls back to a simulated demo if model is not loaded.
    """
    processed = preprocess_image(image)

    if model is not None:
        predictions = model.predict(processed, verbose=0)[0]
    else:
        # Demo mode: random probabilities for showcase
        np.random.seed(hash(str(image.size)) % 999)
        raw = np.random.dirichlet(np.ones(len(class_names)) * 0.5)
        predictions = raw

    results = sorted(
        zip(class_names, predictions.tolist()),
        key=lambda x: x[1],
        reverse=True
    )
    return results


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 DermaScan AI")
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "This application uses **Deep Learning (MobileNetV2)** to analyze skin "
        "images and detect potential dermatological conditions."
    )
    st.markdown("---")
    st.markdown("### 🩺 Detectable Conditions")
    for disease, info in DISEASE_INFO.items():
        st.markdown(
            f"<span style='display:inline-block;width:10px;height:10px;"
            f"border-radius:50%;background:{info['color']};margin-right:8px'></span>"
            f"<span style='color:#cbd5e1;font-size:0.9rem'>{disease}</span>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.markdown(
        "<span style='color:#fca5a5;font-size:0.82rem'>"
        "This tool is for educational purposes only. "
        "Always consult a certified dermatologist for medical advice."
        "</span>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown(
        "<span style='color:#64748b;font-size:0.75rem'>B.Tech CSE Final Year Project</span>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🔬 DermaScan AI</h1>
    <p>Upload a skin image for instant AI-powered disease analysis & recommendations</p>
</div>
""", unsafe_allow_html=True)

# Load model
model, class_names, model_loaded = load_model_and_classes()

if not model_loaded:
    st.markdown("""
    <div style='background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.4);
    border-radius:12px;padding:1rem 1.5rem;color:#fbbf24;font-size:0.9rem;margin-bottom:1rem'>
    ⚠️ <b>Demo Mode</b>: No trained model found. Running with simulated predictions for UI preview.
    Train the model using <code>python model/train_model.py</code> to enable real predictions.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style='background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.3);
    border-radius:12px;padding:0.6rem 1.2rem;color:#4ade80;font-size:0.85rem;margin-bottom:1rem'>
    ✅ Model loaded successfully — Ready for predictions
    </div>
    """, unsafe_allow_html=True)

# ── Upload Section ────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="section-title">📤 Upload Skin Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag & drop or click to browse",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(io.BytesIO(uploaded_file.read()))
        is_valid, msg = validate_image(image)

        if not is_valid:
            st.error(msg)
        else:
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.markdown(
                f"<div style='color:#64748b;font-size:0.8rem;text-align:center'>"
                f"Size: {image.size[0]}×{image.size[1]} px | Mode: {image.mode}</div>",
                unsafe_allow_html=True
            )

            analyze_btn = st.button("🔍 Analyze Image", type="primary")

    else:
        st.markdown("""
        <div style='text-align:center;padding:3rem 1rem;color:#4a5568'>
            <div style='font-size:4rem'>🖼️</div>
            <div style='color:#64748b'>Upload an image to get started</div>
            <div style='color:#475569;font-size:0.8rem;margin-top:0.5rem'>
                Supported: JPG, PNG, BMP, WebP
            </div>
        </div>
        """, unsafe_allow_html=True)
        analyze_btn = False


# ── Results Section ───────────────────────────────────────────
with col2:
    st.markdown('<div class="section-title">📊 Analysis Results</div>', unsafe_allow_html=True)

    if uploaded_file and is_valid and analyze_btn:
        with st.spinner("Analyzing image with AI model..."):
            results = predict(model, class_names, image)

        top_disease, top_conf = results[0]
        disease_info = get_disease_details(top_disease)

        # ── Top Prediction ──────────────────────────────────────
        st.markdown(f"""
        <div class="result-box">
            <div style='color:#a0aec0;font-size:0.85rem;text-transform:uppercase;letter-spacing:1px'>
                Top Prediction
            </div>
            <div class="result-disease">{top_disease}</div>
            <div class="result-confidence">Confidence: {top_conf * 100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Severity badge
        if disease_info:
            sev = disease_info["severity"].lower()
            if "severe" in sev:
                cls = "severity-severe"
            elif "moderate" in sev:
                cls = "severity-moderate"
            else:
                cls = "severity-mild"

            st.markdown(
                f'<div style="text-align:center;margin:0.5rem 0">'
                f'<span class="severity-badge {cls}">⚡ {disease_info["severity"]}</span></div>',
                unsafe_allow_html=True
            )

        # ── Probability Bars ────────────────────────────────────
        st.markdown('<div style="margin-top:1rem">', unsafe_allow_html=True)
        st.markdown("**Top Predictions**")
        for disease, prob in results[:5]:
            pct = round(prob * 100, 1)
            color = DISEASE_INFO.get(disease, {}).get("color", "#a78bfa")
            st.markdown(f'<div class="prob-label">{disease} — {pct}%</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="background:rgba(255,255,255,0.08);border-radius:6px;height:10px;overflow:hidden">'
                f'<div style="width:{pct}%;height:100%;background:{color};border-radius:6px;'
                f'transition:width 1s ease"></div></div>',
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='text-align:center;padding:3rem 1rem'>
            <div style='font-size:3rem'>🩺</div>
            <div style='color:#64748b'>Results will appear here after analysis</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DETAILED DISEASE INFO (below main columns)
# ─────────────────────────────────────────────────────────────
if uploaded_file and is_valid and analyze_btn:
    top_disease = results[0][0]
    disease_info = get_disease_details(top_disease)

    if disease_info:
        st.markdown("---")
        st.markdown(f"## 📋 About: {top_disease}")

        c1, c2, c3 = st.columns(3, gap="medium")

        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📝 Description</div>', unsafe_allow_html=True)
            st.markdown(
                f'<p style="color:#cbd5e1;font-size:0.92rem;line-height:1.6">'
                f'{disease_info["description"]}</p>',
                unsafe_allow_html=True
            )

            st.markdown('<div class="section-title" style="margin-top:1rem">🔍 Symptoms</div>', unsafe_allow_html=True)
            tags = "".join([f'<span class="symptom-tag">{s}</span>' for s in disease_info["symptoms"]])
            st.markdown(tags, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">⚙️ Common Causes</div>', unsafe_allow_html=True)
            for cause in disease_info["causes"]:
                st.markdown(
                    f'<div class="remedy-item" style="border-left-color:#60a5fa">🔹 {cause}</div>',
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)

        with c3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">💊 Remedies & Advice</div>', unsafe_allow_html=True)
            for remedy in disease_info["remedies"]:
                st.markdown(f'<div class="remedy-item">✅ {remedy}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Melanoma urgent warning
        if top_disease == "Melanoma":
            st.markdown("""
            <div class="warning-box">
                🚨 <b>URGENT:</b> Melanoma is a serious skin cancer. Please seek immediate medical attention.
                Do NOT self-treat. Visit a certified dermatologist or oncologist as soon as possible.
            </div>
            """, unsafe_allow_html=True)

        # General disclaimer
        st.markdown("""
        <div class="warning-box" style="background:rgba(100,116,139,0.1);border-color:rgba(100,116,139,0.3);color:#94a3b8">
            ℹ️ <b>Medical Disclaimer:</b> This AI tool is designed for educational purposes and preliminary screening only.
            It is NOT a substitute for professional medical diagnosis. Always consult a licensed dermatologist
            for any skin concerns.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DISEASE ENCYCLOPEDIA (always visible)
# ─────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📚 Skin Disease Encyclopedia — Click to Browse All Conditions"):
    selected = st.selectbox(
        "Select a disease to learn more:",
        options=list(DISEASE_INFO.keys()),
        label_visibility="collapsed"
    )
    info = get_disease_details(selected)
    if info:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"### {selected}")
            st.markdown(
                f'<div style="color:#cbd5e1;line-height:1.6">{info["description"]}</div>',
                unsafe_allow_html=True
            )
            st.markdown("**Symptoms:**")
            for s in info["symptoms"]:
                st.markdown(f"- {s}")
        with col_b:
            st.markdown("**Causes:**")
            for c in info["causes"]:
                st.markdown(f"- {c}")
            st.markdown("**Treatment & Remedies:**")
            for r in info["remedies"]:
                st.markdown(f"- {r}")
