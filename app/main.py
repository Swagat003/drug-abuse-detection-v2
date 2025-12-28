import streamlit as st
import torch
import joblib
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit.components.v1 as components

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Clinical Drug Abuse Detection Dashboard",
    layout="wide"
)

# ==================================================
# FORCE LIGHT MEDICAL THEME
# ==================================================
st.markdown(
    """
    <style>
        html, body, [class*="css"] {
            background-color: #F7F9FB !important;
            color: #2C3E50 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ==================================================
# BASIC GLOBAL STYLES
# ==================================================
st.markdown(
    """
    <style>
        .section-title {
            font-size: 20px;
            font-weight: 600;
            color: #1F3A5F;
            margin-bottom: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ==================================================
# PATH SETUP
# ==================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "Models")

# ==================================================
# DEVICE
# ==================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================================================
# LOAD ML MODELS
# ==================================================
@st.cache_resource
def load_ml_models():
    tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    lr = joblib.load(os.path.join(MODELS_DIR, "logistic_regression_model.pkl"))
    svm = joblib.load(os.path.join(MODELS_DIR, "linear_svm_model.pkl"))
    return tfidf, lr, svm

tfidf, lr_model, svm_model = load_ml_models()

# ==================================================
# LOAD TRANSFORMERS (DEPLOYMENT SAFE)
# ==================================================
@st.cache_resource
def load_transformers():
    model_map = {
        "BERT": "bert-base-uncased",
        "RoBERTa": "roberta-base",
        "SciBERT": "allenai/scibert_scivocab_uncased"
    }

    models = {}
    for name, model_id in model_map.items():
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=2
        )
        model.to(device)
        model.eval()
        models[name] = (tokenizer, model)

    return models

transformer_models = load_transformers()

# ==================================================
# PREDICTION FUNCTIONS
# ==================================================
def predict_ml(model, text):
    X = tfidf.transform([text])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        pred = int(np.argmax(probs))
        conf = float(probs[pred])
    else:
        score = model.decision_function(X)[0]
        prob = 1 / (1 + np.exp(-score))
        pred = 1 if prob >= 0.5 else 0
        conf = float(prob if pred == 1 else (1 - prob))

    return pred, conf

def predict_transformer(tokenizer, model, text):
    encoding = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors="pt"
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
        probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()

    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    return pred, conf

# ==================================================
# ANIMATED CARD RENDER
# ==================================================
def render_result(model_name, pred, conf):
    label = "No Drug Abuse" if pred == 0 else "Drug Abuse Detected"
    color = "#4EC37F" if pred == 0 else "#DC4444"

    percentage = max(0, min(100, int(conf * 100)))
    radius = 34
    circumference = 2 * 3.1416 * radius
    offset = circumference - (percentage / 100) * circumference

    html = f"""
    <div style="
        background:white;
        border-radius:12px;
        padding:16px;
        box-shadow:0 2px 8px rgba(0,0,0,0.05);
        display:flex;
        justify-content:space-between;
        align-items:center;
        margin-bottom:12px;
        font-family:Arial;
    ">
        <div>
            <div style="font-size:16px;font-weight:600;color:#2C3E50;">
                {model_name}
            </div>
            <div style="color:{color};font-size:20px;font-weight:600;">
                {label}
            </div>
        </div>

        <svg width="90" height="90" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="{radius}"
                    stroke="#E6E6FA" stroke-width="8" fill="none"/>
            <circle cx="50" cy="50" r="{radius}"
                    stroke="#6A5ACD" stroke-width="8" fill="none"
                    stroke-dasharray="{circumference}"
                    stroke-dashoffset="{circumference}"
                    style="transform:rotate(-90deg);transform-origin:50% 50%;
                           animation:fill 1.4s ease-out forwards;"/>
            <text x="50" y="56"
                  text-anchor="middle"
                  font-size="20"
                  font-weight="700"
                  fill="#6A5ACD">
                {percentage}%
            </text>
            <style>
                @keyframes fill {{
                    to {{ stroke-dashoffset: {offset}; }}
                }}
            </style>
        </svg>
    </div>
    """

    components.html(html, height=130)

# ==================================================
# HEADER
# ==================================================
st.markdown(
    """
    <h1 style="text-align:center;color:#1F3A5F;">
        Clinical Drug Abuse Detection Dashboard
    </h1>
    <p style="text-align:center;color:#666;">
        NLP & Transformer-based Textual Risk Analysis
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ==================================================
# INPUT
# ==================================================
st.markdown("<div class='section-title'>Patient / Social Media Text</div>", unsafe_allow_html=True)

user_text = st.text_area(
    "",
    height=140,
    placeholder="Enter or paste text for analysis..."
)

analyze_btn = st.button("Run Analysis")

# ==================================================
# RESULTS
# ==================================================
if analyze_btn and user_text.strip():

    st.divider()
    st.markdown("<div class='section-title'>Model-wise Assessment</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>Traditional ML Models</div>", unsafe_allow_html=True)
        render_result("Logistic Regression", *predict_ml(lr_model, user_text))
        render_result("Linear SVM", *predict_ml(svm_model, user_text))

    with col2:
        st.markdown("<div class='section-title'>Transformer Models</div>", unsafe_allow_html=True)
        for name, (tok, mdl) in transformer_models.items():
            render_result(name, *predict_transformer(tok, mdl, user_text))

    # ==================================================
    # CONSENSUS
    # ==================================================
    preds = []
    preds.append(predict_ml(lr_model, user_text)[0])
    preds.append(predict_ml(svm_model, user_text)[0])
    for tok, mdl in transformer_models.values():
        preds.append(predict_transformer(tok, mdl, user_text)[0])

    final_pred = 1 if sum(preds) >= len(preds) / 2 else 0
    final_label = "Drug Abuse Risk Detected" if final_pred else "No Drug Abuse Risk"
    final_color = "#DC4444" if final_pred else "#4EC37F"

    st.divider()
    st.markdown(
        f"""
        <div style="
            background:white;
            border-radius:12px;
            padding:16px;
            box-shadow:0 2px 8px rgba(0,0,0,0.05);
            border-left:6px solid {final_color};
        ">
            <div style="font-size:18px;font-weight:600;color:#1F3A5F;">
                Final Clinical Assessment
            </div>
            <div style="color:{final_color};font-size:22px;margin-top:6px;">
                {final_label}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

elif analyze_btn:
    st.warning("Please enter text for analysis.")

# ==================================================
# FOOTER
# ==================================================
st.divider()
st.caption("For academic and research demonstration only. Not a medical diagnostic tool.")
