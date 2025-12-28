import torch
import joblib
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# PATH SETUP (IMPORTANT)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "Models")

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD ML MODELS
# -----------------------------
tfidf = joblib.load(
    os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
)

lr_model = joblib.load(
    os.path.join(MODELS_DIR, "logistic_regression_model.pkl")
)

svm_model = joblib.load(
    os.path.join(MODELS_DIR, "linear_svm_model.pkl")
)

# -----------------------------
# LOAD TRANSFORMER MODELS
# -----------------------------
TRANSFORMER_MODELS = {
    "BERT": "BERT_bert-base",
    "RoBERTa": "BERT_RoBERTa-base",
    "SciBERT": "BERT_SciBERT-base"
}

transformers_loaded = {}

for name, folder in TRANSFORMER_MODELS.items():
    model_path = os.path.join(MODELS_DIR, folder)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model.to(device)
    model.eval()

    transformers_loaded[name] = (tokenizer, model)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def label_name(label):
    return "Drug Abuse" if label == 1 else "No Drug Abuse"

def predict_ml(model, text):
    X = tfidf.transform([text])

    # Logistic Regression (has predict_proba)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        pred = np.argmax(probs)
        confidence = probs[pred] * 100

    # Linear SVM (NO predict_proba)
    else:
        score = model.decision_function(X)[0]
        prob = 1 / (1 + np.exp(-score))  # sigmoid
        pred = 1 if prob >= 0.5 else 0
        confidence = prob * 100 if pred == 1 else (1 - prob) * 100

    return pred, confidence


def predict_transformer(tokenizer, model, text):
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt"
    )

    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    pred = np.argmax(probs)
    confidence = probs[pred] * 100
    return pred, confidence

# -----------------------------
# MAIN LOOP
# -----------------------------
print("\n=== Drug Abuse Detection | Multi-Model CLI ===")
print("Type 'exit' to quit\n")

while True:
    text = input("Enter text: ")

    if text.lower() == "exit":
        break

    print("\nResults:")
    print("-" * 45)

    pred, conf = predict_ml(lr_model, text)
    print(f"Logistic Regression : {label_name(pred)} ({conf:.2f}%)")

    pred, conf = predict_ml(svm_model, text)
    print(f"Linear SVM          : {label_name(pred)} ({conf:.2f}%)")

    for name, (tokenizer, model) in transformers_loaded.items():
        pred, conf = predict_transformer(tokenizer, model, text)
        print(f"{name:<18}: {label_name(pred)} ({conf:.2f}%)")

    print("-" * 45 + "\n")
