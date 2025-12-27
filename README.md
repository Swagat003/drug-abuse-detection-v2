# 🧠 Drug Abuse Detection from Social Media Text using NLP & Transformers

## 📌 Overview
This project aims to **detect drug abuse behavior from social media text** (tweets, comments, posts) using **Natural Language Processing (NLP)** and **Machine Learning / Deep Learning models**.

The system analyzes textual content to determine whether a user is likely engaging in **drug abuse–related behavior**, based on linguistic patterns, semantics, and contextual meaning.

---

## 🎯 Objectives
- Build a **binary classification system** (Drug Abuse / No Drug Abuse)
- Compare **traditional machine learning models** with **transformer-based models**
- Apply **context-aware NLP architectures** (BERT, RoBERTa, SciBERT)
- Generate **report-ready evaluation metrics and visualizations**

---

## 🧩 Project Pipeline

```
Raw Social Media Text
        ↓
Data Cleaning & Preprocessing
        ↓
Dataset Construction
        ↓
Feature Extraction (TF-IDF)
        ↓
Baseline ML Models
        ↓
Transformer Models (BERT / RoBERTa / SciBERT)
        ↓
Evaluation & Model Comparison
```

---

## 📂 Project Structure

```
├── Datasets/
│   ├── final_dataset_cleaned.csv
│   ├── bert_train.csv
│   ├── bert_val.csv
│   └── bert_test.csv
│
├── Models/
│   ├── logistic_regression_model.pkl
│   ├── linear_svm_model.pkl
│   ├── BERT_bert-base-uncased/
│   ├── BERT_roberta-base/
│   ├── BERT_scibert_scivocab_uncased/
│   └── Evaluation/
│       ├── metrics/
│       ├── reports/
│       ├── plots/
│       └── confusion_matrices/
│
├── notebooks/
│   ├── 0_Data_Preprocessing.ipynb
│   ├── 1_Feature_Extraction_TFIDF.ipynb
│   ├── 2_Model_Training_Baseline.ipynb
│   ├── 3_Model_Evaluation.ipynb
│   ├── 4_BERT_Data_Preprocessing.ipynb
│   ├── 5_BERT_Training_bert_base_uncased.ipynb
│   └── 6_BERT_Evaluation.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧪 Models Used

### 🔹 Traditional Machine Learning
- Logistic Regression (TF-IDF)
- Linear Support Vector Machine (TF-IDF)

### 🔹 Transformer-Based Models
- **BERT** (`bert-base-uncased`)
- **RoBERTa** (`roberta-base`)
- **SciBERT** (`allenai/scibert_scivocab_uncased`)

---

## 📊 Evaluation Metrics
All models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

📌 **Recall is prioritized**, as missing a drug-abuse case is more critical than raising false alarms.

---

## 📈 Visualization Techniques
To ensure meaningful comparison beyond bar charts, the following visualization techniques are used:
- Radar (Spider) Charts
- Heatmaps
- Line Plots
- Error Rate Analysis

These plots help highlight subtle performance differences when metrics are close.

---

## 🚀 Key Results
- Transformer-based models outperform traditional ML models in **semantic understanding**
- **SciBERT** shows strong performance due to its training on biomedical and scientific text
- High accuracy is achieved due to clear lexical separation in the curated dataset

---

## ⚠️ Limitations
- Dataset is curated and relatively clean
- Real-world social media text may include sarcasm, evolving slang, and ambiguous expressions
- Model performance may decrease in uncontrolled, real-world environments

---

## 🔮 Future Work
- Real-time social media stream analysis
- Multilingual drug abuse detection
- Integration with social media platforms
- Explainable AI (XAI) for interpretability
- Deployment using FastAPI or Flask

---

## 🛠️ Tech Stack
- Python
- Scikit-learn
- PyTorch
- HuggingFace Transformers
- Pandas, NumPy
- Matplotlib, Seaborn
- Jupyter Notebook
- Kaggle / Google Colab

---

## 📜 Disclaimer
This project is intended strictly for **academic and research purposes**.  
It does **not diagnose individuals** and should not be used for legal, medical, or law-enforcement decisions.

---

## 🙌 Acknowledgements
- Kaggle Datasets
- HuggingFace Transformers
- Scikit-learn & PyTorch communities

