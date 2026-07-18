# 🤖 LLM Satisfaction Predictor

A beginner-friendly, production-structured Machine Learning project that predicts
the expected **user satisfaction score** for an LLM usage session.

Built with **XGBoost**, **FastAPI**, and **Streamlit**.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Features](#-features)
- [Target Leakage Prevention](#-target-leakage-prevention)
- [ML Pipeline](#-ml-pipeline)
- [Evaluation Metrics](#-evaluation-metrics)
- [Model Limitations and Honest Assessment](#-model-limitations-and-honest-assessment)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Local Setup](#-local-setup)
- [Testing](#-testing)
- [Future Improvements](#-future-improvements)

---

## 🎯 Project Overview

This project learns from historical LLM usage records to predict the expected
user satisfaction score given the configuration of a planned LLM request.

- **Training pipeline** — clean, reproducible, artifact-generating
- **FastAPI backend** — validated REST API with automatic docs
- **Streamlit frontend** — simple dashboard that calls the API
- **Tests** — model and API tests using pytest

---

## 🏋️ Problem Statement

Given the configuration of an LLM usage session (model, domain, task, prompt settings),
predict the expected **user satisfaction score**.

```
LLM Configuration
      ↓
XGBoost Regressor
      ↓
Predicted User Satisfaction (3–5 scale)
```

This is a **supervised regression** problem:
- **Supervised** — we learn from labelled historical records
- **Regression** — we predict a continuous numeric score

---

## 📊 Dataset

**File:** `data/raw/genai_llm_usage_dataset_1000.csv`
**Source:** GenAI LLM Usage Dataset (~1,000 records)

| Property | Value |
|----------|-------|
| Rows | 1,000 |
| Columns | 14 |
| Missing values | None |
| Duplicate rows | None |
| Target range | 3, 4, or 5 |

---

## 🔢 Features

### ✅ Selected Features (7 — all available BEFORE the LLM request)

| Feature | Type | Description |
|---------|------|-------------|
| `model_name` | Categorical | LLM model (GPT-4o, Claude 3.7, Gemma 3, Llama 3.1, Mistral Large, Qwen 2.5) |
| `application_domain` | Categorical | Business domain (Coding, Education, Healthcare, etc.) |
| `task_type` | Categorical | Task category (QA, Summarization, Code Generation, etc.) |
| `prompt_length` | Numerical | Number of characters in the prompt |
| `temperature` | Numerical | LLM randomness parameter (0.0–2.0) |
| `top_p` | Numerical | Nucleus sampling parameter (0.0–1.0) |
| `rag_enabled` | Numerical (0/1) | Whether RAG is enabled |

**Target:** `user_satisfaction` (integer: 3, 4, or 5)

---

## 🚨 Target Leakage Prevention

Target leakage means using information that would not be available at prediction time.
The following columns are **intentionally excluded**:

| Excluded Column | Reason |
|----------------|--------|
| `total_tokens` | Includes completion tokens — only known AFTER the LLM responds |
| `latency_sec` | Response time — only measured AFTER the LLM responds |
| `hallucination_flag` | Only evaluated AFTER reading the response |
| `successful_response` | Only known AFTER the response arrives |
| `estimated_cost_usd` | Fully calculated AFTER usage |
| `session_id` | Row identifier — no predictive value |

**Verified:** `total_tokens - prompt_length` (output tokens) ranges from 21 to 1,200,
confirming it includes post-response completion tokens and must be excluded.

---

## 🔧 ML Pipeline

```
genai_llm_usage_dataset_1000.csv
  ↓
Data Cleaning (drop missing targets, drop duplicates)
  ↓
Feature Selection (7 leakage-free features)
  ↓
Train/Test Split (80% train / 20% test, random_state=42)
  ↓
ColumnTransformer
  ├── StandardScaler      → prompt_length, temperature, top_p, rag_enabled
  └── OneHotEncoder       → model_name, application_domain, task_type
  ↓
XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, ...)
  ↓
Clip predictions to [3.0, 5.0]
  ↓
Evaluation: MAE, RMSE, R²
  ↓
Save: models/satisfaction_pipeline.pkl
```

The **sklearn Pipeline** wraps the preprocessor and model together.
The same preprocessing steps are applied automatically at training and prediction time.

---

## 📈 Evaluation Metrics

### What the metrics mean

| Metric | Meaning | Better when |
|--------|---------|-------------|
| **MAE** (Mean Absolute Error) | Average prediction error in satisfaction points | Lower is better |
| **RMSE** (Root Mean Squared Error) | Like MAE but penalises large errors more | Lower is better |
| **R²** (R-squared) | How much variance the model explains (1.0 = perfect, 0.0 = predicts mean) | Closer to 1.0 |

### Actual results (from trained model)

| Metric | Value |
|--------|-------|
| MAE | 0.5068 |
| RMSE | 0.6113 |
| R² | -0.256 |

---

## ⚠️ Model Limitations and Honest Assessment

> **Why is R² negative?**

A negative R² means the model currently performs **slightly worse than simply
predicting the mean satisfaction score for every request**.

This is expected given the dataset characteristics:

1. **Very small dataset** — only 1,000 records, split 800/200 train/test.
   XGBoost works better with tens of thousands of records.

2. **Low feature correlation with target** — numerical features have very
   weak correlations with `user_satisfaction`:
   - `prompt_length`: −0.09
   - `temperature`: −0.003
   - `top_p`: +0.005
   - `rag_enabled`: +0.003

3. **Almost no group differences** — categorical features also show little
   variation in satisfaction:
   - All models average between 4.25 and 4.47
   - All domains average between 4.24 and 4.41
   - All task types average between 4.27 and 4.41

4. **Narrow target range** — the target only takes 3 values (3, 4, 5) with
   94% of records being 4 or 5. The baseline MAE (always predict the mean)
   is only 0.53, which is hard to beat with this data.

**The model is technically correct but the dataset does not contain strong
predictive signals for satisfaction in the available pre-request features.**

This is a realistic ML outcome that real practitioners encounter. More data,
richer features (e.g., user history, domain-specific metadata), or including
some post-response signals as features would likely improve performance.

---

## 🏗️ Architecture

```
User
  ↓ (browser)
Streamlit Frontend    (frontend/streamlit_app.py)   port 8501
  ↓ HTTP POST /predict
FastAPI Backend       (app/main.py)                 port 8000
  ↓
src/predict.py        (load_pipeline, predict)
  ↓
models/satisfaction_pipeline.pkl
  ├── ColumnTransformer (StandardScaler + OneHotEncoder)
  └── XGBRegressor
  ↓
Predicted User Satisfaction (3–5 scale)
```

---

## 📁 Project Structure

```
Gym project/
├── data/
│   └── raw/
│       └── genai_llm_usage_dataset_1000.csv
│
├── notebooks/
│   ├── EDA.ipynb                         ← LLM dataset analysis
│   └── EDA_powerlifting_archive.ipunb    ← archived old notebook
│
├── src/
│   ├── config.py          ← all paths, features, hyperparams
│   ├── data_processing.py ← load, clean, build_preprocessor
│   ├── train.py           ← end-to-end training script
│   ├── predict.py         ← prediction service
│   └── utils.py           ← logging, JSON helpers
│
├── models/
│   └── satisfaction_pipeline.pkl   ← generated by src/train.py
│
├── app/
│   ├── main.py    ← FastAPI: /, /health, /predict
│   └── schemas.py ← Pydantic request/response models
│
├── frontend/
│   └── streamlit_app.py   ← Streamlit dashboard
│
├── tests/
│   ├── test_model.py   ← pipeline and validation tests
│   └── test_api.py     ← API integration tests
│
├── artifacts/
│   ├── metrics.json
│   ├── feature_importance.csv
│   ├── model_metadata.json
│   └── plots/
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## 💻 Local Setup

### Prerequisites
- Python 3.9+
- Dataset file in `data/raw/genai_llm_usage_dataset_1000.csv`

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train the model
```bash
python src/train.py
```

### Start the FastAPI backend
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
API docs: http://localhost:8000/docs

### Start the Streamlit frontend
```bash
python -m streamlit run frontend/streamlit_app.py
```
UI: http://localhost:8501

---

## 🧪 Testing

```bash
# Run all tests (model-dependent tests skip if model not trained)
python -m pytest tests/ -v

# Run only validation tests (no model required)
python -m pytest tests/test_model.py::TestInputValidation -v
```

---

## 🚀 Future Improvements

| Improvement | Expected Impact |
|-------------|----------------|
| More training data (10,000+ records) | Most impactful improvement |
| Richer pre-request features (user history, prompt complexity score) | Better signal |
| Cross-validation instead of single train/test split | More reliable metrics |
| Hyperparameter tuning (GridSearchCV) | Marginal improvement |
| Try other regression models (RandomForest, Ridge) | Compare baselines |
| Model monitoring in production | Detect data drift |
| Better feature engineering | Encode domain expertise |

---

## 📄 Dataset Limitations

- Dataset has approximately 1,000 records — very small for ML
- Pre-request features show weak correlation with satisfaction
- The model should not be treated as a universal satisfaction predictor
- Predictions represent patterns in this specific dataset only
- The dataset may not represent all LLM users or all LLM systems
