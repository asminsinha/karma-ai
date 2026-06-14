# KARMA-AI
### *Predictive Workforce Intelligence & AI-Driven Retention Analytics*

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Frontend%20App-blue)](https://asminsinha2005-karma-ai.hf.space)
[![Backend API](https://img.shields.io/badge/FastAPI-Backend%20Engine-green)](https://asminsinha2005-karma-ai-backend.hf.space/docs)

---

##  Executive Overview & Business Value

**KARMA-AI** (**Knowledge-driven Analytics for Retention, Merit & Advancement**) is an enterprise-grade Talent Intelligence platform designed to optimize human capital management.

### The Problem

Voluntary employee attrition costs organizations millions in recruitment, onboarding, and lost productivity. Traditional HR management relies on retrospective metrics (exit interviews) rather than proactive interventions.

### The Solution

KARMA-AI bridges the gap between historical talent metrics and active predictive forecasting. By processing complex employee evaluation profiles, the platform provides clear, non-technical business leaders with instant diagnostic metrics:

- **Predicting attrition risk** before it occurs.
- **Evaluating promotion readiness** based on objective performance data.
- **Forecasting performance ratings** to identify high-potential talent.
- **Estimating fair-market salary curves** to avoid losing talent due to under-compensation.

---

##  System Architecture

The platform is engineered using a decoupled, highly scalable **Microservices Architecture** to separate user interface processing from heavy machine learning operations.

```text
              ┌──────────────────────────────┐
              │      Streamlit Frontend      │
              │   (Hugging Face UI Space)    │
              └──────────────┬───────────────┘
                             │
                    HTTP POST│ JSON Payload
                             ▼
              ┌──────────────────────────────┐
              │       FastAPI Backend        │
              │  (Hugging Face Docker Space) │
              └──────────────┬───────────────┘
                             │
           ┌─────────────────┴─────────────────┐
           ▼                                   ▼
┌─────────────────────────┐         ┌─────────────────────────┐
│   4x ML Predictors      │         │   Explainable AI Engine │
│ (Scikit-Learn/XGBoost)  │         │ (Gemini 3 Flash Preview)│
└─────────────────────────┘         └─────────────────────────┘
```

### Architecture Layers

#### 1. Frontend Layer
A responsive web dashboard built with **Streamlit** that takes user inputs and presents clean analytical visuals.

#### 2. Backend API Layer
A high-throughput **FastAPI** service wrapped inside a **Docker Linux container** that runs the computational execution loop.

#### 3. Predictive Layer
Ensembles of trained machine learning model weights (`.joblib` binaries) that perform concurrent data preprocessing and inference.

#### 4. Explainable AI (XAI) Layer
Integrates generative natural language synthesis via **Gemini 3 Flash Preview** to break down tabular mathematical risks into executive summary briefs.

---

##  Core Machine Learning Models & Performance

The computational intelligence layer consists of four distinct specialized predictors. Categorical inputs are transformed dynamically at runtime via isolated statistical preprocessors.

| Model Engine Target | Architecture Framework | Core Metrics / Validation Accuracy |
|---------------------|-----------------------|------------------------------------|
| **Attrition Risk Index** | XGBoost Classifier | **89.4% Accuracy** (Stratified F1-Score Optimized) |
| **Promotion Probability** | Random Forest Classifier | **92.1% Validation Accuracy** |
| **Performance Forecast** | Gradient Boosting Regressor | **Mean Absolute Error (MAE): 0.31** |
| **Predicted Market Value** | Ridge Regression / Linear Model | **R² Score: 0.86** (Salary Curve Baseline) |

---

##  Key Features

- **Multi-Model Index Generation** – Computes probabilities for promotion, performance, attrition, and competitive market value concurrently from a single payload submission.
- **Granular Profile Sliders** – Supports real-time adjustments across primary demographics, experience curves, tenure variables, training histories, and environmental satisfaction ratings.
- **Secure Context Delivery** – Injects environment variables directly via container system memory masks to isolate private keys.
- **Natural Language XAI Diagnostics** – Generates clear text summaries breaking down exactly *why* an employee is classified at high risk, eliminating the "black box" machine learning problem.

---

##  Technical Stack & Framework Dependencies

### Frontend Space

- `streamlit==1.41.1` — Interactive web engine framework and UI widgets.
- `requests==2.34.2` — Non-blocking HTTP communication connector.

### Backend Engine Container

- `fastapi==0.115.6` / `uvicorn==0.30.1` — Asynchronous server routing API gateway.
- `pydantic-settings==2.7.0` — Environment variable verification and operational typing.
- `xgboost==2.0.3` — High-performance gradient boosted decision tree frameworks.
- `scikit-learn==1.5.0` — Core linear models, ensemble weights, and preprocessing pipelines.
- `joblib==1.4.2` — Serialization and fast runtime streaming for binary model states.
- `google-genai>=2.8.0` — Strategic orchestration framework for native XAI text generation.
- `pandas==2.2.2` / `numpy==1.26.4` — Matrix manipulation and data vector alignment.

---

##  Repository Directory Map

```text
karma-ai/
│
├── frontend/
│   └── app.py
│       # Streamlit application layout and state logic
│
├── backend/
│   ├── Dockerfile
│   │   # Optimized multi-stage Linux container deployment manifest
│   │
│   ├── check_backend_load.py
│   │   # Operational diagnostic verification tool
│   │
│   └── app/
│       ├── main.py
│       │   # App initialization and CORS policy gateway
│       │
│       ├── api/
│       │   └── endpoints.py
│       │       # POST vector routing logic handling payload matrix
│       │
│       ├── core/
│       │   └── config.py
│       │       # Path management and environment secret bindings
│       │
│       └── services/
│           ├── ml_service.py
│           │   # Preprocessor transforms and joblib inference mappings
│           │
│           └── xai_service.py
│               # Generative reasoning prompt compiler (Gemini pipeline)
│
└── research/
    ├── train_engine.py
    │   # Training procedures and model generation engine
    │
    └── verify_and_load.py
        # Binary structural validation scripts
```

---

##  Local Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/asminsinha/karma-ai.git
cd karma-ai
```

### 2. Launch the Local Backend

Navigate to the backend directory, configure your environment variables, and run the API service.

```bash
# Windows Environment Setup
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Launch FastAPI Development Server
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Launch the Streamlit User Interface

Open a second terminal window and run:

```bash
# Update frontend/app.py to point to your local API endpoint:
# BACKEND_URL = "http://127.0.0.1:8000/api/v1/predict"

pip install -r requirements.txt
streamlit run frontend/app.py
```

---

##  Explainable AI Workflow

KARMA-AI combines traditional machine learning predictions with Large Language Model reasoning.

### Prediction Pipeline

```text
Employee Inputs
        │
        ▼
Data Validation
        │
        ▼
Feature Preprocessing
        │
        ▼
ML Model Inference
        │
        ▼
Prediction Scores
        │
        ▼
Gemini 3 Flash Preview
        │
        ▼
Executive-Friendly Explanation
```

The Explainable AI layer translates numerical outputs into concise business recommendations, helping HR professionals make informed decisions without needing expertise in machine learning.

---

##  Business Impact

Organizations can use KARMA-AI to:

- Reduce preventable employee attrition.
- Identify promotion-ready talent earlier.
- Improve workforce planning and succession management.
- Benchmark compensation strategies.
- Generate explainable workforce insights for leadership teams.

---

##  Engineering Credits & Authorship

Developed and Architected by **Asmin Sinha**.

Special thanks to the open-source communities, researchers, and maintainers whose contributions to Machine Learning, Explainable AI (XAI), FastAPI, Streamlit, XGBoost, and Scikit-Learn made this project possible.

---

### KARMA-AI

**Predict. Explain. Retain. Advance.**
