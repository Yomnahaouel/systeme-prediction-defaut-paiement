# 🏦 Credit Default Risk Prediction System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-Academic-yellow.svg)]()

> **Machine Learning pipeline for predicting loan default probability**  
> Best Model: CatBoost with **AUC = 0.786**

![Architecture](docs/architecture-diagram.png)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [API Documentation](#-api-documentation)
- [Model Details](#-model-details)
- [Project Structure](#-project-structure)
- [Development](#-development)
- [Security](#-security)
- [Authors](#-authors)

---

## 🎯 Overview

This project predicts the probability of a client defaulting on a loan using machine learning. Built on the **Home Credit Default Risk** dataset (~307K clients), it includes:

- Complete ML pipeline (EDA → Feature Engineering → Training → Evaluation)
- REST API for real-time predictions
- Interactive web dashboard
- Docker containerization

**Business Problem:** How can we predict, from financial and demographic data, whether a client will default on their loan?

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **ML Pipeline** | End-to-end: data processing → model training → evaluation |
| 📊 **5 Models Compared** | CatBoost, LightGBM, XGBoost, Random Forest, Logistic Regression |
| 🎯 **Best AUC: 0.786** | CatBoost with Optuna hyperparameter tuning |
| 🔧 **102 Features** | Engineered financial ratios and aggregations |
| 🌐 **REST API** | FastAPI with 6 endpoints |
| 🎨 **Web Dashboard** | Streamlit interactive UI |
| 🐳 **Dockerized** | One command deployment |
| 📈 **SHAP Explainability** | Feature importance visualization |
| ⚖️ **Optimized Threshold** | Business cost-based (0.35) |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER REQUEST                             │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FRONTEND (Streamlit)                        │
│                        Port: 8501                                │
│  • Interactive input forms                                       │
│  • Risk visualization (gauge chart)                              │
│  • SHAP explanations                                             │
└─────────────────────────────────┬───────────────────────────────┘
                                  │ HTTP
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                        API (FastAPI)                             │
│                        Port: 8000                                │
│  • /predict - Single prediction                                  │
│  • /predict/batch - Batch predictions                            │
│  • /explain - SHAP explanation                                   │
│  • /features - Feature importance                                │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ML MODEL (CatBoost)                          │
│  • 102 features                                                  │
│  • Threshold: 0.35                                               │
│  • AUC: 0.786                                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone <repo-url>
cd systeme-prediction-defaut-paiement

# Start all services
docker-compose up -d

# Access:
# - Dashboard: http://localhost:8501
# - API Docs:  http://localhost:8000/docs
```

### Option 2: Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Start API
uvicorn api.app_v2:app --host 0.0.0.0 --port 8000 --reload

# Start Frontend (new terminal)
streamlit run frontend/app.py
```

---

## 📡 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/info` | Model information |
| GET | `/threshold` | Threshold configuration |
| GET | `/features?top_n=20` | Top feature importance |
| POST | `/predict` | Single client prediction |
| POST | `/predict/batch` | Batch predictions |
| POST | `/explain` | SHAP explanation |

### Example: Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "EXT_SOURCE_1": 0.5,
      "EXT_SOURCE_2": 0.6,
      "EXT_SOURCE_3": 0.4,
      "AMT_CREDIT": 500000,
      "AMT_INCOME_TOTAL": 150000,
      "DAYS_BIRTH": -15000
    }
  }'
```

**Response:**
```json
{
  "client_id": "client_001",
  "default_probability": 0.232554,
  "prediction": 0,
  "risk_level": "LOW-MEDIUM",
  "threshold": 0.35,
  "confidence": "Medium"
}
```

### Interactive Docs

Visit **http://localhost:8000/docs** for Swagger UI.

---

## 🤖 Model Details

### Performance Comparison

| Model | ROC-AUC | Recall | Training Time |
|-------|---------|--------|---------------|
| **CatBoost (Optuna)** | **0.786** | 72% | ~30 min (GPU) |
| LightGBM | 0.76 | 68% | ~10 min |
| XGBoost | 0.74 | 65% | ~15 min |
| Random Forest | 0.73 | 60% | ~20 min |
| Logistic Regression | 0.70 | 71% | ~1 min |

### Top 10 Features (SHAP)

1. FLAG_DOCUMENT_15
2. REG_REGION_NOT_LIVE_REGION
3. AMT_REQ_CREDIT_BUREAU_WEEK
4. INS_PAYMENT_DIFF_MEAN
5. NAME_EDUCATION_TYPE
6. ANNUITY_INCOME_RATIO
7. DAYS_LAST_PHONE_CHANGE
8. FLAG_DOCUMENT_8
9. INS_LATE_COUNT
10. FLAG_DOCUMENT_18

### Threshold Optimization

| Metric | Value |
|--------|-------|
| Optimal Threshold | 0.35 |
| Cost of False Negative | $10,000 |
| Cost of False Positive | $500 |

---

## 📁 Project Structure

```
systeme-prediction-defaut-paiement/
│
├── api/
│   └── app_v2.py              # FastAPI application
│
├── frontend/
│   └── app.py                 # Streamlit dashboard
│
├── src/
│   ├── data_engineering.py    # Data loading & cleaning
│   ├── feature_engineering.py # Feature creation
│   ├── training.py            # Model training
│   ├── threshold_optimizer.py # Business threshold tuning
│   ├── shap_explainer.py      # SHAP analysis
│   └── model_ensemble.py      # Model ensembling
│
├── models/
│   ├── catboost_optimized.joblib  # Best model ⭐
│   ├── threshold_config.json      # Threshold settings
│   └── shap_feature_importance.csv
│
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.frontend
│   └── nginx.conf
│
├── tests/
│   ├── test_api.py
│   └── test_model.py
│
├── docs/
│   └── ARCHITECTURE.md
│
├── docker-compose.yml
├── requirements.txt
├── requirements-api.txt
├── requirements-frontend.txt
├── SECURITY_AUDIT.md
└── README.md
```

---

## 💻 Development

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test
pytest tests/test_model.py -v
```

### Code Quality

```bash
# Format code
black src/ api/ frontend/

# Lint
flake8 src/ api/ frontend/
```

### Docker Commands

```bash
# Build
docker-compose build

# Start
docker-compose up -d

# Logs
docker-compose logs -f api

# Stop
docker-compose down

# Rebuild without cache
docker-compose build --no-cache
```

---

## 🔒 Security

See [SECURITY_AUDIT.md](SECURITY_AUDIT.md) for full report.

**Summary:**
- ✅ Input validation (Pydantic)
- ✅ No SQL injection (no database)
- ✅ Container isolation
- ⚠️ Add rate limiting for production
- ⚠️ Add authentication for production

---

## 📊 Dataset

**Source:** [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/c/home-credit-default-risk)

| File | Description |
|------|-------------|
| application_train.csv | Main training data (307K rows) |
| bureau.csv | Client's previous credits |
| credit_card_balance.csv | Credit card history |
| installments_payments.csv | Payment history |
| previous_application.csv | Previous loan applications |

---

## 🎓 Academic Context

**Course:** Data Science / Machine Learning  
**Institution:** Engineering School  
**Focus:** Credit scoring, risk prediction, MLOps

---

## 👥 Authors

- **Yomna Haouel** - Initial work, ML pipeline
- **7afnawi (AI Assistant)** - API development, Docker, deployment

---

## 📄 License

Academic project - All rights reserved.

---

## 🙏 Acknowledgments

- [Home Credit](https://www.homecredit.net/) for the dataset
- [CatBoost](https://catboost.ai/) for the ML library
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Streamlit](https://streamlit.io/) for the dashboard

---

<div align="center">

**⭐ Star this repo if you found it helpful! ⭐**

Made with ❤️ for learning

</div>
