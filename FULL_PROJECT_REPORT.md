# 🏦 Credit Default Risk Prediction System
## Complete Project Report

**Project:** Système de Prédiction de Défaut de Paiement  
**Author:** Yomna Haouel (Original) + 7afnawi (Enhancements)  
**Date:** March 5-6, 2026  
**Status:** ✅ Production Ready  

---

## 📋 Executive Summary

A complete **end-to-end Machine Learning pipeline** for predicting credit default risk using the **Home Credit dataset** from Kaggle. The system predicts whether a loan applicant will default, helping financial institutions make informed lending decisions.

### Key Achievements

| Metric | Value |
|--------|-------|
| **Model AUC** | 0.786 |
| **Recall** | 72% (catches 72% of actual defaults) |
| **Precision** | 25% |
| **F1 Score** | 0.37 |
| **Dataset Size** | 307,511 loan applications |
| **Features Used** | 167 engineered features |
| **Best Model** | CatBoost (with Optuna tuning) |

---

## 🎯 Business Problem

**Challenge:** Banks need to predict which loan applicants are likely to default to:
- Minimize financial losses from bad loans
- Avoid rejecting good customers
- Make fair, data-driven lending decisions

**Cost Matrix:**
- Cost of False Negative (approving a default): **$10,000**
- Cost of False Positive (rejecting a good customer): **$500**

**Optimal Threshold:** 0.35 (tuned for cost sensitivity)

---

## 🔧 Technical Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     PRODUCTION ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────┐     ┌─────────────┐     ┌─────────────────────────┐
│   Client    │────▶│  Cloudflare │────▶│     Docker Compose      │
│  (Browser)  │     │   Tunnel    │     │  ┌─────────┐ ┌───────┐  │
└─────────────┘     └─────────────┘     │  │Frontend │ │  API  │  │
                                        │  │Streamlit│◀│FastAPI│  │
                                        │  │  :8501  │ │ :8000 │  │
                                        │  └─────────┘ └───────┘  │
                                        └─────────────────────────┘
```

### Tech Stack

| Component | Technology |
|-----------|------------|
| ML Framework | CatBoost, LightGBM, XGBoost, Scikit-learn |
| Optimization | Optuna (100 trials) |
| Explainability | SHAP |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Containerization | Docker + Docker Compose |
| Deployment | Cloudflare Tunnel |
| Experiment Tracking | MLflow |

---

## 📊 Dataset Overview

### Source
**Home Credit Default Risk** - Kaggle Competition Dataset

### Statistics

| Attribute | Value |
|-----------|-------|
| Total Samples | 307,511 |
| Features (Raw) | 122 |
| Features (Engineered) | 167 |
| Target Variable | Binary (0 = No Default, 1 = Default) |
| Class Imbalance | ~8% defaults (handled with SMOTE) |
| Train/Test Split | 80% / 20% |

### Data Sources Merged
1. **application_train.csv** - Main application data
2. **bureau.csv** - Credit bureau data
3. **previous_application.csv** - Previous loan applications
4. **POS_CASH_balance.csv** - Point of sale/cash loan balance
5. **installments_payments.csv** - Payment history
6. **credit_card_balance.csv** - Credit card balance

---

## 🔬 Feature Engineering

### Financial Ratios Created

| Feature | Formula | Business Meaning |
|---------|---------|------------------|
| CREDIT_INCOME_RATIO | Credit / Income | Debt burden |
| ANNUITY_INCOME_RATIO | Annuity / Income | Monthly payment burden |
| CREDIT_TERM | Credit / Annuity | Loan duration |
| CREDIT_GOODS_RATIO | Credit / Goods Price | Loan-to-value ratio |
| INCOME_PER_PERSON | Income / Family Size | Household wealth |

### Top 10 Most Important Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | EXT_SOURCE_MEAN | 9.07 |
| 2 | EXT_SOURCE_WEIGHTED | 3.15 |
| 3 | CREDIT_TERM | 2.79 |
| 4 | CREDIT_TERM_SQ | 2.71 |
| 5 | EXT_SOURCE_MIN | 2.58 |
| 6 | POS_INSTALMENT_FUTURE_MEAN | 2.29 |
| 7 | CREDIT_GOODS_RATIO | 2.06 |
| 8 | DAYS_BIRTH (Age) | 1.93 |
| 9 | INS_LATE_RATIO | 1.80 |
| 10 | AMT_ANNUITY | 1.76 |

### SHAP Analysis - Key Drivers

| Feature | SHAP Impact | Interpretation |
|---------|-------------|----------------|
| FLAG_DOCUMENT_15 | 6.42 | Missing documents = higher risk |
| REG_REGION_NOT_LIVE_REGION | 4.08 | Address mismatch = higher risk |
| AMT_REQ_CREDIT_BUREAU_WEEK | 2.96 | Recent credit inquiries = higher risk |
| NAME_EDUCATION_TYPE | 2.67 | Higher education = lower risk |
| ANNUITY_INCOME_RATIO | 2.56 | High payment/income = higher risk |

---

## 🤖 Model Development

### Models Trained

| Model | AUC | Training Time |
|-------|-----|---------------|
| **CatBoost (Best)** | **0.786** | ~5 min |
| LightGBM | 0.782 | ~3 min |
| XGBoost | 0.778 | ~4 min |
| Random Forest | 0.765 | ~10 min |
| Logistic Regression | 0.742 | ~1 min |

### Hyperparameter Optimization

**Method:** Optuna (Bayesian Optimization)  
**Trials:** 100  
**Search Space:**
- learning_rate: [0.01, 0.3]
- depth: [4, 10]
- l2_leaf_reg: [1, 10]
- iterations: [100, 1000]

### Final Model Configuration

```python
CatBoostClassifier(
    iterations=800,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=False
)
```

---

## 🖥️ API Endpoints

### Base URL
- **Local:** `http://localhost:8000`
- **Docs:** `http://localhost:8000/docs` (Swagger UI)

### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/info` | GET | Model information |
| `/features` | GET | Feature importance |
| `/threshold` | GET | Optimal threshold config |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/explain/{client_id}` | GET | SHAP explanation |

### Example Prediction Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "AMT_INCOME_TOTAL": 150000,
      "AMT_CREDIT": 500000,
      "AMT_ANNUITY": 25000,
      "DAYS_BIRTH": -12000,
      "CODE_GENDER": 1
    }
  }'
```

### Example Response

```json
{
  "probability": 0.23,
  "prediction": 0,
  "risk_level": "MEDIUM",
  "threshold_used": 0.35,
  "message": "Loan recommended with standard monitoring"
}
```

---

## 🎨 Frontend Dashboard

### Features

1. **Home Dashboard**
   - Model status and metrics
   - Quick prediction form
   - Risk gauge visualization

2. **Single Prediction**
   - Input client features
   - Real-time risk assessment
   - SHAP waterfall explanation

3. **Batch Prediction**
   - Upload CSV file
   - Process multiple clients
   - Download results

4. **Feature Importance**
   - Interactive bar charts
   - SHAP summary plots
   - Feature correlation matrix

5. **Model Information**
   - Performance metrics
   - Threshold configuration
   - API health status

---

## 🐳 Docker Deployment

### Quick Start

```bash
# Clone and enter project
cd ~/systeme-prediction-defaut-paiement

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| api | 8000 | FastAPI backend |
| frontend | 8501 | Streamlit dashboard |
| nginx | 80 | Reverse proxy (production) |

### Resource Limits

| Service | CPU | Memory |
|---------|-----|--------|
| API | 2 cores | 2GB |
| Frontend | 1 core | 1GB |

---

## 🌐 Current Deployment

### Live URLs

| Service | URL |
|---------|-----|
| **Dashboard** | https://ace-essence-picked-alone.trycloudflare.com |
| **API (local)** | http://localhost:8000 |
| **API Docs** | http://localhost:8000/docs |

---

## 📁 Project Structure

```
systeme-prediction-defaut-paiement/
│
├── api/
│   ├── app.py              # FastAPI v1
│   └── app_v2.py           # FastAPI v2 (production)
│
├── code/
│   ├── main.py             # Pipeline orchestrator
│   ├── exploration_donnees.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   ├── training.py
│   └── model_comparison.py
│
├── frontend/
│   └── app.py              # Streamlit dashboard
│
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.frontend
│   └── nginx.conf
│
├── models/
│   ├── best_model.joblib
│   ├── ultimate_model.joblib
│   ├── feature_importance.csv
│   ├── threshold_config.json
│   └── ...
│
├── data/
│   └── feature_matrix.csv  # 307K rows × 167 features
│
├── plots/                  # Generated visualizations
├── mlruns/                 # MLflow experiments
│
├── docker-compose.yml
├── config.py
├── requirements.txt
└── README.md
```

---

## 📈 Future Improvements

| Priority | Improvement | Status |
|----------|-------------|--------|
| High | Real-time model monitoring | ⏳ Planned |
| High | A/B testing framework | ⏳ Planned |
| Medium | Model versioning (MLflow) | ✅ Done |
| Medium | Fairness analysis | ⏳ Planned |
| Low | Multi-model ensemble | ✅ Done |
| Low | GPU training support | ✅ Done |

---

## 👨‍💻 Team

- **Original Author:** Yomna Haouel
- **Enhancements & Deployment:** 7afnawi
- **For:** Hefny (Seif)

---

## 📄 License

Academic project for IT Engineering program.

---

*Report generated: March 10, 2026*
