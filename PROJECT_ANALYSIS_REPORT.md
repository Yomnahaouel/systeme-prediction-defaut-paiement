# 📊 Credit Default Risk Prediction — Project Analysis Report

**Analyzed by:** 7afnawi  
**Date:** March 5, 2026  
**Project Author:** Yomna Haouel  

---

## 🎯 Executive Summary

This is a **complete ML pipeline** for predicting credit default risk using the Home Credit dataset from Kaggle. The project follows industry best practices including anti data-leakage, SMOTE for imbalance handling, MLflow tracking, and FastAPI deployment.

**Goal:** Predict whether a client will default on a loan (binary classification).

**Current Status:**
- ✅ Code: Complete and well-structured
- ✅ Dependencies: Installed
- ⏳ Dataset: Downloading (688MB)
- ❌ Trained Model: Not yet (needs dataset)
- ❌ API: Can't run without model

---

## 📁 Project Structure

```
systeme-prediction-defaut-paiement/
│
├── config.py                    # Central configuration (paths, params, grids)
├── requirements.txt             # Python dependencies
├── README.md                    # Documentation
├── generate_report.py           # PDF report generator
│
├── code/                        # ML Pipeline modules
│   ├── main.py                  # 🎯 ENTRY POINT — Orchestrator
│   ├── exploration_donnees.py   # Step 1: EDA
│   ├── preprocessing.py         # Step 2: Clean + Split + Transform
│   ├── feature_engineering.py   # Step 3: Create financial ratios
│   ├── feature_selection.py     # Step 4: Select best features
│   ├── training.py              # Step 5: Train 5 models
│   └── model_comparison.py      # Step 6: Compare & select best
│
├── api/
│   └── app.py                   # FastAPI deployment endpoint
│
├── data/                        # Dataset folder
│   └── feature_matrix.csv       # Required (downloading now)
│
├── models/                      # Saved models (.joblib)
├── plots/                       # Generated visualizations
│   ├── eda/
│   ├── training/
│   └── comparison/
│
└── mlruns/                      # MLflow experiment tracking
```

---

## 🔄 Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MAIN PIPELINE                                 │
│                     (python code/main.py)                           │
└─────────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
   ┌─────────┐            ┌─────────┐            ┌─────────┐
   │  EDA    │            │ Preproc │            │ Feature │
   │(optional)│──────────▶│         │──────────▶│   Eng   │
   └─────────┘            └─────────┘            └─────────┘
                                │                       │
                                ▼                       ▼
                          ┌─────────┐            ┌─────────┐
                          │ Feature │            │ Training│
                          │  Select │◀───────────│ 5 models│
                          └─────────┘            └─────────┘
                                │                       │
                                ▼                       ▼
                          ┌─────────┐            ┌─────────┐
                          │ Compare │            │  Save   │
                          │& Select │──────────▶│  Best   │
                          └─────────┘            └─────────┘
                                                        │
                                                        ▼
                                                  ┌─────────┐
                                                  │ FastAPI │
                                                  │   API   │
                                                  └─────────┘
```

---

## 📋 Step-by-Step Breakdown

### Step 1: EDA (`exploration_donnees.py`)
**Purpose:** Understand the data before modeling

What it does:
- Distribution of TARGET (default rate ~8%)
- Missing value analysis
- Outlier detection (IQR method)
- Feature correlations with TARGET
- Generates plots in `plots/eda/`

**Run:** `python code/main.py --eda` or `--eda-only`

---

### Step 2: Preprocessing (`preprocessing.py`)
**Purpose:** Clean data and prepare for ML

Key operations:
1. Load `data/feature_matrix.csv`
2. Keep only training rows (TARGET != null)
3. Drop columns with >60% missing values
4. **STRATIFIED SPLIT FIRST** (anti data-leakage!)
5. Apply sklearn Pipeline:
   - Numeric: `SimpleImputer(median)` → `StandardScaler`
   - Categorical: `SimpleImputer(mode)` → `OrdinalEncoder`

**Output:** `X_train`, `X_test`, `y_train`, `y_test`

---

### Step 3: Feature Engineering (`feature_engineering.py`)
**Purpose:** Create business-meaningful features

| Feature | Formula | Business Meaning |
|---------|---------|------------------|
| CREDIT_INCOME_RATIO | Credit / Income | Debt capacity |
| ANNUITY_INCOME_RATIO | Annuity / Income | Monthly financial burden |
| CREDIT_TERM | Credit / Annuity | Estimated loan duration |
| INCOME_PER_PERSON | Income / Family Size | Per-capita income |
| AGE_YEARS | -DAYS_BIRTH / 365.25 | Client age |
| EMPLOYMENT_YEARS | -DAYS_EMPLOYED / 365.25 | Job tenure |
| EMPLOYED_ANOMALY | DAYS_EMPLOYED == 365243 | Unemployed/retired flag |
| EXT_SOURCE_MEAN | avg(EXT_1, EXT_2, EXT_3) | Aggregated credit score |
| EXT_SOURCE_1x2 | EXT_1 × EXT_2 | Non-linear interaction |
| DOCUMENTS_PROVIDED | sum(FLAG_DOCUMENT_*) | Application completeness |
| REGION_MISMATCH_SUM | sum(REG_*_NOT_*) | Geographic stability |

---

### Step 4: Feature Selection (`feature_selection.py`)
**Purpose:** Remove noise, keep predictive features

Three-stage process:
1. **Correlation filter** (threshold 0.85) — removes redundant features
2. **Mutual Information** (top 120) — statistical relevance
3. **LightGBM importance** (threshold 0.001) — non-linear relevance

Strategy: Union or Intersection (configurable in `config.py`)

---

### Step 5: Training (`training.py`)
**Purpose:** Train and evaluate multiple models

| Model | Imbalance Strategy | Speed |
|-------|-------------------|-------|
| Logistic Regression | class_weight='balanced' | ⚡ Fast |
| Random Forest | class_weight='balanced' | 🔄 Medium |
| XGBoost | scale_pos_weight=10 | ⚡ Fast |
| LightGBM | is_unbalance=True | ⚡ Very Fast |
| CatBoost | auto_class_weights='Balanced' | 🔄 Medium |

- **SMOTE** applied on train set only (no leakage)
- Cross-validation with 5 folds
- Metrics: ROC-AUC, Recall, Precision, F1
- MLflow tracking for experiment management
- Models saved as `.joblib` files

---

### Step 6: Model Comparison (`model_comparison.py`)
**Purpose:** Select the best model objectively

Selection criteria (weighted):
- 50% ROC-AUC
- 30% Recall (important for detecting defaults!)
- 20% F1-score

Outputs:
- Comparison table
- Combined ROC curves plot
- SHAP feature importance analysis
- `best_model.joblib` saved for deployment

---

### Step 7: API Deployment (`api/app.py`)
**Purpose:** Serve predictions via REST API

Endpoints:
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/info` | Model info & expected features |
| POST | `/predict` | Predict for a client |

**Example request:**
```json
POST /predict
{
  "features": {
    "AMT_CREDIT": 500000.0,
    "AMT_INCOME_TOTAL": 150000.0,
    "EXT_SOURCE_1": 0.5,
    "EXT_SOURCE_2": 0.6
  }
}
```

**Response:**
```json
{
  "default_probability": 0.1234,
  "prediction": 0,
  "risk_level": "LOW"
}
```

---

## ⚙️ Configuration (`config.py`)

All parameters are centralized:

```python
# Key settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
MISSING_THRESHOLD = 0.60

# Feature Selection
CORRELATION_THRESHOLD = 0.85
MI_TOP_K = 120
IMPORTANCE_THRESHOLD = 0.001

# SMOTE
SMOTE_SAMPLING_STRATEGY = 0.5

# Models to train
MODELS_TO_TRAIN = [
    "LogisticRegression",
    "RandomForest", 
    "XGBoost",
    "LightGBM",
    "CatBoost",
]
```

---

## 🚀 How to Run the Project

### 1. Full Pipeline (without EDA)
```bash
cd ~/systeme-prediction-defaut-paiement
source venv/bin/activate
python code/main.py
```

### 2. Full Pipeline with EDA
```bash
python code/main.py --eda
```

### 3. EDA Only
```bash
python code/main.py --eda-only
```

### 4. Run the API
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
# Open: http://localhost:8000/docs
```

### 5. View MLflow Dashboard
```bash
mlflow ui --port 5000
# Open: http://localhost:5000
```

---

## 📊 Expected Outputs After Running

### Models Generated:
- `models/LogisticRegression.joblib`
- `models/RandomForest.joblib`
- `models/XGBoost.joblib`
- `models/LightGBM.joblib`
- `models/CatBoost.joblib`
- `models/best_model.joblib` ← Used by API
- `models/selected_features.json`
- `models/preprocessing_pipeline.joblib`

### Plots Generated:
- `plots/eda/` — EDA visualizations
- `plots/training/cm_*.png` — Confusion matrices
- `plots/training/roc_*.png` — ROC curves
- `plots/comparison/` — Model comparison charts

---

## 🎯 What We Need to Do

### Immediate (after download completes):
1. ✅ Wait for dataset download to finish
2. ⏳ Run the pipeline: `python code/main.py --eda`
3. ⏳ Check plots and metrics
4. ⏳ Launch API: `uvicorn api.app:app --reload`
5. ⏳ Test predictions

### The dataset needs preprocessing:
The raw Kaggle data has multiple CSV files. This project expects a pre-processed `feature_matrix.csv`. We may need to:
1. Check if the project includes a data preparation script
2. Or create `feature_matrix.csv` from raw Kaggle files

---

## 💡 Key Takeaways

### Strengths of this project:
- ✅ **Anti data-leakage**: Split BEFORE any transformation
- ✅ **Reproducibility**: RANDOM_STATE=42 everywhere
- ✅ **Configuration**: All params in `config.py`
- ✅ **MLflow**: Experiment tracking built-in
- ✅ **SMOTE**: Handles class imbalance properly
- ✅ **Multiple models**: Compares 5 algorithms
- ✅ **API ready**: FastAPI with Swagger docs
- ✅ **Well documented**: Clear README and docstrings

### Potential improvements:
- Add hyperparameter tuning (GridSearchCV/Optuna)
- Add threshold optimization for business needs
- Add more feature interactions
- Consider ensemble/stacking of best models

---

## 📝 Next Steps Summary

| Step | Action | Status |
|------|--------|--------|
| 1 | Download dataset | ⏳ In progress |
| 2 | Prepare feature_matrix.csv | ⏳ After download |
| 3 | Run full pipeline | ⏳ Pending |
| 4 | Review results | ⏳ Pending |
| 5 | Launch API | ⏳ Pending |
| 6 | Test predictions | ⏳ Pending |

---

*Report generated by 7afnawi — Let's get this model running! 🚀*
