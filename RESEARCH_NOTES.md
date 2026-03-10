# Research Notes: Credit Default Prediction System 📚

> Team reference guide for key concepts, best practices, and pitfalls.

---

## Table of Contents
1. [Class Imbalance Handling](#1-class-imbalance-handling)
2. [Gradient Boosting Models](#2-gradient-boosting-models)
3. [Evaluation Metrics for Credit Scoring](#3-evaluation-metrics-for-credit-scoring)
4. [Feature Aggregation Best Practices](#4-feature-aggregation-best-practices)
5. [Quick Reference: Hyperparameters](#5-quick-reference-hyperparameters)

---

## 1. Class Imbalance Handling

### The Problem
Credit default datasets are inherently imbalanced—typically 2-10% default rate. Models trained on imbalanced data tend to predict the majority class (non-default), achieving high accuracy but missing actual defaults.

### SMOTE (Synthetic Minority Over-sampling Technique)

**What it does:** Creates synthetic samples of the minority class by interpolating between existing minority samples and their k-nearest neighbors.

**How it works:**
1. Select a minority class sample
2. Find its k nearest neighbors (default k=5)
3. Randomly select one neighbor
4. Create synthetic sample along the line connecting them: `x_new = x + rand(0,1) * (x_neighbor - x)`

**Variants we should consider:**
| Variant | Use Case |
|---------|----------|
| **SMOTE** | Basic, works well for continuous features |
| **SMOTE-NC** | When you have categorical features (NC = Nominal Continuous) |
| **BorderlineSMOTE** | Focuses on samples near decision boundary |
| **ADASYN** | Adaptive—generates more samples in harder-to-learn regions |

**⚠️ Pitfalls to Avoid:**
- **Never apply SMOTE before train/test split!** → Data leakage. Apply only to training set.
- Don't oversample to 50/50—often 30-40% minority is enough
- SMOTE doesn't work well with very high-dimensional sparse data
- Synthetic samples can introduce noise if original data has outliers

**Best Practice:**
```python
# Correct pipeline
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

pipeline = Pipeline([
    ('smote', SMOTE(sampling_strategy=0.5, random_state=42)),
    ('classifier', XGBClassifier())
])
# Use this in cross-validation
```

### Alternative Strategies

| Strategy | Pros | Cons |
|----------|------|------|
| **Class weights** | No synthetic data, fast | Less effective for severe imbalance |
| **Undersampling** | Fast, avoids overfitting | Loses information |
| **SMOTE + Tomek** | Cleaner decision boundary | Slower |
| **Cost-sensitive learning** | Directly optimizes business cost | Requires cost matrix |

**Our Recommendation:** Start with `scale_pos_weight` in XGBoost (= n_negative/n_positive), then try SMOTE if results need improvement.

---

## 2. Gradient Boosting Models

### Core Concept
All three (XGBoost, LightGBM, CatBoost) are gradient boosting frameworks—they build an ensemble of weak learners (decision trees) sequentially, where each tree corrects the errors of previous ones.

### XGBoost (eXtreme Gradient Boosting)

**Key Features:**
- Uses second-order gradients (Hessian) for optimization
- Built-in L1/L2 regularization
- Handles missing values natively
- Tree pruning via `max_depth` and `gamma`

**Strengths:** Most mature, extensive documentation, great for structured data

**When to use:** Default choice; when you need explainability tools (SHAP integration)

### LightGBM (Light Gradient Boosting Machine)

**Key Features:**
- **Leaf-wise growth** (vs level-wise)—faster, can overfit more
- Histogram-based algorithm—bins continuous features
- Native categorical feature support
- Gradient-based One-Side Sampling (GOSS)

**Strengths:** Fastest training, handles large datasets well, memory efficient

**When to use:** Large datasets (>100K rows), need fast iteration

**⚠️ Warning:** More prone to overfitting on small datasets due to leaf-wise growth

### CatBoost (Categorical Boosting)

**Key Features:**
- **Ordered boosting**—reduces prediction shift/overfitting
- Native categorical encoding (no one-hot needed)
- Symmetric trees (faster inference)
- Built-in handling of missing values

**Strengths:** Best out-of-box performance, handles categoricals elegantly

**When to use:** Many categorical features, limited tuning time available

### Model Comparison

| Aspect | XGBoost | LightGBM | CatBoost |
|--------|---------|----------|----------|
| Speed | Medium | Fast | Slow to train |
| Accuracy | High | High | Highest (often) |
| Categoricals | Manual encoding | `categorical_feature` | Native (best) |
| Overfitting risk | Medium | Higher | Lower |
| Missing values | ✅ | ✅ | ✅ |
| GPU support | ✅ | ✅ | ✅ |

**Our Approach:** Train all three, ensemble the best performers.

---

## 3. Evaluation Metrics for Credit Scoring

### Why Accuracy Fails
With 5% default rate, predicting "no default" always gives 95% accuracy but catches zero defaults. **Useless for credit scoring.**

### Key Metrics Explained

#### ROC-AUC (Area Under ROC Curve)
**What it measures:** Model's ability to distinguish between classes across all thresholds.

**Range:** 0.5 (random) to 1.0 (perfect)

**Credit Scoring Interpretation:**
- 0.5-0.6: Poor, barely better than random
- 0.6-0.7: Fair, some discrimination
- 0.7-0.8: Good, acceptable for credit scoring
- 0.8-0.9: Excellent
- 0.9+: Outstanding (verify you're not leaking data!)

**Pros:** Threshold-independent, works well for imbalanced data
**Cons:** Doesn't tell you what happens at specific threshold, can be optimistic

#### Precision
**Formula:** `TP / (TP + FP)`

**Credit Question:** "Of all loans we flagged as risky, how many actually defaulted?"

**Business Impact:** Low precision = rejecting good customers (lost revenue)

#### Recall (Sensitivity)
**Formula:** `TP / (TP + FN)`

**Credit Question:** "Of all actual defaults, how many did we catch?"

**Business Impact:** Low recall = approving bad loans (direct losses)

#### F1 Score
**Formula:** `2 * (Precision * Recall) / (Precision + Recall)`

**When to use:** When you need a single metric balancing precision and recall

**⚠️ Note:** Equal weight to precision/recall may not match business needs

### The Precision-Recall Trade-off in Credit Scoring

```
High Threshold (conservative):
  ↑ Precision, ↓ Recall
  → Fewer false alarms, but miss more defaults
  → Good when capital is limited
  
Low Threshold (aggressive):
  ↓ Precision, ↑ Recall
  → Catch more defaults, but more false alarms
  → Good when default cost >> rejection cost
```

### Recommended Metrics for Our Project

| Metric | Why |
|--------|-----|
| **ROC-AUC** | Primary ranking metric, compare models |
| **PR-AUC** | Better for imbalanced data than ROC-AUC |
| **Recall @ Fixed FPR** | "Catch rate at 5% false positive rate" |
| **Precision @ Top K%** | "Precision in riskiest 10% of applicants" |

### Business Cost Matrix Approach
```
                    Predicted
                 No Default | Default
Actual No Default    TN     |   FP (reject good customer: -$500)
Actual Default       FN     |   TP (prevented loss: +$10,000)
                (bad loan: -$10,000)
```

**Cost-sensitive threshold:** Optimize threshold for total expected cost, not just F1.

---

## 4. Feature Aggregation Best Practices

### Types of Aggregations

For transactional/behavioral data, aggregate over time windows:

| Aggregation | Use Case |
|-------------|----------|
| **count** | Transaction frequency |
| **sum** | Total spending |
| **mean** | Average transaction size |
| **std** | Spending volatility |
| **min/max** | Extremes |
| **last** | Most recent behavior |
| **trend** | Slope over time |

### Time Windows
- Short: 7, 14, 30 days (recent behavior)
- Medium: 60, 90 days (patterns)
- Long: 180, 365 days (stability)

### ⚠️ Pitfalls to Avoid

1. **Future leakage:** Never aggregate future data relative to prediction point
2. **Missing window handling:** Decide strategy (0, NaN, or imputation)
3. **Infinite values:** std=0 can cause issues when computing ratios
4. **Feature explosion:** 50 features × 5 aggregations × 4 windows = 1000 features

### Smart Aggregation Features for Credit
```python
# Examples of predictive features
"days_since_last_payment"
"payment_to_balance_ratio_30d"
"num_missed_payments_90d"
"spending_trend_90d"  # slope
"max_utilization_30d"
"payment_volatility_90d"  # std of payment amounts
```

---

## 5. Quick Reference: Hyperparameters

### XGBoost
```python
{
    'n_estimators': [100, 300, 500],      # Start lower, increase with early stopping
    'max_depth': [3, 5, 7],               # Keep shallow to prevent overfitting
    'learning_rate': [0.01, 0.05, 0.1],   # Lower = more trees needed
    'subsample': [0.7, 0.8, 0.9],         # Row sampling
    'colsample_bytree': [0.7, 0.8, 0.9],  # Column sampling
    'min_child_weight': [1, 3, 5],        # Regularization
    'scale_pos_weight': [ratio],          # n_neg / n_pos for imbalance
    'gamma': [0, 0.1, 0.2],               # Min loss reduction for split
}
```

### LightGBM
```python
{
    'n_estimators': [100, 300, 500],
    'num_leaves': [31, 63, 127],          # Main control for complexity
    'max_depth': [-1, 10, 20],            # -1 = no limit
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_samples': [20, 50, 100],   # Regularization
    'reg_alpha': [0, 0.1, 1],             # L1
    'reg_lambda': [0, 0.1, 1],            # L2
    'is_unbalance': [True],               # OR use scale_pos_weight
}
```

### CatBoost
```python
{
    'iterations': [300, 500, 1000],
    'depth': [4, 6, 8],                   # Usually 6-10
    'learning_rate': [0.01, 0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5, 7],          # Regularization
    'border_count': [32, 64, 128],        # Splits for numerical
    'auto_class_weights': ['Balanced'],   # For imbalance
    'random_strength': [0, 0.5, 1],       # Randomness for splits
}
```

### Early Stopping (Use This!)
```python
# Always use early stopping to prevent overfitting
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,  # Stop if no improvement in 50 rounds
    verbose=100
)
```

---

## Quick Decision Guide

```
Dataset has many categoricals?
  → CatBoost (or LightGBM with categorical_feature)

Dataset is large (>500K rows)?
  → LightGBM first (fastest)

Need best accuracy, time not critical?
  → Train all three, ensemble

Severe class imbalance (<2% positive)?
  → SMOTE-NC + scale_pos_weight + threshold tuning

Which metric to optimize?
  → ROC-AUC for model selection
  → Business cost matrix for threshold selection
```

---

## References

- [SMOTE Paper](https://arxiv.org/abs/1106.1813)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/docs/)
- [Imbalanced-learn](https://imbalanced-learn.org/)

---

*Last updated: 2026-03-05 | Agent-Researcher 📚*
