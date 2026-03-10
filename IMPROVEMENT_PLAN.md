# 🚀 Credit Scoring Improvement Plan

**Created:** March 5, 2026 — 17:50 EST  
**Team Lead:** 7afnawi  
**Goal:** Improve model from 0.74 AUC → 0.78+ AUC

---

## 📊 Current Baseline

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.7436 |
| Recall | 71.34% |
| Precision | 14.79% |
| F1-Score | 0.2450 |
| Features Used | 72 (from application_train.csv only) |

---

## 🎯 Target Results

| Metric | Target |
|--------|--------|
| ROC-AUC | 0.78 - 0.80 |
| Recall | 60 - 70% |
| Precision | 30 - 40% |
| F1-Score | 0.40 - 0.50 |

---

## 👥 Agent Team Structure

```
                    7afnawi (Coordinator)
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
     Agent-Bureau    Agent-Payment    Agent-Apps
          │                │                │
          ▼                ▼                ▼
    bureau.csv       credit_card.csv  previous_app.csv
    bureau_balance   installments     POS_CASH_balance
          │                │                │
          ▼                ▼                ▼
    bureau_agg.csv   payment_agg.csv  apps_agg.csv
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                  MERGED feature_matrix.csv
                           │
                           ▼
                    RETRAIN MODELS
                           │
                           ▼
                    IMPROVED RESULTS
```

---

## 📋 Detailed Task Breakdown

### Agent-Bureau: Credit Bureau Features
**Files:** `bureau.csv`, `bureau_balance.csv`

**Features to Create:**
```python
# Per client (SK_ID_CURR) aggregations:
- BUREAU_LOAN_COUNT              # Number of previous credits
- BUREAU_LOAN_TYPES              # Types of credits (consumer, car, mortgage)
- BUREAU_ACTIVE_COUNT            # Currently active credits
- BUREAU_CLOSED_COUNT            # Closed credits
- BUREAU_DAYS_CREDIT_MEAN        # Avg days since credit started
- BUREAU_DAYS_CREDIT_ENDDATE     # Avg days until credit ends
- BUREAU_AMT_CREDIT_SUM          # Total credit amount
- BUREAU_AMT_CREDIT_SUM_DEBT     # Total current debt
- BUREAU_AMT_CREDIT_SUM_OVERDUE  # Total overdue amount
- BUREAU_CNT_CREDIT_PROLONG      # Number of prolongations
- BUREAU_CREDIT_ACTIVE_RATIO     # Active / Total credits

# From bureau_balance:
- BB_MONTHS_BALANCE_COUNT        # Number of monthly records
- BB_STATUS_0_RATIO              # % months with status 0 (paid)
- BB_STATUS_C_RATIO              # % months closed
- BB_STATUS_X_RATIO              # % unknown status
- BB_DPD_MAX                     # Max days past due ever
- BB_DPD_MEAN                    # Average days past due
```

**Expected:** ~20 new features

---

### Agent-Payment: Payment Behavior Features
**Files:** `credit_card_balance.csv`, `installments_payments.csv`

**Features to Create:**
```python
# Credit Card (per client):
- CC_COUNT                       # Number of credit cards
- CC_AMT_BALANCE_MEAN            # Avg balance
- CC_AMT_BALANCE_MAX             # Max balance ever
- CC_AMT_CREDIT_LIMIT_MEAN       # Avg credit limit
- CC_UTILIZATION_MEAN            # Avg balance/limit ratio
- CC_AMT_DRAWINGS_ATM_MEAN       # Avg ATM withdrawals
- CC_AMT_PAYMENT_TOTAL_MEAN      # Avg payments
- CC_CNT_DRAWINGS_MEAN           # Avg number of drawings
- CC_SK_DPD_MAX                  # Max days past due
- CC_SK_DPD_MEAN                 # Avg days past due

# Installments (per client):
- INS_COUNT                      # Number of installment records
- INS_DAYS_ENTRY_PAYMENT_MEAN    # Avg days for payment
- INS_AMT_INSTALMENT_MEAN        # Avg installment amount
- INS_AMT_PAYMENT_MEAN           # Avg payment amount
- INS_PAYMENT_RATIO_MEAN         # Avg payment/installment ratio
- INS_PAYMENT_DIFF_MEAN          # Avg payment - installment
- INS_LATE_PAYMENT_COUNT         # Count of late payments
- INS_LATE_PAYMENT_RATIO         # % late payments
- INS_DAYS_LATE_MEAN             # Avg days late (when late)
- INS_DAYS_EARLY_MEAN            # Avg days early (when early)
```

**Expected:** ~20 new features

---

### Agent-Apps: Previous Applications Features
**Files:** `previous_application.csv`, `POS_CASH_balance.csv`

**Features to Create:**
```python
# Previous Applications (per client):
- PREV_APP_COUNT                 # Total previous applications
- PREV_APP_APPROVED_COUNT        # Approved applications
- PREV_APP_REFUSED_COUNT         # Refused applications
- PREV_APP_APPROVED_RATIO        # Approval rate
- PREV_APP_AMT_APPLICATION_MEAN  # Avg application amount
- PREV_APP_AMT_CREDIT_MEAN       # Avg credit amount
- PREV_APP_AMT_DOWN_PAYMENT_MEAN # Avg down payment
- PREV_APP_DAYS_DECISION_MEAN    # Avg days to decision
- PREV_APP_CNT_PAYMENT_MEAN      # Avg payment count
- PREV_APP_RATE_DOWN_PAYMENT_MEAN # Avg down payment rate

# POS Cash (per client):
- POS_COUNT                      # Number of POS loans
- POS_MONTHS_BALANCE_COUNT       # Total monthly records
- POS_CNT_INSTALMENT_MEAN        # Avg installment count
- POS_SK_DPD_MAX                 # Max days past due
- POS_SK_DPD_MEAN                # Mean days past due
- POS_SK_DPD_DEF_MAX             # Max DPD (defined)
- POS_COMPLETED_RATIO            # % completed contracts
- POS_ACTIVE_RATIO               # % active contracts
```

**Expected:** ~18 new features

---

## 📈 Timeline

| Time | Phase | Details |
|------|-------|---------|
| 17:50 | Start | Create plan, spawn agents |
| 17:50-18:20 | Phase 1 | Agents process data in parallel (~30 min) |
| 18:20-18:30 | Phase 2 | Merge all features (~10 min) |
| 18:30-18:50 | Phase 3 | Retrain models with new features (~20 min) |
| 18:50-19:00 | Phase 4 | Evaluate & compare results (~10 min) |
| **~19:00** | **Done** | **Final results** |

**Total estimated time: ~1 hour 10 minutes**

---

## ✅ Success Criteria

- [ ] 50+ new aggregated features created
- [ ] All data files processed
- [ ] New feature_matrix.csv generated
- [ ] Models retrained
- [ ] ROC-AUC > 0.77
- [ ] F1-Score > 0.35

---

## 📊 Progress Tracking

| Task | Status | Agent | Time |
|------|--------|-------|------|
| Bureau features | ⏳ Pending | Agent-Bureau | - |
| Payment features | ⏳ Pending | Agent-Payment | - |
| Apps features | ⏳ Pending | Agent-Apps | - |
| Merge features | ⏳ Pending | 7afnawi | - |
| Retrain models | ⏳ Pending | 7afnawi | - |
| Final evaluation | ⏳ Pending | 7afnawi | - |

---

*Plan created by 7afnawi — Let's get these results!* 🚀
