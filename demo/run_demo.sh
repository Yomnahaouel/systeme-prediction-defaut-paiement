#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Credit Risk Prediction System - DEMO SCRIPT
# ═══════════════════════════════════════════════════════════════

clear
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     🏦 CREDIT RISK PREDICTION SYSTEM - LIVE DEMO 🏦          ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
sleep 2

# 1. Health Check
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📡 1. API HEALTH CHECK"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -s http://localhost:8000/ | python3 -m json.tool
echo ""
sleep 3

# 2. Model Info
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🤖 2. MODEL INFORMATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -s http://localhost:8000/info | python3 -m json.tool
echo ""
sleep 3

# 3. Low Risk Client
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 3. PREDICTING LOW RISK CLIENT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Client Profile:"
echo "  • External Score 1: 0.7 (High)"
echo "  • External Score 2: 0.8 (High)"
echo "  • External Score 3: 0.6 (Good)"
echo "  • Income: 300,000"
echo "  • Credit Amount: 200,000"
echo "  • Age: 40 years"
echo ""
echo "Prediction:"
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"EXT_SOURCE_1": 0.7, "EXT_SOURCE_2": 0.8, "EXT_SOURCE_3": 0.6, "AMT_CREDIT": 200000, "AMT_INCOME_TOTAL": 300000, "DAYS_BIRTH": -14600}}' | python3 -m json.tool
echo ""
sleep 3

# 4. High Risk Client
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚠️  4. PREDICTING HIGH RISK CLIENT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Client Profile:"
echo "  • External Score 1: 0.1 (Very Low)"
echo "  • External Score 2: 0.15 (Very Low)"
echo "  • External Score 3: 0.1 (Very Low)"
echo "  • Income: 50,000"
echo "  • Credit Amount: 900,000 (High debt ratio!)"
echo "  • Age: 22 years (Young)"
echo ""
echo "Prediction:"
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"EXT_SOURCE_1": 0.1, "EXT_SOURCE_2": 0.15, "EXT_SOURCE_3": 0.1, "AMT_CREDIT": 900000, "AMT_INCOME_TOTAL": 50000, "DAYS_BIRTH": -8000}}' | python3 -m json.tool
echo ""
sleep 3

# 5. Batch Prediction
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 5. BATCH PREDICTION (3 CLIENTS)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -s -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"clients": [
    {"EXT_SOURCE_2": 0.9, "AMT_CREDIT": 100000, "AMT_INCOME_TOTAL": 200000},
    {"EXT_SOURCE_2": 0.5, "AMT_CREDIT": 400000, "AMT_INCOME_TOTAL": 150000},
    {"EXT_SOURCE_2": 0.2, "AMT_CREDIT": 800000, "AMT_INCOME_TOTAL": 60000}
  ]}' | python3 -m json.tool
echo ""
sleep 3

# 6. Top Features
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📈 6. TOP 5 IMPORTANT FEATURES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -s "http://localhost:8000/features?top_n=5" | python3 -m json.tool
echo ""
sleep 3

# 7. Threshold Info
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚖️  7. THRESHOLD CONFIGURATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -s http://localhost:8000/threshold | python3 -m json.tool
echo ""
sleep 2

# Summary
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                    ✅ DEMO COMPLETE ✅                        ║"
echo "╠═══════════════════════════════════════════════════════════════╣"
echo "║  Dashboard:  http://localhost:8501                            ║"
echo "║  API Docs:   http://localhost:8000/docs                       ║"
echo "║  Model:      CatBoost (AUC: 0.786)                            ║"
echo "║  Threshold:  0.35 (Business Optimized)                        ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
