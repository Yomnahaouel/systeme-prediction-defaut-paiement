"""
app.py — Streamlit Dashboard for Credit Risk Prediction

Interactive dashboard for:
- Single client predictions
- Batch predictions
- Model explanations (SHAP)
- Feature importance visualization

Author: 7afnawi for Hefny
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
import os
from typing import Dict

# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-low { color: #4CAF50; font-weight: bold; }
    .risk-medium { color: #FF9800; font-weight: bold; }
    .risk-high { color: #f44336; font-weight: bold; }
    .stAlert { margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════

def check_api_health() -> Dict:
    """Check API health status."""
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None


def get_model_info() -> Dict:
    """Get model information from API."""
    try:
        response = requests.get(f"{API_URL}/info", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None


def get_feature_importance(top_n: int = 20) -> pd.DataFrame:
    """Get feature importance from API."""
    try:
        response = requests.get(f"{API_URL}/features?top_n={top_n}", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data['features'])
        return None
    except:
        return None


def predict_single(features: Dict) -> Dict:
    """Make single prediction."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": features},
            timeout=10
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None


def explain_prediction(features: Dict) -> Dict:
    """Get SHAP explanation for prediction."""
    try:
        response = requests.post(
            f"{API_URL}/explain",
            json={"features": features},
            timeout=15
        )
        return response.json() if response.status_code == 200 else None
    except:
        return None


def get_risk_color(risk_level: str) -> str:
    """Get color for risk level."""
    colors = {
        "LOW": "#4CAF50",
        "LOW-MEDIUM": "#8BC34A",
        "MEDIUM": "#FF9800",
        "HIGH": "#f44336",
        "VERY HIGH": "#B71C1C"
    }
    return colors.get(risk_level, "#757575")


# ══════════════════════════════════════════════════════════════════════
# Main App
# ══════════════════════════════════════════════════════════════════════

def main():
    # Header
    st.markdown('<h1 class="main-header">🏦 Credit Risk Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/bank-building.png", width=80)
        st.title("Navigation")
        
        page = st.radio(
            "Select Page",
            ["🎯 Predict", "📊 Feature Importance", "ℹ️ Model Info"],
            index=0
        )
        
        st.markdown("---")
        
        # API Status
        health = check_api_health()
        if health:
            st.success("✅ API Connected")
            st.metric("Threshold", f"{health.get('threshold', 0.5):.2f}")
        else:
            st.error("❌ API Disconnected")
            st.info(f"API URL: {API_URL}")
    
    # ════════════════════════════════════════════════════════════════════
    # Page: Predict
    # ════════════════════════════════════════════════════════════════════
    if page == "🎯 Predict":
        st.header("Client Default Risk Prediction")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📝 Enter Client Information")
            
            # Key features input
            with st.form("prediction_form"):
                st.markdown("**External Source Scores** (0-1 scale)")
                ext1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, 0.5, 0.01)
                ext2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, 0.5, 0.01)
                ext3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, 0.5, 0.01)
                
                st.markdown("**Financial Information**")
                income = st.number_input("Annual Income (AMT_INCOME_TOTAL)", 
                                        min_value=0, max_value=10000000, value=150000, step=10000)
                credit = st.number_input("Credit Amount (AMT_CREDIT)", 
                                        min_value=0, max_value=10000000, value=500000, step=10000)
                annuity = st.number_input("Annuity Amount (AMT_ANNUITY)", 
                                         min_value=0, max_value=500000, value=25000, step=1000)
                
                st.markdown("**Personal Information**")
                age = st.slider("Age (years)", 18, 80, 35)
                employment = st.slider("Years Employed", 0, 40, 5)
                
                submitted = st.form_submit_button("🔮 Predict Risk", use_container_width=True)
        
        with col2:
            if submitted:
                # Prepare features
                features = {
                    "EXT_SOURCE_1": ext1,
                    "EXT_SOURCE_2": ext2,
                    "EXT_SOURCE_3": ext3,
                    "AMT_INCOME_TOTAL": income,
                    "AMT_CREDIT": credit,
                    "AMT_ANNUITY": annuity,
                    "DAYS_BIRTH": -age * 365,
                    "DAYS_EMPLOYED": -employment * 365 if employment > 0 else 365243,
                    "EXT_SOURCE_MEAN": (ext1 + ext2 + ext3) / 3,
                    "CREDIT_INCOME_RATIO": credit / income if income > 0 else 0,
                    "ANNUITY_INCOME_RATIO": annuity / income if income > 0 else 0,
                }
                
                with st.spinner("Analyzing..."):
                    result = predict_single(features)
                
                if result:
                    st.subheader("📊 Prediction Result")
                    
                    # Gauge chart for probability
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=result['default_probability'] * 100,
                        title={'text': "Default Probability", 'font': {'size': 24}},
                        delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1},
                            'bar': {'color': get_risk_color(result['risk_level'])},
                            'steps': [
                                {'range': [0, 20], 'color': "#E8F5E9"},
                                {'range': [20, 40], 'color': "#FFF3E0"},
                                {'range': [40, 60], 'color': "#FFECB3"},
                                {'range': [60, 80], 'color': "#FFCDD2"},
                                {'range': [80, 100], 'color': "#FFCDD2"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': result['threshold'] * 100
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Metrics
                    mcol1, mcol2, mcol3 = st.columns(3)
                    with mcol1:
                        st.metric("Risk Level", result['risk_level'])
                    with mcol2:
                        st.metric("Prediction", "DEFAULT" if result['prediction'] == 1 else "NO DEFAULT")
                    with mcol3:
                        st.metric("Confidence", result.get('confidence', 'N/A'))
                    
                    # Risk interpretation
                    if result['prediction'] == 1:
                        st.error(f"⚠️ **HIGH RISK CLIENT** — Default probability: {result['default_probability']:.1%}")
                    else:
                        st.success(f"✅ **LOW RISK CLIENT** — Default probability: {result['default_probability']:.1%}")
                    
                    # Explanation
                    st.markdown("---")
                    st.subheader("🔍 SHAP Explanation")
                    
                    with st.spinner("Generating explanation..."):
                        explanation = explain_prediction(features)
                    
                    if explanation:
                        st.info(explanation.get('risk_summary', 'No summary available'))
                        
                        if explanation.get('top_positive_factors'):
                            st.markdown("**Risk Increasing Factors:**")
                            for f in explanation['top_positive_factors'][:3]:
                                st.markdown(f"- 📈 {f['feature']}: +{f['impact']:.4f}")
                        
                        if explanation.get('top_negative_factors'):
                            st.markdown("**Risk Decreasing Factors:**")
                            for f in explanation['top_negative_factors'][:3]:
                                st.markdown(f"- 📉 {f['feature']}: {f['impact']:.4f}")
    
    # ════════════════════════════════════════════════════════════════════
    # Page: Feature Importance
    # ════════════════════════════════════════════════════════════════════
    elif page == "📊 Feature Importance":
        st.header("Feature Importance Analysis")
        
        top_n = st.slider("Number of features to display", 10, 50, 20)
        
        with st.spinner("Loading feature importance..."):
            importance_df = get_feature_importance(top_n)
        
        if importance_df is not None:
            # Bar chart
            fig = px.bar(
                importance_df,
                x='importance',
                y='feature',
                orientation='h',
                title=f"Top {top_n} Most Important Features (SHAP)",
                labels={'importance': 'Mean |SHAP Value|', 'feature': 'Feature'},
                color='importance',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=max(400, top_n * 25), yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Table
            st.subheader("📋 Feature Importance Table")
            st.dataframe(importance_df, use_container_width=True)
        else:
            st.warning("Feature importance data not available. Run SHAP analysis first.")
    
    # ════════════════════════════════════════════════════════════════════
    # Page: Model Info
    # ════════════════════════════════════════════════════════════════════
    elif page == "ℹ️ Model Info":
        st.header("Model Information")
        
        info = get_model_info()
        
        if info:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Model Type", info.get('model_type', 'N/A'))
            with col2:
                auc = info.get('model_auc')
                st.metric("ROC-AUC", f"{auc:.4f}" if auc else "N/A")
            with col3:
                st.metric("Features", info.get('n_features', 0))
            
            st.markdown("---")
            
            st.subheader("🎯 Optimal Threshold")
            st.info(f"The model uses a business-optimized threshold of **{info.get('optimal_threshold', 0.5):.3f}** to balance false positives (rejecting good customers) and false negatives (approving bad loans).")
            
            st.subheader("📊 Risk Levels")
            risk_levels = info.get('risk_levels', [])
            for level in risk_levels:
                color = get_risk_color(level)
                st.markdown(f'<span style="color:{color}; font-weight:bold;">● {level}</span>', unsafe_allow_html=True)
            
            st.subheader("🔢 Sample Features")
            st.code(", ".join(info.get('sample_features', [])))
        else:
            st.error("Could not connect to API")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "🏦 Credit Risk Prediction System v2.0 | Built by 7afnawi for Hefny"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
