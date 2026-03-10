# 🏗️ System Architecture

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CREDIT RISK PREDICTION SYSTEM                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│    ┌──────────────┐         ┌──────────────┐         ┌──────────────┐   │
│    │   NGINX      │         │   FRONTEND   │         │     API      │   │
│    │   (Port 80)  │────────▶│  (Port 8501) │────────▶│  (Port 8000) │   │
│    │   Optional   │         │   Streamlit  │         │   FastAPI    │   │
│    └──────────────┘         └──────────────┘         └──────┬───────┘   │
│                                                              │           │
│                                                              ▼           │
│                              ┌────────────────────────────────────────┐  │
│                              │           ML MODELS (Volume)           │  │
│                              │  • catboost_optimized.joblib (Best)    │  │
│                              │  • LightGBM.joblib                     │  │
│                              │  • XGBoost.joblib                      │  │
│                              │  • threshold_config.json               │  │
│                              │  • feature_importance.csv              │  │
│                              └────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Container Details

### 1. API Container (`credit-risk-api`)
- **Image:** `systeme-prediction-defaut-paiement-api`
- **Port:** 8000
- **Framework:** FastAPI + Uvicorn
- **Size:** ~2.1GB (includes ML libraries)

**Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/info` | GET | Model info |
| `/threshold` | GET | Threshold config |
| `/features` | GET | Feature importance |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/explain` | POST | SHAP explanation |

### 2. Frontend Container (`credit-risk-frontend`)
- **Image:** `systeme-prediction-defaut-paiement-frontend`
- **Port:** 8501
- **Framework:** Streamlit
- **Size:** ~650MB

**Features:**
- Interactive client input form
- Risk gauge visualization
- SHAP explanation display
- Feature importance charts

### 3. Nginx Container (Optional - Production)
- **Image:** `nginx:alpine`
- **Port:** 80 (HTTP), 443 (HTTPS)
- **Purpose:** Reverse proxy, SSL termination, load balancing

## Network Architecture

```
┌─────────────────────────────────────────────┐
│           credit-risk-network               │
│              (bridge driver)                │
│                                             │
│   ┌─────────┐  ┌──────────┐  ┌─────────┐   │
│   │  nginx  │──│ frontend │──│   api   │   │
│   │  :80    │  │  :8501   │  │  :8000  │   │
│   └─────────┘  └──────────┘  └─────────┘   │
│                                             │
└─────────────────────────────────────────────┘
        │              │             │
        ▼              ▼             ▼
    External       External      Internal
    (Users)        (Users)       (API calls)
```

## Volume Mounts

| Volume | Container | Mount Path | Purpose |
|--------|-----------|------------|---------|
| `./models` | api | `/app/models` | ML models (read-only) |
| `pip-cache` | api | `/root/.cache/pip` | Cached pip packages |

## Data Flow

```
User Request
     │
     ▼
┌─────────────────┐
│    Frontend     │  (Streamlit UI)
│    :8501        │
└────────┬────────┘
         │ HTTP POST
         ▼
┌─────────────────┐
│      API        │  (FastAPI)
│    :8000        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CatBoost Model │  (AUC: 0.786)
│  + SHAP         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Response      │
│ {probability,   │
│  risk_level,    │
│  explanation}   │
└─────────────────┘
```

## Quick Commands

```bash
# Start all services
docker-compose up -d

# Start only API (development)
docker-compose up -d api

# View logs
docker-compose logs -f api

# Rebuild with cache
docker-compose build

# Rebuild without cache
docker-compose build --no-cache

# Stop all
docker-compose down

# Check status
docker-compose ps
```

## Ports Summary

| Service | Internal Port | External Port | URL |
|---------|---------------|---------------|-----|
| API | 8000 | 8000 | http://localhost:8000 |
| Frontend | 8501 | 8501 | http://localhost:8501 |
| Nginx | 80 | 80 | http://localhost |

---
*Architecture by 7afnawi for Hefny*
