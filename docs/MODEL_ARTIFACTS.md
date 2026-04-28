# Model Artifacts

This repository contains the full training/deployment code and metadata files, but the trained binary model is intentionally not committed by default.

## Required file for real predictions

The API searches `models/` in this order:

1. `catboost_optimized.joblib`
2. `best_model_v2.joblib`
3. `best_model.joblib`
4. `CatBoost.joblib`

At least one of these files must exist for `/predict`, `/predict/batch`, and `/explain` to be ready.

## How to create the model

1. Download the Home Credit Default Risk dataset from Kaggle.
2. Place the raw data under `data/` according to the pipeline expectations.
3. Run the training pipeline, for example:

```bash
python src/main.py
# or, for the advanced pipeline:
python train_ultimate.py
```

4. Confirm that a `.joblib` model file was created under `models/`.
5. Start the API and check readiness:

```bash
uvicorn api.app_v2:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/ready
```

## Health endpoints

- `GET /` is a liveness/status endpoint. It can return `degraded` if the API is running but the model is missing.
- `GET /ready` is a readiness endpoint. It returns HTTP `503` until both the model and feature list are loaded.

## Important limitation

If the model artifact is missing, the dashboard can load but real predictions are unavailable. Do not present outputs from a placeholder/demo model as real credit-risk predictions.
