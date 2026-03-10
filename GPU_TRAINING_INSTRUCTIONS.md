# 🚀 GPU Training Instructions

## On Your Windows Machine:

### Step 1: Copy data from Kali VM
```powershell
# In PowerShell, create folder
mkdir C:\ml-training
cd C:\ml-training

# Copy from Kali (use shared folder or scp)
# Or download fresh from Kaggle
```

### Step 2: Copy training files
Copy these from `~/systeme-prediction-defaut-paiement/docker/`:
- `Dockerfile.gpu`
- `train_gpu.py`

### Step 3: Build Docker image
```powershell
docker build -f Dockerfile.gpu -t gpu-training .
```

### Step 4: Run training
```powershell
docker run --gpus all -v C:\ml-training\data:/app/data gpu-training
```

## Expected Results:
- **100 Optuna trials** with GPU acceleration
- **Full 300K dataset** (not 65K subset)
- **Target: 0.78-0.80 AUC**
- **Time: ~30-60 minutes** (vs hours on CPU)

## Quick Alternative (if Docker is complex):

Just run in WSL2 with CUDA:
```bash
pip install catboost optuna
python train_gpu.py
```
