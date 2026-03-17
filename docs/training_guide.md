# Training Guide

This document explains how to launch experiments, override settings, and monitor progress.

---

## 1. Launch Training

Basic training run:

```bash
python scripts/train.py model=unet dataset=isic2018
```

## 2. Override Hyperparameters

Change the learning rate:
```bash
python scripts/train.py train.optimizer.lr=1e-4
```
Use a different batch size:
```bash
python scripts/train.py project.batch_size=8
```
Use multiple overrides:
```bash
python scripts/train.py model=swinunet project.batch_size=4 project.epochs=200
```

### 3. Training Artifacts
Saved under: `runs/mlruns/<experiment_id>/<run_id>/artifacts/`
Includes:

- checkpoints 
- predicted samples 
- ONNX graphs

### 4. Resume Training

```bash
python scripts/resume.py logging.run_id=<run_id>
```

---
### 5. Monitor Training in MLflow
Start server:
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```
Open:
```
http://127.0.0.1:5000
```
Monitor:
- Loss curves
- Segmentation and system metrics
- Hyperparameters 
- Model graphs 
- Predictions
