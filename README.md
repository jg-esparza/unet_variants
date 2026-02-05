# U-Net Variants Benchmark (CNN · Transformers)


A clean, reproducible benchmark framework for **binary segmentation**:
- **CNN-Based**
    - U-Net
- **Transformer-based**
    
Designed to be:
- Config-driven
- Reproducible
- Easy to extend
- Practical for research + engineering portfolios

---


## Features

✅ Unified experiment runner:
- Build model from config
- Parameter count
- FLOPs estimation (input-size dependent)
- `torchinfo` summary
- Train / Evaluate
- Save / Load checkpoints

✅ Experiment tracking:
- MLflow logging (metrics, params, artifacts)

✅ Modular design:
- Model factory for quick additions
- Separate training, evaluation, inspection, checkpointing modules

---

## Evaluation Metrics

- Dice Similarity Coefficient
- Mean Intersection over Union
- Accuracy
- Sensitivity
- Specificity

---

## Requirements

Please make sure that your equipment meets the necessary requirements.

- Linux 
- NVIDIA GPU

## Installation

Create and activate a conda environment by:
```bash
conda create --name <my-env> python=3.11
```
```bash
conda activate <my-env> 
```
```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

### Libraries for evaluation and experiment tracking

```bash
pip install mlflow psutil pynvml torchinfo ultralytics-thop einops
```
---

