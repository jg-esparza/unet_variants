# U‑Net Benchmarking Framework (CNN · Transformers · Mamba)

A modular, extensible, and reproducible framework for benchmarking CNN‑based, Transformer‑based, and SSM‑based UNet architectures for binary medical image segmentation.

This repository provides a unified, configuration‑driven training and evaluation pipeline using Hydra, MLflow, and PyTorch, with full experiment logging, automatic checkpointing, predicted samples, ONNX export, and failure case analysis.

---
## Features

### ⚙️ Hydra‑based configuration system
Reproducible experiments with clean override support.

### 📊 MLflow experiment tracking
Logs system and segmentation metrics, hyperparameters, checkpoints, ONNX exports, predicted samples.

### 🔍 Model inspection
Params, FLOPs, inference speed, memory footprint.

### 🧪 Unified training/evaluation engine
Dice, IoU, Accuracy, Sensitivity, Specificity.

### 🧱 Modular model, loss and optimization strategy factories
Easily plug in new architectures, loss function, optimizers and schedulers.

---
## 🧬 Supported Architectures
### CNN‑based
- **UNet** (unet). Classic encoder–decoder segmentation network.
- **ResNet+UNet** (resnet_unet). UNet using ResNet34 backbone for feature extraction.

### Transformer‑based
- **TransUNet**(transunet). Hybrid CNN-Vit.
- **Swin‑UNet**(swinunet). Hierarchical windowed‑attention transformer adapted to segmentation.

---

## 📚 Supported Datasets

Place datasets inside the `data/` folder following this structure:

Currently supported:

- **ISIC2017**. Divided into a 7:3 ratio, following prior procedure from [VM-UNet](https://github.com/JCruan519/VM-UNet).

---

## 📏 Evaluation Metrics

- Dice Similarity Coefficient
- Mean Intersection over Union
- Accuracy
- Sensitivity
- Specificity
---
## 📁 Repository Structure
```markdown
unet_variants/
│
├── config/                      # Hydra configs (models, inspect, dataset, logging, training, evaluating)
├── data/                        # Place datasets here
├── runs/                        # Hydra, MLflow runs (auto-generated)
├── scripts/                     # Training, evaluation, benchmarking scripts
├── README.md
├── pyproject.toml
└── src/
    └── unet_variants/
        ├── data/                # Dataset loaders, augmentations        
        ├── engine/              # Train, val, evaluator
        ├── loss/                # Losses
        ├── metrics/             # Binary Segmentation(Dice, IoU, Acc, Sensitivity, Specificity)
        ├── models/              # CNN, Transformer, SSM
        ├── optim/               # Optimization strategy
        └── utils/               # Helpers, logging, early stopper, visualization
```

---

## ⚙️ Installation

### 1. Create environment
```
conda create --name unet-variants python=3.11
conda activate unet-variants
```

### 2. Install PyTorch + CUDA
```
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install package locally
```
pip install -e .
```

## 🚀 Getting Started

### 1. Run model inspection
```
python scripts/model_inspect.py model=<model_name>
```
This generates:

- FLOPs
- parameter count
- architecture summary
- optional ONNX for visualization

ONNX saved in `runs/onnx/`


### 2. Run a training experiment

```
python scripts/train.py model=<model_name> dataset=<dataset_name>
```

Artifacts saved in `runs/mlruns/<experiment_id>/<run_id>/artifacts/`


### 3. View MLflow dashboard
```
mlflow server --backend-store-uri ./runs/mlruns
```
Open the URL to inspect:

- Training curves
- Model parameters
- Artifacts (checkpoints, sample predictions, failure cases)
- Metrics across experiments

--- 

🙏 Acknowledgements

- [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
- [TransUNet](https://github.com/Beckschen/TransUNet)
- [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
- [VM-UNet](https://github.com/JCruan519/VM-UNet)
- [ISIC 2017](https://challenge.isic-archive.com/data/) Challenge Dataset (public dermoscopic images)

Special thanks to the authors for providing their research and public resources.
