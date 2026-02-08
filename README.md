
 U-Net Variants Benchmark (CNN · Transformers · Mamba)

A clean, reproducible benchmark framework for **binary medical image segmentation**, supporting multiple public datasets and a growing collection of U‑Net variants (CNN-based, Transformer-based and Mamba-based).

This project is **inspired by the engineering patterns developed during my M.Sc. thesis**, but it is a **new, broader, independent framework** focused on:

- extensibility  
- reproducibility  
- fair model comparison  
- modular research workflows  
- clean engineering design

---

## 🚀 Objectives

- Provide a unified and reproducible pipeline for **training / evaluation / inference**  
- Enable rapid experimentation with different **U‑Net architectures**  
- Offer a modular framework that researchers and engineers can easily extend  
- Support multi-dataset benchmarking with consistent metrics and preprocessing  

---

## 🧩 Supported Model Families

### **CNN-Based**
- U-Net
- (soon) ResNet‑U‑Net

### **Transformer-Based**
- (upcoming) Swin-UNet  

### **Mamba-Based**
- (upcoming) VM‑UNet  

---

## ✨ Key Features

### **Unified Experiment Runner**
- Build models from Hydra config files  
- Automatic parameter count  
- FLOPs estimation (input-size dependent)  
- Detailed `torchinfo` model summaries  
- Train / Evaluate with consistent metrics  
- Save / Load checkpoints  

### **Experiment Tracking**
- MLflow logging  
- Metrics, parameters, curves  
- Artifacts (model weights, visual outputs, summaries)  

### **Modular Architecture**
- Dataset wrappers  
- Dataloaders  
- Training / evaluation engines  
- Model factory  
- Inspection utilities (FLOPs, summaries, visualizations)

---

## 📚 Supported Datasets

Place datasets inside the `data/` folder following this structure:

Currently supported:

- **ISIC2017**
- (upcoming)**Kvasir-SEG**
- (upcoming)**BUSI**

---

## 📏 Evaluation Metrics

- **Dice Similarity Coefficient (DSC)**
- **Mean Intersection over Union (mIoU)**
- **Accuracy**
- **Sensitivity**
- **Specificity**

---

## 🧱 Project Architecture Overview Diagram
```markdown
## 🏗️ Project Architecture Overview

unet_variants/
│
├── data/                         # Dataset modules (loaders, transforms, preparation)
│   ├── dataset.py
│   ├── loaders.py
│   ├── transforms.py
│   └── prepare.py
│
├── engine/                       # Core training & evaluation logic
│   ├── trainer.py
│   ├── evaluator.py
│   ├── inference.py
│   └── checkpoint.py
│
├── inspection/                   # Introspection and profiling utilities
│   ├── flops.py
│   ├── summary.py
│   ├── viz.py
│   └── inspector.py
│   └── onnx.py
│
├── losses/                       # Loss functions (BCE+Dice, etc.)
│   └── bce_dice.py
│
├── metrics/                      # Metrics for segmentation evaluation
│   └── segmentation.py
│
├── models/                       # Model zoo and building blocks
│   ├── components/               # Shared blocks (conv blocks, attention, upsample)
│   ├── unet/                     # Baseline U-Net implementation
│   └── factory.py                # Model factory for dynamic instantiation
│
├── runners/                      # Experiment runner (Hydra + MLflow)
│   └── experiment.py
│
├── utils/                        # General-purpose utilities
│   ├── bootstrap.py
│   ├── device.py
│   ├── io.py
│   ├── logging.py
│   ├── seeds.py
│   └── utils.py
│
└── scripts/                      # Entry points (train.py, eval.py, inspect.py)
```

## ⚙️ Installation

### 1. Create environment
```
conda create --name unet-benchmark python=3.11
conda activate unet-benchmark
```

### 2. Install PyTorch + CUDA
```
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install package locally
```
pip install -e .
```
