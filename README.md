
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

## 🧱 Project Architecture Overview
```markdown

unet_variants/
├─ data/                              # Data storage
│  └─ datasets
├─ configs/
│  ├─ config.yaml                     # Top-level Hydra config; composes model/data/train/inspect/logging/task/paths
│  ├─ data/
│  │  └─ isic.yaml                    # Example dataset config (paths, input size, normalization)
│  ├─ eval/
│  │  └─ default.yaml                 # Common evaluation parameters
│  ├─ inspect/
│  │  └─ default.yaml                 # Common model inspection parameters
│  ├─ logging/
│  │   └─ mlflow.yaml                 # Tracking URI, experiment name, run naming
│  ├─ model/
│  │   └─ unet.yaml                   # U-Net config
│  ├─ task/
│  │   └─ default.yaml                # Segmentation task (Binary, output channels)  
│  └─ train/
│  │   └─ default.yaml                # Common training params (optimizer, scheduler, loss, batch_size)
├─ src/
│  ├─ unet_variants/
│  │  ├─ data/                        # Dataset modules (loaders, transforms, preparation)
│  │  │  ├── dataset.py
│  │  │  ├── loaders.py
│  │  │  ├── transforms.py
│  │  │  └── prepare.py
│  │  ├─ engine/                      # Core training & evaluation logic
│  │  │  ├── trainer.py
│  │  │  ├── evaluator.py
│  │  │  ├── inference.py
│  │  │  └── checkpoint.py   
│  ├── inspection/                     # Introspection and profiling utilities
│  │  │   ├── flops.py
│  │  │   ├── summary.py
│  │  │   ├── viz.py
│  │  │   ├── inspector.py
│  │  │   └── onnx.py
│  │  ├── losses/                      # Loss functions (BCE+Dice, etc.)
│  │  │   └── bce_dice.py
│  │  ├── metrics/                     # Metrics for segmentation evaluation
│  │  │   └── segmentation.py
│  │  ├── runners/                     # Experiment runner (Hydra + MLflow)
│  │  │   └── experiment.py
│  │  ├── utils/                       # General-purpose utilities
│  │  │   ├── bootstrap.py
│  │  │   ├── device.py
│  │  │   ├── io.py
│  │  │   ├── logging.py
│  │  │   └── seeds.py
│  │  ├── models/                      # All U-Net variants live here
│  │  │   ├── components/              # Reusable blocks (conv blocks, attention, upsample)
│  │  │   ├── unet/                    # Baseline U-Net implementation
│  │  │   └── factory.py               # 🔑 Model registry/factory (maps string keys → model classes)
├─ scripts/
│  ├─ run_train.sh
│  └─ run_eval.sh
├─ runs/
│  ├─ hydraruns                       
│  └─ mlruns
├─ .gitignore
├─ README.md
├─ LICENSE
└─ pyproject.toml                 # Packaging + minimal dependencies
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


## 🚀 Getting Started

After installing the environment and the package, you can immediately run an experiment.

### 1. Run a training experiment

```
python scripts/train.py model=unet dataset=isic2017
```

Hydra will automatically create a timestamped folder inside:
```markdown
runs/
│  ├── hydra/
│  ├── mlflow/
```
### 2. View MLflow dashboard
```
mlflow ui --backend-store-uri runs/mlflow
```
Open the URL to inspect:

- training curves
- model parameters
- artifacts (checkpoints, sample predictions)
= metrics across experiments

### 3. Run model inspection (FLOPs, summary)
```
python scripts/inspect.py model=unet input_size=1,3,256,256
```
This generates:

- FLOPs
- parameter count
- architecture summary
- optional visualizations

### 4. Perform inference
```
python scripts/infer.py model=unet ckpt=path/to/checkpoint.png input=path/to/image.png
```
The output will be saved inside:
```markdown
runs/
│  ├── mlflow/
│  │  ├── <run_id>/
│  │  │  ├── predictions/
```
