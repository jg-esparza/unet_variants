
# Getting Started

This guide introduces the core concepts of the U‑Net Benchmarking Framework and helps you run your first experiment.

---

## 1. Project Overview
This framework provides:

- CNN · Transformer · SSM‑based U‑Net variants 
- Hydra‑based configuration system
- Model inspection (FLOPs, Params, Estimated Memory)
- MLflow logging 
- Unified training/evaluation engine 
- ONNX export 
- Easily extendable model/loss/optimizer factory

## 2. Quick Examples
Inspect a model:
```bash
python scripts/model_inspect.py model=unet
```
Train a model:
```bash
python scripts/train.py model=unet dataset=isic2018
```
Evaluate a run:
```bash
python scripts/evaluate.py logging.run_id=<run_id>
```
Resume training:
```bash
python scripts/resume.py logging.run_id=<run_id>
```

## 3. Config Structure
See the main README for full structure, or browse:
```markdown

configs/
├── dataset/                       
    ├── isic2017.yaml        # Dataset config, name, task, local directory, normalization stats
    ├── isic2018.yaml        
    └── augmentation.yaml    # Enable augmentation and set random transformation probability
├── model/
    ├── unet_config/         # Base Unet variants with ResNet18 and ResNet34 from torchvision
    ├── unet.yaml            # Base Unet
    ├── att_unet.yaml        # Attention U-Net, adapted from original
    ├── malunet.yaml         # MALUNet, following original config
    ├── transunet.yaml       # TransUNet, following original config
    ├── swinunet.yaml        # Swin-Unet, following original config
    └── vmunet.yaml          # VM-UNet, following original config
├── inspect/
    └── default.yaml         # Model inspection config
├── logging/
    └── mlflow.yaml          # MLflow config
├── train/
    └── default.yaml         # Loss, optimizer, scheduler
└── eval/
    └── default.yaml         # Task, threshold
```

## 4. Source Code Structure Summary
See the main README for full structure, or browse:
```markdown

src/
├── dataset/                 # Datasets and augmentations       
├── engine/                  # Train/validate/evaluate loops
├── factory/                 # Model, loss, optimizer, scheduler builders
├── models/                  # CNN, Transformer architectures
└── utils/                   # Logging, metrics, visualization, seed
```
