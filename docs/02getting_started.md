
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
## 3. Folder Structure Summary
See the main README for full structure, or browse:
```markdown

src/
├── data/                    # Datasets and augmentations       
├── engine/                  # Train/validate/evaluate loops
├── factory/                 # Model, loss, optimizer, scheduler builders
├── models/                  # CNN, Transformer architectures
└── utils/                   # Logging, metrics, visualization, seed
```
