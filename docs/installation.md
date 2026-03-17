# Installation Guide

This document explains how to install dependencies and set up the environment.

---

## 1. Requirements

- Python 3.11
- Conda
- PyTorch (GPU recommended)
- CUDA 12.8+ (optional)

---

## 2. Create the environment

```bash
conda env create -f environment.yml
conda activate unet-variants
```

## 3. Install editable package
```bash
pip install -e .
```

## 4. Installing Mamba (for VM‑UNet)
The Mamba SSM implementation will be included later.
For now:
```bash
pip install mamba-ssm
```
Optional GPU‑accelerated version:
```bash
pip install causal-conv1d
```
## 5. Testing Installation
```bash
python -c "import torch; print(torch.rand(1))"
```
If this runs, your environment is ready.

