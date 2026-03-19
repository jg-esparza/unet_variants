# Installation Guide

This document explains how to install dependencies and set up the environment.

---

## 1. Requirements

- Linux
- NVIDIA GPU
- Conda
- PyTorch 2.4.1+
- CUDA 12.4+

---

## 2. Create the environment

```bash
conda env create -f environment.yml
conda activate unet-variants
```

## 3. Testing Installation
```bash
python -c "import torch; print(torch.rand(1))"
```
If this runs, your environment is ready.

