# Model Zoo

This document describes all architectures supported by the framework.

---

## 1. CNN‑Based Models

### UNet
Classic encoder–decoder built from scratch.

Variants:
- UNet
- ResUNet (torchvision backbones)

Strengths:
- Robust for medical images
- Low FLOPs
- Easy to train

### MALUNet
Muti-Attention and Light-weight UNet

Features:
- Gated attention mechanisms
- External attention 
- Multi-stage features
---

## 2. Transformer‑Based Models

### TransUNet
CNN encoder (ResNet50) + ViT + U‑Net decoder.

Features:
- Global attention
- Hybrid design

---

## Swin‑UNet
Hierarchical transformer using windowed attention.

Benefits:
- Efficient local–global modeling
- Low FLOPs

---

# 3. State‑Space Models (SSM)

## VM‑UNet (in progress)
U‑Net with VMamba blocks.

Advantages:
- Linear complexity
- Long‑range modeling without attention
- Superior memory efficiency

---

# Adding a New Model

1. Implement model in `src/models/<family>`
2. Register it in `src/factory/models.py`
3. Create a config in `configs/model/`

Example:

```yaml
name: my_model
family: CNN
n_skip: 4
# Bottleneck
hidden_layers: 1024
# Decoder
decoder_channels: [256, 128, 64, 16]
```