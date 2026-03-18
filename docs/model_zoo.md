# Model Zoo

This document describes all architectures supported by the framework.

---
## Supported Models

### 1. CNN‑Based Models

#### UNet (unet)
Classic encoder–decoder built from scratch.

Variants:
- UNet
- ResUNet (torchvision backbones)

#### Attention U-Net (att_unet)

Features:
- Gated attention mechanisms
- AGs easy to integrated

#### MALUNet (malunet)
Muti-Attention and Light-weight UNet

Features:
- Gated attention mechanisms
- Multi-stage features

#### CNN Strengths:
- Robust for medical images
- Low FLOPs
- Easy to train

---

### 2. Transformer‑Based Models

#### TransUNet (transunet)
CNN encoder (ResNet50) + ViT + U‑Net decoder.

Features:
- Global attention
- Hybrid design

---

#### Swin‑UNet (swinunet)
Hierarchical transformer using windowed attention.

Features:
- Swin Transformer with shifted windows
- Efficient local–global modeling

#### Transformers Strengths:
- Global modeling

---

### 3. State‑Space Models (SSM)

#### VM‑UNet (in progress)
U‑Net with VMamba blocks.

- Pure SSM-based
- VSS blocks derived from [VMamaba](https://github.com/MzeroMiko/VMamba)

#### SSM Strengths:
- Linear complexity
- Long‑range modeling without attention

---

## Adding a New Model

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