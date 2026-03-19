# U‑Net Benchmarking Framework

**CNN · Transformers · State‑Space Models (Mamba)**

A modular, extensible, and reproducible framework for benchmarking U‑Net variants across CNNs, Transformers, and State‑Space Models (SSMs) for medical image segmentation.
This framework provides:

- ⚙️ Hydra‑based configuration
- 🔍 Model inspection: Params · FLOPs · Estimated Memory · ONNX export
- 🧪 Unified training/evaluation engine 
- 📊 Full MLflow experiment tracking
- 🧱 Plug‑and‑play model/loss/optimizer factory
---
## ✨ Features

### ⚙️ Hydra Configuration System
Modular experiments with clean overrides and YAML‑based composition.

### 🔍 Model Inspection Tools
Compute:
- FLOPs 
- Parameter count
- Memory footprint 
- ONNX export

### 📊 MLflow Tracking
Automatically logs metrics, hyperparameters, checkpoints, model graphs, predictions, and ONNX exports.

### 🧪 Unified Training & Evaluation
Supports:
- Dice 
- mIoU 
- Accuracy 
- Sensitivity 
- Specificity

### 🧱 Factory Modules
Easily add:
- New architectures 
- Custom losses 
- Optimizers 
- Schedulers
---

## 🧬 Supported Architectures
### CNN‑based

- **UNet**: Convolutional Networks for Biomedical Image SegmentationOptional use of **ResUNet** backbones from `torchvision`
  - https://arxiv.org/abs/1505.04597
- **Attention U-Net**: Learning Where to Look for the Pancreas
  - https://arxiv.org/abs/1804.03999
- **MALUNet**: A Multi-Attention and Light-weight UNet for Skin Lesion Segmentation
  - https://arxiv.org/pdf/2211.01784

### Transformer‑based

- **TransUNet**: Transformers Make Strong Encoders for Medical Image Segmentation
  - https://arxiv.org/pdf/2102.04306
- **Swin‑UNet**: Unet-like Pure Transformer for Medical Image Segmentation
  - https://arxiv.org/pdf/2105.05537

### State‑Space Models

- **VM‑UNet**: Vision Mamba UNet for Medical Image Segmentation
  - https://arxiv.org/pdf/2402.02491

---

## 📚 Supported Datasets

Datasets must follow the structure defined in configs/dataset/.
Current support:

- ISIC2017
- ISIC2018

Both follow a 7:3 split, consistent with prior work (e.g., [VM-UNet](https://github.com/JCruan519/VM-UNet)).

See docs/datasets.md for preparation instructions.

---

## 📏 Computational Benchmark (1×3×256×256 Input)


<!-- BEGIN_BENCHMARK_TABLE -->
|          Model           | Params (M) | FLOPs (G) | Size (MB) |
|:------------------------:|:----------:|:---------:|:---------:|
|      UNet(ResNet34)      |   26.76    |   7.13    |  209.69   |
| Swin‑UNet(Window Size 8) |   27.17    |   7.72    |  496.03   |
|         VM‑UNet          |   27.43    |   4.11    |  344.66   |
<!-- END_BENCHMARK_TABLE -->

Example computed using

- torchinfo (parameters)
- ultralytics-thop (FLOPs)

More details in docs/benchmarks.md.

---

## 📁 Repository Structure
```markdown
unet_variants/
│
├── README.md
├── LICENSE
├── environment.yml
├── pyproject.toml
├── data/                        # Dataset folders (ignored by git)
├── scripts/                     # Training, evaluation, benchmark scripts
├── pretrained_ckpt/             # Pretrained checkpoints
├── configs/                     # Hydra configuration system
    ├── dataset/                 
    ├── model/                   
    ├── inspect/
    ├── logging/
    ├── train/
    └── eval/
└── src/
    ├── data/                    # Datasets and augmentations       
    ├── engine/                  # Train/validate/evaluate loops
    ├── factory/                 # Model, loss, optimizer, scheduler builders
    ├── models/                  # CNN, Transformer architectures
    └── utils/                   # Logging, metrics, visualization, seed
```

---

## 🚀 Quickstart

### Install environment
```
conda env create -f environment.yml
conda activate unet-variants
```

### Model inspection
```
python scripts/model_inspect.py model=unet
```

### Training

```
python scripts/train.py model=unet dataset=isic2017
```

### Run computational benchmark with input 224,512, 1024

```
./scripts/computational_bench.sh
```

### MLflow UI
```
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

--- 

## 🙏 Acknowledgements

- [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
- [Attention U-Net](https://github.com/ozan-oktay/Attention-Gated-Networks)
- [MALUNet](https://github.com/JCruan519/MALUNet)
- [TransUNet](https://github.com/Beckschen/TransUNet)
- [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
- [VM-UNet](https://github.com/JCruan519/VM-UNet)
- [ISIC Challenges(2017,2018)](https://challenge.isic-archive.com/data/)

Special thanks to the authors for providing their research and public resources.
