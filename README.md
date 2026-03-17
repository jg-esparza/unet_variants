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

- **UNet** — classic encoder‑decoder. Optional use of **ResUNet** backbones from `torchvision`.

### Transformer‑based

- **TransUNet** — hybrid CNN + ViT 
- **Swin‑UNet** — hierarchical windowed self‑attention

### State‑Space Models

- VM‑UNet (in progress)

---

## 📚 Supported Datasets

Datasets must follow the structure defined in configs/dataset/.
Current support:

- ISIC2017
- ISIC2018

Both follow a 7:3 split, consistent with prior work (e.g., [VM-UNet](https://github.com/JCruan519/VM-UNet)).

See docs/datasets.md for preparation instructions.

---

## 📏 Computational Benchmark (224×224 Input)


<!-- BEGIN_BENCHMARK_TABLE -->
|      Model     | Params (M) | FLOPs (G) | Size (MB) |
|:--------------:|:----------:|:---------:|:---------:|
| UNet(ResNet34) |    26.76   |    1.78   |   132.71  |
|    TransUNet   |   105.28   |   24.67   |   834.64  |
|    Swin‑UNet   |    27.17   |    5.91   |   405.22  |
<!-- END_BENCHMARK_TABLE -->

Computed using

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

🙏 Acknowledgements

- [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
- [TransUNet](https://github.com/Beckschen/TransUNet)
- [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
- [VM-UNet](https://github.com/JCruan519/VM-UNet)
- [ISIC Challenges(2017,2018)](https://challenge.isic-archive.com/data/)

Special thanks to the authors for providing their research and public resources.
