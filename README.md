# U‑Net Benchmarking Framework (CNN · Transformers · Mamba)

A modular, extensible, and reproducible framework for benchmarking CNN‑based, Transformer‑based, and SSM‑based UNet architectures for binary medical image segmentation.

This repository provides a unified, configuration‑driven training and evaluation pipeline using Hydra, MLflow, and PyTorch, with full experiment logging, automatic checkpointing, predicted samples, ONNX export, and failure case analysis.

---
## Features

### ⚙️ Hydra‑based configuration system
Reproducible experiments with clean override support.

### 📊 MLflow experiment tracking
Logs system and segmentation metrics, hyperparameters, checkpoints, ONNX exports, predicted samples.

### 🔍 Model inspection
Params, FLOPs, inference speed, memory footprint.

### 🧪 Unified training/evaluation engine
Dice, IoU, Accuracy, Sensitivity, Specificity.

### 🧱 Modular model, loss and optimization strategy factories
Easily plug in new architectures, loss function, optimizers and schedulers.

---
## 🧬 Supported Architectures
### CNN‑based
- **UNet**. Classic encoder–decoder segmentation network built from scratch. Optional use of **ResUNet** backbones from `torchvision`.

### Transformer‑based
- **TransUNet**. Hybrid CNN-Vit follows original config.
- **Swin‑UNet**. Hierarchical windowed‑attention transformer, follows original config.

### SSM-based

- **VM-UNet**. In progress.

---

## 📚 Supported Datasets

Place datasets inside the `data/` folder following this structure:

Divided into a 7:3 ratio, following prior procedure from [VM-UNet](https://github.com/JCruan519/VM-UNet)

- ISIC2017
- ISIC2018

---

## 📏 Evaluation Metrics

- Dice Similarity Coefficient
- Mean Intersection over Union
- Accuracy
- Sensitivity
- Specificity
---


## 🧪 Computational Benchmark

Results measured with Input size: (1×3×224×224).

<!-- BEGIN_BENCHMARK_TABLE -->
|   Model   | Params (M) | FLOPs (G) | Estimated Total Size (MB) |
|:---------:|:---------:|:---------:|:-------------------------:|
|   UNet    |   18.02   |   20.22   |          341.23           |
| TransUNet |  105.28   |   24.67   |          834.64           |
| Swin‑UNet |   27.17   |   5.91    |          405.22           |

<!-- END_BENCHMARK_TABLE -->

**Measured**: 

- Params - `torchinfo`
- FLOPs - `ultralytics-thop`.


## 📁 Repository Structure
```markdown
unet_variants/
│
├── README.md
├── environment.yml
├── pyproject.toml
├── data/                        # Place datasets here
├── scripts/                     # Training, evaluation, benchmarking scripts
├── pretrained_ckpt/             # Pretrained checkpoints
├── config/                      # Hydra configs 
    ├── dataset/                 
    ├── model/                   
    ├── inspect/
    ├── logging/
    ├── train/
    └── eval/
└── src/
    ├── data/                    # Dataset loaders, augmentations        
    ├── engine/                  # Experiment manager, train, validate, evaluate
    ├── factory/                 # 
    ├── metrics/                 # Binary Segmentation(Dice, IoU, Acc, Sensitivity, Specificity)
    ├── models/                  # CNN, Transformer, SSM
    ├── optim/                   # Optimization strategy
    └── utils/                   # Helpers, logging, early stopper, visualization
```

---

## ⚙️ Installation

### 1. Create env from the `environment.yml` file
```
conda env create -f environment.yml
```

### 2. Activate env
```
conda activate unet-variants
```

## 🚀 Getting Started

### 1. Run model inspection
```
python scripts/model_inspect.py model=<model_name>
```
This generates:

- FLOPs
- parameter count
- architecture summary
- optional ONNX for visualization

ONNX saved in `runs/reports/img_<dataset.image_size>_inspect_bs<inspect.batch_size>/<model.name>`

### 2. Run a training experiment

```
python scripts/train.py model=<model_name>
```

Artifacts saved in `runs/mlruns/<experiment_id>/<run_id>/artifacts/`.

### 3. Run multiples experiments
Run in **Linux** for all available models. 

```
./scripts/bench.sh
```
Only Computational benchmark.

Run it manually as:
```
python ./scripts/computational_benchmark.py -m +computational_benchmark=computational_benchmark model=<model_name1>,<model_name2> project.image_size=256,512
```

Csv file with inference results saved in `runs/reports/bench_img_<dataset.image_size>_inspect_bs<inspect.batch_size>/<date_time>/`.

### 3. View MLflow dashboard
```
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```
Open the URL to inspect:

- Training curves
- Model parameters
- Artifacts (checkpoints, sample predictions, failure cases)
- Metrics across experiments

### 4. Resume an experiment
Option to resume an experiment if interrupted by using the run_id from mlflow.
```
python scripts/resume.py logging.run_id=<run_id>
```

### 5. Evaluate an experiment
```
python scripts/evaluate.py logging.run_id=<run_id>
```

--- 

🙏 Acknowledgements

- [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
- [TransUNet](https://github.com/Beckschen/TransUNet)
- [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
- [VM-UNet](https://github.com/JCruan519/VM-UNet)
- [ISIC 2017](https://challenge.isic-archive.com/data/#2017) Challenge Dataset (public dermoscopic images)
- [ISIC 2018](https://challenge.isic-archive.com/data/#2018) Challenge Dataset (public dermoscopic images)

Special thanks to the authors for providing their research and public resources.
