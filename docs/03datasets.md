# Dataset Preparation

This framework currently supports:

- **ISIC 2017**
- **ISIC 2018**

Datasets must be manually downloaded and placed inside:
```markdown
data/
├── isic2017/
└── isic2018/
```
---
## 1. Dataset Directory Structure

For each dataset:
```markdown
data/
├── isic2017/
        ├── images/
        └── masks/
└── isic2018/
        ├── images/
        └── masks/
```

Splits follow a **7:3 ratio**, consistent with the [VM-UNet](https://github.com/JCruan519/VM-UNet) workflow

---

### 2. Custom Datasets

Create a corresponding Hydra config in `configs/dataset/`

Example
```yaml

# configs/dataset/my_custom.yaml
name: my_custom
path: ${base.data_dir}/my_custom
image_size: 256
```