# Benchmarks

This section includes computational benchmarks for all supported architectures.

---

## 1. Benchmark Script

```bash
python ./scripts/computational_benchmark.py -m +computational_benchmark=computational_benchmark model=unet,transunet,swinunet project.image_size=224
```

2. Results (Input 224×224)
## 📏 Computational Benchmark (256×256 Input)


<!-- BEGIN_BENCHMARK_TABLE -->
|          Model           | Params (M) | FLOPs (G) | Size (MB) |
|:------------------------:|:----------:|:---------:|:---------:|
|      UNet(ResNet34)      |    26.76   |   7.13    |  209.69   |
|     Attention U-Net      |    34.87   |   66.63   |  905.58   |
|         MALUNet          |    0.175   |   0.083   |   25.69   |
| TransUNet(Patch Size 16) |   105.28   |   32.23   |  961.85   |
| Swin‑UNet(Window Size 8) |    27.17   |   7.72    |  496.03   |
<!-- END_BENCHMARK_TABLE -->

Computed using

- torchinfo (parameters)
- ultralytics-thop (FLOPs)

