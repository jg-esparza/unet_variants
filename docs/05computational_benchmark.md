# Benchmarks

This section includes computational benchmarks for all supported architectures.

---

## 1. Computational benchmark Script

```bash
./scripts/computational_bench.sh
```
Or:
```bash
python ./scripts/computational_benchmark.py -m +computational_benchmark=computational_benchmark model=unet,malunet,transunet,swinunet,vmunet project.image_size=256
```

## 📏 Computational Benchmark (1×3×256×256 Input)


<!-- BEGIN_BENCHMARK_TABLE -->
|          Model           | Params (M) | FLOPs (G) | Size (MB) |
|:------------------------:|:----------:|:---------:|:---------:|
|      UNet(ResNet34)      |    26.76   |   7.13    |  209.69   |
|     Attention U-Net      |    34.87   |   66.63   |  905.58   |
|         MALUNet          |    0.175   |   0.083   |   25.69   |
| TransUNet(Patch Size 16) |   105.28   |   32.23   |  961.85   |
| Swin‑UNet(Window Size 8) |    27.17   |   7.72    |  496.03   |
|         VM‑UNet          |   27.43    |   4.11    |  344.66   |
<!-- END_BENCHMARK_TABLE -->

Computed using

- torchinfo (parameters)
- ultralytics-thop (FLOPs)

