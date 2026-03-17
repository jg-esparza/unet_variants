# Benchmarks

This section includes computational benchmarks for all supported architectures.

---

## 1. Benchmark Script

```bash
python ./scripts/computational_benchmark.py -m +computational_benchmark=computational_benchmark model=unet,transunet,swinunet project.image_size=224
```

2. Results (Input 224×224)
<!-- BEGIN_BENCHMARK_TABLE -->
|      Model     | Params (M) | FLOPs (G) | Size (MB) |
|:--------------:|:----------:|:---------:|:---------:|
| UNet(ResNet34) |    26.76   |    1.78   |   132.71  |
|    TransUNet   |   105.28   |   24.67   |   834.64  |
|    Swin‑UNet   |    27.17   |    5.91   |   405.22  |
<!-- END_BENCHMARK_TABLE -->

