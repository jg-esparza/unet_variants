from __future__ import annotations

from unet_variants.utils.bootstrap import set_repo_root_env
set_repo_root_env()  # must run before hydra.main

import hydra
from omegaconf import DictConfig

from unet_variants.experiments.experiment import ExperimentManager
from unet_variants.utils.io import append_to_aggregated_csv

columns = [
            "model",
            "family",
            "total_params",
            "trainable_params",
            "flops",
            "params_profiler",
            "input_size",
        ]

@hydra.main(version_base="1.3", config_path="../configs", config_name="benchmark")
def main(cfg: DictConfig) -> None:
    print("=== Config (resolved) ===")
    exp = ExperimentManager(cfg)
    print("=== Build experiment API (resolved)===")
    print(f"=== Model  {cfg.model.name} ===")
    print(f"=== Benchmark ===")
    if cfg.inference_bench:
        print(f"=== Inference Benchmark ===")
        report = exp.inspector.get_report(verbose=True)
        append_to_aggregated_csv(records=report, columns=columns, path=cfg.bench_dir, file_name="inference_benchmark.csv")
        print(f"===Results saved into {cfg.bench_dir} ===")
    if cfg.segmentation_bench:
        print("=== Segmentation Benchmark ===")
        print(f"=== Experiment  {cfg.logging.experiment_name} ===")
        if cfg.project.use_pretrained_ckpt:
            print("===Use pretrained checkpoint active---")
            exp.load_pretrained_ckpt()
        exp.run()


if __name__ == "__main__":
    main()