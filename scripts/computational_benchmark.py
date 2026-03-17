from __future__ import annotations

from utils.bootstrap import set_repo_root_env
set_repo_root_env()  # must run before hydra.main

import hydra
from omegaconf import DictConfig

import torch

from factory.models import ModelFactory
from utils.model_inspection import ModelInspector
from utils.io import append_to_aggregated_csv, save_json


@hydra.main(version_base="1.3", config_path="../configs", config_name="computational_benchmark")
def main(cfg: DictConfig) -> None:
    print("=== Config (resolved) ===")
    device = torch.device(cfg.project.device if torch.cuda.is_available() else "cpu")
    # Build model from config
    model = ModelFactory.build(cfg.model).to(device)
    # Create inspector
    inspector = ModelInspector(model=model, cfg=cfg.inspect, device=device)
    print("=== Build experiment API (resolved)===")
    print(f"=== Computational Benchmark ===")
    print(f"=== Model  {cfg.model.name} ===")
    report = inspector.get_report(verbose=True)
    append_to_aggregated_csv(records=report, columns=cfg.columns, path=cfg.bench_dir, file_name=cfg.results_file)
    inspector.model_summary(verbose=cfg.inspect.verbose)
    save_json(report, cfg.inspect.report_path)
    # inspector.export_onnx()
    print(f"===Results saved into {cfg.bench_dir} ===")


if __name__ == "__main__":
    main()