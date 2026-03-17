from __future__ import annotations
from utils.bootstrap import set_repo_root_env
set_repo_root_env()  # must run before hydra.main

import hydra
from omegaconf import DictConfig
import torch

from factory.models import ModelFactory
from utils.model_inspection import ModelInspector


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=== Config (resolved) ===")
    device = torch.device(cfg.project.device if torch.cuda.is_available() else "cpu")
    # Build model from config
    model = ModelFactory.build(cfg.model).to(device)
    # Create inspector
    inspector = ModelInspector(model=model, cfg=cfg.inspect, device=device)
    # Print summary / params / flops
    inspector.model_summary()
    report = inspector.get_report(verbose=True)
    print(report)

if __name__ == "__main__":
    main()
