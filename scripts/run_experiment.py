from __future__ import annotations
from unet_variants.utils.bootstrap import set_repo_root_env
set_repo_root_env()  # must run before hydra.main

import hydra
from omegaconf import DictConfig

from unet_variants.experiments.experiment import ExperimentManager

import torch

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=== Config (resolved) ===")
    # print(OmegaConf.to_yaml(cfg, resolve=True))

    exp = ExperimentManager(cfg)

    # exp.model_summary()
    # exp.model_flops_thop()
    # exp.model_onnx()
    exp.run()
    # exp.resume(run_id="")

if __name__ == "__main__":
    main()