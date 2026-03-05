from __future__ import annotations
from unet_variants.utils.bootstrap import set_repo_root_env
set_repo_root_env()  # must run before hydra.main

import hydra
from omegaconf import DictConfig

from unet_variants.experiments.experiment import ExperimentManager

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=== Config (resolved) ===")
    exp = ExperimentManager(cfg)
    print("=== Build experiment API (resolved)===")
    print(f"=== Model  {cfg.model.name} ===")
    print(f"=== Experiment  {cfg.logging.experiment_name} ===")
    exp.model_flops()
    # Load pretrained weights
    if cfg.project.use_pretrained_ckpt:
        print("===Use pretrained checkpoint active---")
        exp.load_pretrained_ckpt()
    # exp.run()
    # exp.resume(run_id="feb8a8e63f164dff90ac1cc858d5048e")
    # exp.evaluate_run(run_id="feb8a8e63f164dff90ac1cc858d5048e")

if __name__ == "__main__":
    main()