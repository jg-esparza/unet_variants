from __future__ import annotations
from unet_variants.utils.bootstrap import set_repo_root_env
set_repo_root_env()  # must run before hydra.main

import hydra
from omegaconf import DictConfig

from unet_variants.experiments.experiment import ExperimentManager

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=== Config (resolved) ===")
    # print(OmegaConf.to_yaml(cfg, resolve=True))

    exp = ExperimentManager(cfg)

    exp.model_summary()
    exp.model_flops()
    # exp.model_onnx()
    exp.run()
    # exp.resume(run_id="")
    # checkpoint = torch.load("D:/joseg/Documents/Projects_practice/unet_variants/runs/mlruns/288275763051620023/659fb6241c334c468926e37f5422b9a6/artifacts/best.pth", map_location=torch.device('cpu'))
    # exp.model.load_state_dict(checkpoint)
    # exp.model.to(exp.device)
    # exp.evaluate()

if __name__ == "__main__":
    main()