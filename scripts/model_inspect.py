from __future__ import annotations
from unet_variants.utils.bootstrap import set_repo_root_env
set_repo_root_env()  # must run before hydra.main

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from unet_variants.models.factory import ModelFactory
from unet_variants.inspection.inspector import ModelInspector


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=== Config (resolved) ===")
    # print(OmegaConf.to_yaml(cfg, resolve=True))

    device = torch.device(cfg.project.device if torch.cuda.is_available() else "cpu")

    # Build model from config
    model = ModelFactory.build(cfg.model).to(device)

    # Create inspector
    inspector = ModelInspector(model=model, config=cfg.inspect, device=device)
    # Print summary / params / flops
    report = inspector.report(
        print_summary=cfg.inspect.torchinfo_summary.print_summary,
        export_summary=cfg.inspect.torchinfo_summary.export_summary,
        save_summary_path=cfg.inspect.torchinfo_summary.export_path,
        print_flops_report=cfg.inspect.flops_thop.print_report,
        export_flops_report=cfg.inspect.flops_thop.export_report,
        save_flops_txt=cfg.inspect.flops_thop.save_txt,
        save_flops_json=cfg.inspect.flops_thop.save_json,
        export_onnx_path=(cfg.inspect.export_onnx.path if cfg.inspect.export_onnx.enable else None)
    )

if __name__ == "__main__":
    main()
