
from __future__ import annotations

from omegaconf import DictConfig

import torch

from unet_variants.models.factory import ModelFactory
from unet_variants.inspection.inspector import ModelInspector
from unet_variants.loss.factory import LossFactory
from unet_variants.optim.build_optim import OptimizerFactory
from unet_variants.optim.build_scheduler import SchedulerFactory
from unet_variants.data.loaders import build_dataloaders

class ExperimentManager:
    """
    Orchestrates a full experiment lifecycle:
      - config load/resolve
      - run dir + logging
      - seeding + device selection
      - data/model/optim/scheduler setup
      - train/val/test
      - checkpoints (best/last) + resume
      - model inspection (summary/flops/onnx)
    """

    def __init__(self, config: DictConfig = None):
        # Hydra config
        # self.cfg = OmegaConf.to_yaml(config, resolve=True)
        self.cfg = config
        """
        # Run directory
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(self.cfg["experiment"]["output_root"]) / \
                       self._resolve_exp_name(self.cfg["experiment"]) / self.timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        """
        # Device
        # self.device = self._select_device(self.cfg.get("device", {}))
        self.device = torch.device(self.cfg.project.device if torch.cuda.is_available() else "cpu")

        # Logging
        """
        self.logger = FileLogger(self.run_dir)
        self.logger.start_run(self.cfg["experiment"]["name"])
        """

        # Seed
        # set_all_seeds(self.cfg.get("seed", 42))

        # Model
        self.model = ModelFactory.build(cfg=self.cfg.model)
        self.model.to(self.device)

        self.criterion = LossFactory.build(cfg=self.cfg.train.loss)
        self.optimizer = OptimizerFactory.build(model=self.model, cfg=self.cfg.train.optim)
        self.scheduler = SchedulerFactory.build(optimizer=self.optimizer, cfg=self.cfg.train.scheduler)

        # Inspector

        self.inspector = ModelInspector(model=self.model, config=self.cfg.inspect, device=self.device)

        # Data
        self.train_loader, self.val_loader = build_dataloaders(self.cfg)

    # ---------- Public API ----------

    def run(self):
        print("Starting experiment")
