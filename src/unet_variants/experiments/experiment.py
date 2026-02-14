from __future__ import annotations

from omegaconf import DictConfig

import copy
import torch

from tqdm import tqdm

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

        # Loss Function
        self.criterion = LossFactory.build(cfg=self.cfg.train.loss)

        # Optimization strategy
        self.optimizer = OptimizerFactory.build(model=self.model, cfg=self.cfg.train.optim)
        self.scheduler = SchedulerFactory.build(optimizer=self.optimizer, cfg=self.cfg.train.scheduler)

        # Inspector
        self.inspector = ModelInspector(model=self.model, config=self.cfg.inspect, device=self.device)

        # Data
        self.train_loader, self.val_loader = build_dataloaders(self.cfg)

    # ---------- Public API ----------

    def _setup_metrics(self):
        """self.metrics = {"min_loss": float("inf"),
                        "min_epoch": 1,
                        "train_loss": 0,
                        "val_loss": 0}
                        """
        self.train_loss = 0
        self.val_loss = 0
        self.min_loss = float("inf")
        self.min_epoch = 1

    def model_summary(self):
        self.inspector.model_summary(print_summary = True,
                                     save_summary = False,
                                     save_path= None)

    def model_flops_thop(self):
        self.inspector.model_flops(print_report = True,
                                   save_report = False,
                                   save_txt = None,
                                   save_json = None)

    def model_onnx(self):
        onnx_path = self.inspector.export_onnx(export_path=self.cfg.inspect.export_onnx.path)
        self.inspector.view_onnx(onnx_path=onnx_path, port= 8081, host= "127.0.0.1", browse= True)
