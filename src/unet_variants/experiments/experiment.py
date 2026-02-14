from __future__ import annotations

import os.path

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
from unet_variants.engine.train import train_one_epoch
from unet_variants.engine.validate import validate_one_epoch
from unet_variants.utils.logging import MLFlowLogger

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

        self.logger = MLFlowLogger(self.cfg.logging)

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

        # Training
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.start_epoch = 1
        self.max_epochs = self.cfg.train.epochs

        # Metrics
        self.train_loss = 0
        self.val_loss = 0
        self.min_loss = float("inf")
        self.min_epoch = 1


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

    def run(self):
        self.logger.start_run()
        self._log_run_info()
        for epoch in tqdm(range(self.start_epoch, self.max_epochs + 1)):
            torch.cuda.empty_cache()
            self.train_loss = train_one_epoch(self.model, self.criterion, self.optimizer, self.train_loader, self.device)
            self.val_loss = validate_one_epoch(self.model, self.criterion, self.val_loader, self.device)
            self.scheduler.step()
            self.log_metrics(epoch)

        self.logger.end_run()

    def _log_run_info(self):
        self._report_for_current_run()
        self.logger.set_tags(self.logger.tags)
        self.logger.log_params(self.logger.params)

    def _report_for_current_run(self):
        report = self.inspector.report(print_summary=False,
                                       export_summary=self.cfg.logging.save_model_summary,
                                       save_summary_path=self.logger.artifact_path("summary.txt"),
                                       print_flops_report=False,
                                       export_flops_report=self.cfg.logging.save_flops_report,
                                       save_flops_txt=self.logger.artifact_path("flops.txt"),
                                       save_flops_json=self.logger.artifact_path("flops.json"),
                                       export_onnx=self.cfg.logging.export_onnx,
                                       export_onnx_path=self.logger.artifact_path("model.onnx")
                                       )
        self.logger.tags["total_params"] = report.total_params
        self.logger.params["trainable_params"] = report.trainable_params
        self.logger.params["flops"] = report.flops
        self.logger.params["macs"] = report.macs

    def _get_current_lr(self):
        return self.optimizer.state_dict()['param_groups'][0]['lr']

    def log_metrics(self, epoch):
        self.logger.log_metrics({"lr": self._get_current_lr(),
                                 "train_loss": self.train_loss,
                                 "val_loss": self.val_loss
                                 },
                                epoch)