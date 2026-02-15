from __future__ import annotations

import copy
import time
from typing import Dict

from mlflow import ActiveRun
from omegaconf import DictConfig

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

    Responsibilities
    ---------------
    - Config intake / device selection
    - Logging initialization (MLflow) + run metadata
    - Data/model/loss/optimizer/scheduler setup
    - Training & validation loop
    - Checkpoints (best/last) + resume support
    - Model inspection (summary/FLOPs/ONNX)

    Parameters
    ----------
    config : DictConfig
        Hydra/OmegaConf configuration object. Expected keys:
        - project.device : Optional[str], e.g. "cuda", "cuda:1", "cpu", "mps"
        - logging : MLflow logger configuration consumed by MLFlowLogger
        - model : model configuration for ModelFactory
        - train :
            - epochs : int
            - loss : loss configuration for LossFactory
            - optim : optimizer configuration for OptimizerFactory
            - scheduler : scheduler configuration for SchedulerFactory (optional)
            - vis_interval : Optional[int], visualization interval in epochs
        - inspect : ModelInspector configuration
        - data : (consumed by build_dataloaders via cfg)
    """

    def __init__(self, config: DictConfig = None):

        # Store config
        self.cfg = config

        # Device
        self.device = self._select_device()

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

    def model_summary(self) -> None:
        """
        Print a formatted model summary from torchinfo.
        """
        self.inspector.model_summary(print_summary = True,
                                     save_summary = False,
                                     save_path= None)

    def model_flops_thop(self) -> None:
        """
        Print a FLOPs/MACs report to stdout.
        """
        self.inspector.model_flops(print_report = True,
                                   save_report = False,
                                   save_txt = None,
                                   save_json = None)

    def model_onnx(self, view: bool = False) -> None:
        """
        Export the current model to ONNX and optionally open a local viewer.
        """
        onnx_path = self.inspector.export_onnx(export_path=self.cfg.inspect.export_onnx.path)
        print("ONNX export path: {}".format(onnx_path))
        if view:
            self.inspector.view_onnx(onnx_path=onnx_path, port= 8081, host= "127.0.0.1", browse= True)

    def run(self) -> None:
        """
        Start a fresh MLflow run and execute the training loop.
        """
        self.logger.start_run()
        print("Starting new run {}".format(self.logger.run_id))
        self._log_run_info()
        self._run_training_loop()
        self.logger.end_run()

    def resume(self, run_id: str) -> None:
        """
        Resume training from the last checkpoint of a prior run.

        Parameters
        ----------
        run_id : str
            MLflow run ID to resume.
        """
        self.logger.run_id=run_id
        self.logger.set_artifact_location()
        # Log model stats/flops/onnx for this run's artifacts (optional)
        self._report_for_current_run()
        # Load state from checkpoint
        self.load_checkpoint()
        # Continue the run under the same MLflow run_id
        self.logger.start_run(run_id=run_id)
        self._run_training_loop()
        self.logger.end_run()

    # ---------- Internal Helpers ----------

    def _select_device(self) -> torch.device:
        """
        Select a compute device.
        """
        return torch.device(self.cfg.project.device if torch.cuda.is_available() else "cpu")

    def _run_training_loop(self) -> None:
        """
        Execute the core training loop across epochs:
        - training epoch
        - validation epoch
        - scheduler step
        - logging + checkpoints + best - weights tracking
        """
        for epoch in tqdm(range(self.start_epoch, self.max_epochs + 1)):
            torch.cuda.empty_cache()
            start_t = time.time()
            # ---- Train ----
            train_metrics = train_one_epoch(self.model, self.criterion, self.optimizer, self.train_loader, self.device)
            # ---- Validate ----
            val_metrics = validate_one_epoch(self.model, self.criterion, self.val_loader, self.device)
            # ---- Scheduler ----
            self.scheduler.step()
            # ---- Logging ----
            if isinstance(self.logger.active_run, ActiveRun):
                self.log_metrics(epoch, train_metrics, val_metrics, start_t)
            # ---- Checkpoints ----
            self.save_checkpoint(epoch)
            self.save_weights_if_best_loss(epoch)

            if epoch % self.cfg.train.vis_interva == 0:
                # self.save_sample_prediction(epoch, sample_size=3) still pending, i need some ideas to get images, masks and predictions to show
                print("Vis")

    def _log_run_info(self) -> None:
        """Log run-wide information (tags and params)."""
        self._report_for_current_run()
        self.logger.set_tags(self.logger.tags)
        self.logger.log_params(self.logger.params)

    def _report_for_current_run(self) -> None:
        """
        Produce a model report (summary, FLOPs, and optional ONNX) and
        populate MLflow tags/params accordingly.
        """

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

    def _get_current_lr(self) -> float:
        """Return current learning rate from optimizer."""
        return self.optimizer.state_dict()['param_groups'][0]['lr']

    @staticmethod
    def get_epoch_time(start_t: float) -> float:
        """Return elapsed seconds since `start_t`."""
        return time.time() - start_t

    def log_metrics(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float], start_t: float) -> None:
        """
        Merge and log metrics to MLflow for a given epoch.

        Notes
        -----
        Adds:
        - lr: first param group LR (for convenience)
        - epoch_time: epoch wall-clock duration (seconds)
        """
        train_metrics["lr"] = self._get_current_lr()
        train_metrics["epoch_time"] = self.get_epoch_time(start_t)
        metrics_to_log = {**train_metrics, **val_metrics}
        self.logger.log_metrics(metrics_to_log, epoch)

    def save_weights_if_best_loss(self, epoch: int) -> None:
        """
        If the current validation loss improves the best loss,
        keep a copy of the best model weights.
        """
        if self.val_loss < self.min_loss:
            self.min_loss = self.val_loss
            self.best_model_wts = copy.deepcopy(self.model.state_dict())
            self.min_epoch = epoch

    def save_checkpoint(self, epoch: int) -> None:
        """
        Persist a training checkpoint with:
        - epoch, best_epoch, min_loss
        - model/optimizer/scheduler state
        - loss (criterion) object
        - best model weights snapshot
        """
        torch.save({
            "epoch": epoch,
            "best_epoch": self.min_epoch,
            "min_loss": self.min_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": self.criterion,
            "best_model_wts": self.best_model_wts,
            }, self.logger.artifact_path("checkpoint.pth"))

    def load_checkpoint(self):
        """
        Load the latest checkpoint from the logger's artifact path and
        restore training state.
        """
        checkpoint = torch.load(self.logger.artifact_path("checkpoint.pth"),
                                weights_only=False)
        saved_epoch = checkpoint['epoch']
        self.min_epoch = checkpoint['best_epoch']
        self.min_loss = checkpoint['min_loss']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.criterion = checkpoint['loss']
        self.best_model_wts = checkpoint['best_model_wts']
        self.start_epoch = saved_epoch + 1
        print(f'Resume training {self.logger.run_id} from checkpoint\n'
              f'Running from epoch {self.start_epoch} of {self.max_epochs}')
