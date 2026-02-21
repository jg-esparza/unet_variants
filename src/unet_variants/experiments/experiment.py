from __future__ import annotations

import time
from typing import Dict, Optional

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
from unet_variants.engine.evaluate import evaluate
from unet_variants.metrics.binary import BinarySegmentationMetrics
from unet_variants.utils.logging import MLFlowLogger
from unet_variants.utils.early_stopping import EarlyStopping
from unet_variants.utils.visualization import choose_visualizer

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
        self._build_components()
        self._prepare_training_state()
        # Control
        self.best_val_loss = float("inf")
        self.best_epoch = 1

    # ---------- Build / Prepare ----------
    def _build_components(self) -> None:
        """Build model, loss, optimizer, scheduler, inspector, dataloaders, device."""
        # Device
        self.device = self._select_device()
        # Logging
        self.logger = MLFlowLogger(self.cfg.logging)
        # Seed
        # set_all_seeds(self.cfg.get("seed", 42))
        # Model
        self.model = ModelFactory.build(cfg=self.cfg.model)
        self.model.to(self.device)
        # Inspector
        self.inspector = ModelInspector(model=self.model, config=self.cfg.inspect, device=self.device)
        # Loss Function
        self.criterion = LossFactory.build(cfg=self.cfg.train.loss)
        # Optimization strategy
        self.optimizer = OptimizerFactory.build(model=self.model, cfg=self.cfg.train.optim)
        self.scheduler = SchedulerFactory.build(optimizer=self.optimizer, cfg=self.cfg.train.scheduler)
        # Early Stopper
        self.early_stopper = EarlyStopping(cfg=self.cfg.train.early_stopping)
        # Data
        self.train_loader, self.val_loader = build_dataloaders(self.cfg)
        # Segmentation Metrics
        self.metrics = BinarySegmentationMetrics(cfg=self.cfg.eval)

    def _prepare_training_state(self) -> None:
        """Initialize counters/metrics/seed and best weights."""
        # self._set_all_seeds(int(getattr(self.cfg.train, "seed", 42)))
        self.start_epoch = 1
        self.num_epochs = self.cfg.train.epochs
        self.vis_interval = self.cfg.train.vis.interval
        self.vis_sample_size = self.cfg.train.vis.sample_size
        self.vis_threshold = self.cfg.train.vis.threshold

    # ---------- Public API ----------

    def model_summary(self) -> None:
        """
        Print a formatted model summary from torchinfo.
        """
        self.inspector.model_summary(print_summary = True,
                                     save_summary = False,
                                     save_path= None)

    def model_flops(self) -> None:
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
        self._log_run_metadata()
        self._run_training_loop()
        self.evaluate()
        self.logger.end_run()


    def resume(self, run_id: str) -> None:
        """
        Resume training from the last checkpoint of a prior run.
        """
        # TODO: what if run is already done?
        self.logger.run_id=run_id
        self.logger.set_artifact_location()
        # Log model stats/flops/onnx for this run's artifacts (optional)
        self._generate_model_report()
        # Load state from checkpoint
        self._load_checkpoint()
        # Continue the run under the same MLflow run_id
        self.logger.start_run(run_id=run_id)
        self._run_training_loop()
        self.evaluate()
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
        for epoch in tqdm(range(self.start_epoch, self.num_epochs + 1)):
            # if torch.cuda.is_available():
            torch.cuda.empty_cache()
            start_t = time.time()
            # ---- Train ----
            train_metrics = train_one_epoch(self.model, self.criterion, self.optimizer, self.train_loader, self.device)
            # ---- Validate ----
            val_metrics = validate_one_epoch(self.model, self.criterion, self.val_loader, self.device, self.metrics)
            # ---- Scheduler ----
            self.scheduler.step()
            # ---- Logging ----
            self.log_metrics(epoch, train_metrics, val_metrics, start_t)
            # ---- Checkpoints ----
            self._save_checkpoint(epoch)
            # ---- Evaluate val loss ----
            self._save_best_model_if_best_loss(epoch, val_metrics["val/loss"])
            # ---- Sample images ----
            if epoch % self.vis_interval == 0:
                self.save_prediction_sample(epoch)
            # ---- Early Stopper ----
            if self.early_stopper.step(val_metrics["val/loss"]):
                break

    def _log_run_metadata(self) -> None:
        """
        Log run-wide information (tags and params).
        """
        self._generate_model_report()
        self.logger.set_tags(self.logger.tags)
        self.logger.log_params(self.logger.params)

    def _generate_model_report(self) -> None:
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
        self.logger.tags["total_params"] = int(report.total_params)
        self.logger.params["trainable_params"] = int(report.trainable_params)
        self.logger.params["flops"] = float(report.flops)
        self.logger.params["macs"] = float(report.macs)

    def _current_lrs(self) -> float:
        """
        Return current learning rate from optimizer.
        """
        return self.optimizer.state_dict()['param_groups'][0]['lr']

    @staticmethod
    def get_epoch_time(start_t: float) -> float:
        """
        Return elapsed seconds since `start_t`.
        """
        return time.time() - start_t

    def log_metrics(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float], start_t: float) -> None:
        """
        Merge and log metrics to MLflow for a given epoch.
        """
        train_metrics["lr"] = self._current_lrs()
        train_metrics["epoch_time"] = self.get_epoch_time(start_t)
        # change this
        metrics_to_log = {**train_metrics, **val_metrics}
        self.logger.log_metrics(metrics_to_log, epoch)

    def _save_best_model_if_best_loss(self, epoch: int, val_loss: float) -> None:
        """
        If the current validation loss improves the best validation loss.
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.save_best_model()

    def save_best_model(self) -> None:
        """
        Save the best-performing model's weights (state_dict) to the current run's
        artifact directory as ``best.pth``.
        """
        torch.save(self.model.state_dict(),  self.logger.artifact_path("best.pth"))

    def _save_checkpoint(self, epoch: int) -> None:
        """
        Persist a training checkpoint with:
        - epoch, best_epoch, best_val_loss
        - model/optimizer/scheduler state
        - loss (criterion) object
        - best model weights snapshot
        """
        torch.save({
            "epoch": epoch,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            }, self.logger.artifact_path("latest.pth"))

    def _load_checkpoint(self):
        """
        Load the latest checkpoint from the logger's artifact path and
        restore training state.
        """
        checkpoint = torch.load(self.logger.artifact_path("latest.pth"),
                                map_location=torch.device('cpu'))
        saved_epoch = checkpoint['epoch']
        self.best_epoch = checkpoint['best_epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = saved_epoch + 1
        print(f'Resume training {self.logger.run_id} from checkpoint\n'
              f'Running from epoch {self.start_epoch} of {self.num_epochs}')

    def save_prediction_sample(self, epoch: Optional[int] = None) -> None:
        """
        Save sample predictions to MLflow artifacts:
        - Pull a batch from val_loader
        - Run model in eval + no_grad
        - Save either a grid with sample_size rows (default) or a single triplet
        """
        self.model.eval()
        with torch.no_grad():
            # Get a batch
            batch = next(iter(self.val_loader))
            images, masks = batch["image"].to(self.device), batch["mask"]
            preds = self.model(images)

        # Move to CPU for visualization
        images_cpu = images.detach().cpu()
        masks_cpu = masks.detach().cpu()
        preds_cpu = preds.detach().cpu()
        if self.vis_threshold:
            preds_cpu = torch.where(preds_cpu >= 0.5, 1, 0)

        batch_size = images_cpu.shape[0]
        sample_size = self.vis_sample_size if self.vis_sample_size < batch_size else batch_size
        visualizer = choose_visualizer(sample_size=sample_size)
        if epoch is None:
            subtitle = f"Evaluation"
            artifact_file = f"Evaluation_sample.png"
        else:
            subtitle = f"Prediction sample epoch {epoch}"
            artifact_file = f"Sample_{epoch}.png"
        fig = visualizer(subtitle=subtitle, images=images_cpu, masks=masks_cpu, preds=preds_cpu, sample_size=sample_size)
        self.logger.log_figure(fig, artifact_file)

    def evaluate(self):
        metrics = evaluate(model=self.model, test_loader=self.val_loader, device=self.device, metrics=self.metrics)
        if self.logger.active_run is not None:
            self.logger.log_metrics(metrics)
            self.save_prediction_sample()
        print("Segmentation Results")
        print(f'DSC: {metrics["seg/dsc"]:.3f}')
        print(f'IoU: {metrics["seg/iou"]:.3f}')
        print(f'Acc: {metrics["seg/acc"]:.3f}')
        print(f'Sen: {metrics["seg/sen"]:.3f}')
        print(f'Spe: {metrics["seg/spe"]:.3f}')
