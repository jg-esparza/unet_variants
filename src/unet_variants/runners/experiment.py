
from __future__ import annotations

from omegaconf import DictConfig

import torch

from unet_variants.models.factory import ModelFactory
from unet_variants.inspection.inspector import ModelInspector
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
        self.model = ModelFactory.create(self.cfg.model.name, self.cfg.model)
        self.model.to(self.device)

        # Inspector
        self.inspector = ModelInspector(model=self.model, config=self.cfg.inspect, device=self.device)



        # Data
        self.train_loader, self.val_loader = build_dataloaders(self.cfg)


    # ---------- Public API ----------

    def run(self):
        print("Starting experiment")

    def resume(self):
        """
        Manual resume: load checkpoint and return self (chainable).
        """
        # self.ckpt.try_resume(self.model, self.optimizer, self.scheduler, checkpoint_path, device=self.device)
        print("Resuming from checkpoint")
        return self

    # ---- Inspector convenience methods (delegate to ModelInspector) ----

    def model_summary(self):
        print("Summary")

    def model_flops(self):
        print("Flops")


    def export_onnx(self):
        print("Exporting onnx model")

    def view_inspection(self):
        """Console-friendly structure printout via inspector."""
        print("Inspection")

    def log_artifact(self):
       print("Logging artifact")

    # ---------- Internals ----------

    def _select_device(self):
        print("Select device")

    def _build_optim(self):
        print("Building optim")

    def _build_scheduler(self):
        print("Building scheduler")

    def _resolve_exp_name(self):
        print("Resolve experiment name")

    def _run_inspector(self):
        print("Run inspector")
