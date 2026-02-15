from __future__ import annotations

import os
from typing import Any, Dict, Optional, Literal
from omegaconf import DictConfig

import mlflow

class MLFlowLogger:
    """
    A wrapper class to handle MLflow lifecycle, logging, and artifact management.
    """

    """
    A lightweight wrapper around MLflow lifecycle, logging, and artifact management.

    Typical usage:
        logger = MlflowLogger(experiment_name="MyExperiment", tracking_uri="http://localhost:5000")
        with logger.start_run():
        logger.log_params({...})
        logger.log_metrics({"train/loss": 0.123}, step=1)
        logger.log_artifact("path/to/file")
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Args:
            cfg (DictConfig): logging configuration. MLFlow as default.
        """
        self.cfg = cfg
        self.run_name = self.cfg.run_name
        self.experiment_name = self.cfg.experiment_name

        mlflow.set_tracking_uri(self.cfg.tracking_uri)
        # Create or get experiment
        exp = mlflow.get_experiment_by_name(self.experiment_name)
        if exp is None:
            mlflow.set_experiment(self.experiment_name)
            exp = mlflow.get_experiment_by_name(self.experiment_name)

        self.experiment_id = exp.experiment_id
        self.active_run = None
        self.run_id = None
        self.artifact_location = None
        self.tags = self.cfg.tags
        self.params = self.cfg.params
        # self.temp_path = self.cfg.temp_path
        # self._create_temp_folder()

        # ---------- Context management ----------

    def __enter__(self) -> "MLFlowLogger":
        # Using MlflowLogger as context without automatically starting a run is allowed.
        # You can still explicitly call start_run() inside the with-block.
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Ensure run is closed if left open
        if self.active_run is not None:
            self.end_run(status="FAILED" if exc_type else "FINISHED")

    def start_run(self, run_id: Optional[str] = None,
                  ) -> "MLFlowLogger":
        """
        Start or resume an MLflow run.

        Args:
            run_id: If provided, resumes an existing run id.

        Returns:
            self (to allow `with logger.start_run(...):` usage)
        """
        if self.active_run is not None:
            raise RuntimeError("An MLflow run is already active. End it before starting another.")

        self.active_run = mlflow.start_run(
            run_id=run_id,
            experiment_id=self.experiment_id,
            run_name=self.cfg.run_name
        )
        self.run_id = self.active_run.info.run_id
        self._setup_system_metrics_sampling()
        self.set_artifact_location()
        return self

    def end_run(self, status: Literal["FINISHED", "FAILED", "KILLED"] = "FINISHED") -> None:
        """End the currently active run with a status."""
        if self.active_run is not None:
            mlflow.end_run(status=status)
            self.active_run = None

    def set_artifact_location(self) -> None:
        self.artifact_location = os.path.join(self.cfg.mlruns_path, self.experiment_id, self.run_id, "artifacts")

    def artifact_path(self, file: str) -> os.path:
        return os.path.join(self.artifact_location, file)

    # def _create_temp_folder(self) -> None:
    #     os.makedirs(self.temp_path, exist_ok=True)

    def _setup_system_metrics_sampling(self):
        """Set the system metrics sampling if installed."""
        mlflow.set_system_metrics_sampling_interval(self.cfg.system_metrics.sampling_interval)
        mlflow.set_system_metrics_samples_before_logging(self.cfg.system_metrics.samples_before_logging)

    # ---------- Logging helpers ----------
    @staticmethod
    def set_tags(tags) -> None:
        mlflow.set_tags(tags)

    @staticmethod
    def log_params(params: Dict[str, Any]) -> None:
        # flat = self._flatten_dict(params, parent_key=prefix)
        mlflow.log_params(params)

    @staticmethod
    def log_param(key: str, value: Any) -> None:
        mlflow.log_param(key, value)

    @staticmethod
    def log_metrics(metrics: Dict[str, float], step: Optional[int] = None, prefix: str = "") -> None:
        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        mlflow.log_metrics(metrics, step=step)

    @staticmethod
    def log_artifact(local_path: str, artifact_path: Optional[str] = None) -> None:
        mlflow.log_artifact(local_path, artifact_path=artifact_path)

    @staticmethod
    def log_artifacts(local_dir: str, artifact_path: Optional[str] = None) -> None:
        mlflow.log_artifacts(local_dir, artifact_path=artifact_path)
