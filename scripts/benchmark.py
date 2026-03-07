from __future__ import annotations

from unet_variants.utils.bootstrap import set_repo_root_env
set_repo_root_env()  # must run before hydra.main

from typing import Dict, Any

import hydra
from omegaconf import DictConfig

from unet_variants.experiments.experiment import ExperimentManager
from unet_variants.utils.io import make_path, write_csv_row

def append_to_aggregated_csv(
        records: Dict[str, Any],
        report_path: str,
        file_name: str = "benchmark.csv"
    ) -> None:
    """
    Append one or more benchmark records to a consolidated CSV table.
    If the CSV doesn't exist, it's created with a header.
    Returns the CSV path for convenience.
    """
    csv_path = make_path(report_path, file_name)
    # Define the canonical column order for readability
    columns = [
        "model",
        "variant",
        "params_total",
        "macs",
        "flops",
        # input
        "batch",
        "channels",
        "image_size"
    ]
    write_csv_row(csv_path, columns, records)

@hydra.main(version_base="1.3", config_path="../configs", config_name="benchmark")
def main(cfg: DictConfig) -> None:
    print("=== Config (resolved) ===")
    exp = ExperimentManager(cfg)
    print("=== Build experiment API (resolved)===")
    print(f"=== Model  {cfg.model.name} ===")
    print(f"=== Benchmark ===")
    record = {"model": cfg.model.name,
              "variant": cfg.model.family,
              # input
              "batch": cfg.inspect.batch_size,
              "channels": cfg.inspect.in_channels,
              "image_size": cfg.inspect.image_size}
    if cfg.inference_bench:
        print(f"=== Inference Benchmark ===")
        report = exp.inspector.report(
            print_summary=cfg.torchinfo_summary.print_summary,
            export_summary=cfg.torchinfo_summary.export_summary,
            save_summary_path=cfg.torchinfo_summary.export_path,
            print_flops_report=cfg.flops_thop.print_report,
            export_flops_report=cfg.flops_thop.export_report,
            save_flops_txt=cfg.flops_thop.save_txt,
            save_flops_json=cfg.flops_thop.save_json,
            export_onnx=cfg.export_onnx.enable,
            export_onnx_path=cfg.export_onnx.path
        )
        record["params_total"] = report.total_params
        record["macs"]= report.macs
        record["flops"]= report.flops
        append_to_aggregated_csv(record, cfg.bench_dir)
        print(f"Results saved into {cfg.bench_dir}")
    if cfg.segmentation_bench:
        print("=== Segmentation Benchmark ===")
        print(f"=== Experiment  {cfg.logging.experiment_name} ===")
        if cfg.project.use_pretrained_ckpt:
            print("===Use pretrained checkpoint active---")
            exp.load_pretrained_ckpt()
        exp.run()


if __name__ == "__main__":
    main()