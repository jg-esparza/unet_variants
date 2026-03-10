from __future__ import annotations

from typing import Optional, Tuple, Union, Dict

from omegaconf import DictConfig

import torch
from torch import nn

from thop import profile
from torchinfo import summary as torchinfo_summary

from unet_variants.utils.io import save_text, is_file


def to_readable_units(x: int) -> str:
    """Convert a large scalar to readable units (K, M, G, T)."""
    if x is None:
        return "N/A"
    units = ["", "K", "M", "G", "T", "P"]
    v = float(x)
    i = 0
    while abs(v) >= 1000.0 and i < len(units) - 1:
        v /= 1000.0
        i += 1
    return f"{v:.3f}{units[i]}"

class ModelInspector:
    """
    High-level model inspection facade.

    Provides:
    - parameter counts
    - torchinfo summary (string + optional file saving)
    - FLOPs via THOP (structured)
    - ONNX export + view
    """

    def __init__(self, model: nn.Module, cfg: DictConfig, device: Union[str, torch.device] = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.b = int(self.cfg.batch_size)
        self.c = int(self.cfg.in_channels)
        self.h = int(self.cfg.image_size)
        self.w = int(self.cfg.image_size)
        self.input_size = (self.b, self.c,self.h, self.w)

    def count_params(self, verbose: Optional[bool] = False) -> Tuple[int, int]:
        """Estimate total number of parameters and trainable parameters."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if verbose:
            print(f"\n===Parameter Count==="
                  f"\nTotal number of parameters: {total} ({to_readable_units(total)} Parameters)"
                  f"\nTrainable parameters: {trainable} ({to_readable_units(trainable)} Parameters)")
        return int(total), int(trainable)

    def estimate_flops_thop(self):
        """Estimate FLOPs/MACs using THOP."""
        return profile(self.model, inputs=(torch.randn(self.b, self.c, self.h, self.w, device=self.device),), verbose=False)

    @torch.no_grad()
    def model_flops(self, verbose: Optional[bool] = False) -> Tuple[float, float]:
        """Estimate FLOPs and parameters via THOP, optionally print."""
        flops, params_profiler = self.estimate_flops_thop()
        if verbose:
            print("\n" + self.format_flops(flops, params_profiler))
        return flops, params_profiler

    def format_flops(self, flops: int, params_profiler:int) -> str:
        """Format FLOPs/MACs results as a readable report string."""
        lines = [
            "=== FLOPs / Params (profiler) ===",
            f"Input size: (B={self.b}, C={self.c}, H={self.h}, W={self.w})",
            f"FLOPs: {flops} ({to_readable_units(flops)} FLOPs)",
            f"Params (profiler): {params_profiler} ({to_readable_units(params_profiler)} Parameters)",
        ]
        return "\n".join(lines)

    def model_summary(self, verbose: Optional[bool] = True, export: Optional[bool] = True) -> None:
        """Generate torchinfo summary as string, optionally print and/or save to file."""
        print("\n=== torchinfo summary ===")
        summary = torchinfo_summary(
            self.model,
            input_size=self.input_size,
            device=str(self.device),
            verbose=1 if verbose else 0,
            col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"),
        )
        if export:
            try:
                save_text(str(summary), self.cfg.summary_path)
                print("=== Saved torchinfo summary ===")
            except Exception as e:
                print(f"Failed to save torchinfo summary: {e}")

    def get_report(self, verbose: Optional[bool] = False) -> Dict[str, int]:
        """Get model report including parameter count and total operations"""
        total_params, trainable_params = self.count_params(verbose)
        flops, params_profiler = self.model_flops(verbose)
        return {"model": self.cfg.model_name,
                "input_size": self.input_size,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "flops": flops,
                "params_profiler": params_profiler}

    @torch.no_grad()
    def export_onnx(self) -> None:
        """Export ONNX graph."""
        try:
            torch.onnx.export(self.model,  # model being run
                              torch.randn(self.b, self.c, self.h, self.w, device=self.device),  # model input (or a tuple for multiple inputs)
                              self.cfg.onnxx_path,  # where to save the model (can be a file or file-like object)
                              input_names=['input'],  # the model's input names
                              output_names=['output'])
        except RuntimeError as e:
            print(f"Failed to export ONNX: {e}")

    def view_onnx(self, port: int = 8081, host: str = "127.0.0.1", browse: bool = True) -> None:
        """View ONNX graph if Netron is installed."""
        if not is_file(self.cfg.onnxx_path):
            raise FileNotFoundError(f"ONNX file not found: {self.cfg.onnxx_path}")

        try:
            import netron
            netron.start(self.cfg.onnxx_path, address=(host, port), browse=browse)
            print(f"Netron started at http://{host}:{port} (showing {self.cfg.onnxx_path})")
        except ImportError:
            print("Netron is not installed.")
            print("Option A: pip install netron and rerun view_onnx (jupyter notebook)")
            print("Option B: open the file in the web viewer at https://netron.app/")
