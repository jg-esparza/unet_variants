
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Any

import os
import torch
from torch import nn


@dataclass
class InspectionReport:
    total_params: int
    trainable_params: int
    flops: Optional[float] = None
    macs: Optional[float] = None
    onnx_path: Optional[str] = None


class ModelInspector:
    """
    Model inspector.

    Features:
      - torchinfo summary
      - parameter counts
      - FLOPs/MACs estimate (THOP if installed)
      - export to ONNX
      - view ONNX graph via Netron (if installed) or netron.app
    """

    def __init__(
        self,
        model: nn.Module,
        device: Union[str, torch.device] = "cpu",
    ):
        self.model = model
        self.device = torch.device(device)
        self.model = self.model.to(self.device).eval()

    # -------------------------
    # Basics
    # -------------------------
    def count_params(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return int(total), int(trainable)

    # -------------------------
    # torchinfo summary
    # -------------------------
    def summary(
        self,
        input_size: Tuple[int, int, int, int],
        verbose: int = 1,
    ) -> None:
        """
        Print a torchinfo summary. input_size should be (B, C, H, W).
        """
        try:
            from torchinfo import summary as torchinfo_summary
        except ImportError as e:
            raise ImportError("Install torchinfo: pip install torchinfo") from e

        print("\n=== torchinfo summary ===")
        torchinfo_summary(
            self.model,
            input_size=input_size,
            device=str(self.device),
            verbose=verbose,
            col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"),
        )

    # -------------------------
    # FLOPs / MACs
    # -------------------------
    @torch.no_grad()
    def flops(
        self,
        input_size: Tuple[int, int, int, int],
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Estimate FLOPs/MACs using THOP if available.
        Returns (flops, macs) or (None, None) if THOP isn't installed or fails.
        """
        try:
            from thop import profile
        except ImportError:
            return None, None

        b, c, h, w = input_size
        x = torch.randn(b, c, h, w, device=self.device)

        try:
            macs, _params = profile(self.model, inputs=(x,), verbose=False)
            flops = 2.0 * macs  # common approximation: FLOPs ≈ 2 * MACs
            return float(flops), float(macs)
        except Exception:
            return None, None

    # -------------------------
    # Export ONNX
    # -------------------------
    @torch.no_grad()
    def export_onnx(
        self,
        export_path: str,
        input_size: Tuple[int, int, int, int],
        opset: int = 17,
    ) -> str:
        """
        Export ONNX using torch.onnx.export.
        This produces a standardized graph file that can be viewed in Netron. [2](https://docs.pytorch.org/docs/stable/onnx_export.html)[3](http://docs.pytorch.wiki/en/onnx.html)
        """
        os.makedirs(os.path.dirname(export_path), exist_ok=True)

        b, c, h, w = input_size
        dummy = torch.randn(b, c, h, w, device=self.device)

        self.model.eval()

        # PyTorch exporter (classic API). [3](http://docs.pytorch.wiki/en/onnx.html)
        result = torch.onnx.export(
            self.model,
            dummy,
            export_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )

        # Some newer flows can return an ONNXProgram; keep this robust:
        if hasattr(result, "save"):
            result.save(export_path)

        return export_path

    # -------------------------
    # View ONNX (Netron)
    # -------------------------
    def view_onnx(
        self,
        onnx_path: str,
        port: int = 8081,
        host: str = "127.0.0.1",
        browse: bool = True,
    ) -> None:
        """
        View an ONNX file using Netron.
        - If netron Python package is installed, starts a local viewer server.
        - Otherwise, prints instructions to open via https://netron.app/ or the desktop app. [4](https://github.com/lutzroeder/netron)[6](https://netron.app/)[5](https://pypi.org/project/netron/)
        """
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

        try:
            import netron
            # Netron supports Python usage: `pip install netron`, run netron.start(file). [4](https://github.com/lutzroeder/netron)[5](https://pypi.org/project/netron/)
            netron.start(onnx_path, address=(host, port), browse=browse)
            print(f"Netron started at http://{host}:{port} (showing {onnx_path})")
        except ImportError:
            print("Netron is not installed.")
            print("Option A (recommended): pip install netron  (then rerun view_onnx)")
            print("Option B: open the file in the web viewer at https://netron.app/")
            print("Option C: install the Netron desktop app from the Netron GitHub releases.")

    # -------------------------
    # One-shot report
    # -------------------------
    def report(
        self,
        input_size: Tuple[int, int, int, int],
        export_onnx_path: Optional[str] = None,
        print_summary: bool = True,
    ) -> InspectionReport:
        total, trainable = self.count_params()

        if print_summary:
            self.summary(input_size=input_size)

        flops, macs = self.flops(input_size=input_size)

        onnx_path = None
        if export_onnx_path is not None:
            onnx_path = self.export_onnx(export_onnx_path, input_size=input_size)

        return InspectionReport(
            total_params=total,
            trainable_params=trainable,
            flops=flops,
            macs=macs,
            onnx_path=onnx_path,
        )
