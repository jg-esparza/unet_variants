
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

from omegaconf import DictConfig

import torch
from torch import nn

from .summary import get_torchinfo_summary, save_torchinfo_summary
from .flops import FlopsResult, estimate_flops_thop, format_flops, save_flops_report
from .onnx import export_onnx_graph, view_onnx_graph


@dataclass
class InspectionReport:
    """
    Output of a one-shot model inspection.

    Attributes
    ----------
    total_params:
        Total number of parameters in the model.
    trainable_params:
        Trainable parameters (requires_grad=True).
    flops:
        FLOPs estimate (may be None).
    macs:
        MACs estimate (may be None).
    onnx_path:
        ONNX export path (if exported).
    summary_path:
        Path to saved torchinfo summary (if saved).
    flops_txt_path:
        Path to saved FLOPs report (txt) (if saved).
    flops_json_path:
        Path to saved FLOPs report (json) (if saved).
    """
    total_params: int
    trainable_params: int
    flops: Optional[float] = None
    macs: Optional[float] = None
    onnx_path: Optional[str] = None
    summary_path: Optional[str] = None
    flops_txt_path: Optional[str] = None
    flops_json_path: Optional[str] = None


class ModelInspector:
    """
    High-level model inspection facade.

    Provides:
    - parameter counts
    - torchinfo summary (string + optional file saving)
    - FLOPs/MACs via THOP (structured + optional txt/json saving)
    - ONNX export + view (no logging here yet)
    """

    def __init__(self, model: nn.Module, config: DictConfig = None, device: Union[str, torch.device] = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        b = int(config.batch_size)
        c = int(config.in_channels)  # recommended: interpolate from data.in_channels
        h = int(config.image_size)
        w = int(config.image_size)
        self.input_size = (b, c, h, w)

    def count_params(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return int(total), int(trainable)

    def summary(
        self,
        verbose: int = 0,
        print_summary: bool = True,
        save_summary: bool = False,
        save_path: Optional[str] = None,
    ) -> str:
        """
        Generate torchinfo summary as string, optionally print and/or save to file.
        """
        text = get_torchinfo_summary(self.model, self.input_size, self.device, verbose=verbose)

        if print_summary:
            print("\n=== torchinfo summary ===")
            print(text)

        if save_summary is not False:
            save_torchinfo_summary(self.model, self.input_size, self.device, path=save_path, verbose=verbose)
        return text

    @torch.no_grad()
    def flops(
        self,
        print_report: bool = True,
        save_report: bool = False,
        save_txt: Optional[str] = None,
        save_json: Optional[str] = None,
    ) -> FlopsResult:
        """
        Estimate FLOPs/MACs via THOP, optionally print and/or save as txt/json.
        """
        result = estimate_flops_thop(
            model=self.model,
            input_size=self.input_size,
            device=self.device
        )

        if print_report:
            print("\n" + format_flops(result))

        if save_report:
            save_flops_report(result, txt_path=save_txt, json_path=save_json)
        return result

    @torch.no_grad()
    def export_onnx(self, export_path: str, opset: int = 17) -> str:
        """
        Export ONNX graph. (Logging as MLflow artifact can be added in the runner later.)
        """
        return export_onnx_graph(self.model, export_path, self.input_size, self.device, opset=opset)

    @staticmethod
    def view_onnx(onnx_path: str, port: int = 8081, host: str = "127.0.0.1", browse: bool = True) -> None:
        view_onnx_graph(onnx_path=onnx_path, host=host, port=port, browse=browse)

    def report(
        self,
        export_onnx_path: Optional[str] = None,
        print_summary: bool = True,
        export_summary: bool = False,
        save_summary_path: Optional[str] = None,
        print_flops_report: bool = True,
        export_flops_report: bool = False,
        save_flops_txt: Optional[str] = None,
        save_flops_json: Optional[str] = None,
    ) -> InspectionReport:
        """
        Run a one-shot inspection pass and return a report.

        This is designed to integrate cleanly with MLflow later by producing
        file paths for artifacts (summary/flops reports/onnx).
        """
        total, trainable = self.count_params()
        """
        summary_path = None
        if print_summary or save_summary_path:
            self.summary(
                input_size=input_size,
                verbose=verbose,
                print_summary=print_summary,
                save_path=save_summary_path,
            )
            summary_path = save_summary_path
        """
        self.summary(
            print_summary=print_summary,
            save_summary=export_summary,
            save_path=save_summary_path,
        )

        flops_result = self.flops(
            print_report=print_flops_report,
            save_report=export_flops_report,
            save_txt=save_flops_txt,
            save_json=save_flops_json,
        )

        onnx_path = None
        if export_onnx_path is not None:
            onnx_path = self.export_onnx(export_onnx_path)

        return InspectionReport(
            total_params=total,
            trainable_params=trainable,
            flops=flops_result.flops,
            macs=flops_result.macs,
            onnx_path=onnx_path,
            summary_path=save_summary_path,
            flops_txt_path=save_flops_txt,
            flops_json_path=save_flops_json,
        )
