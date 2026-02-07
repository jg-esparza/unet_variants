from __future__ import annotations

from typing import Tuple
import os
import torch
from torch import nn


def export_onnx(
    model: nn.Module,
    export_path: str,
    input_size: Tuple[int, int, int, int],
    device: torch.device,
    opset: int = 17,
) -> str:
    """
    Export model to ONNX for visualization in Netron.
    """
    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    b, c, h, w = input_size
    dummy = torch.randn(b, c, h, w, device=device)

    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            export_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
    return export_path
