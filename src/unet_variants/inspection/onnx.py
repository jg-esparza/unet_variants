
from __future__ import annotations

import os
from typing import Tuple

import torch
from torch import nn


@torch.no_grad()
def export_onnx_graph(
    model: nn.Module,
    export_path: str,
    input_size: Tuple[int, int, int, int],
    device: torch.device,
    opset: int = 17,
) -> str:
    """
    Export a model to ONNX format using torch.onnx.export.

    Parameters
    ----------
    model:
        Model to export (should be in eval mode).
    export_path:
        Output file path (e.g., "runs/onnx/model.onnx").
    input_size:
        Input tensor shape (B, C, H, W).
    device:
        Device for the dummy input.
    opset:
        ONNX opset version.

    Returns
    -------
    export_path:
        The path where the ONNX file was written.

    Notes
    -----
    - Uses a dummy input to trace the graph.
    - Sets dynamic axes for batch dimension.
    """
    export_dir = os.path.dirname(export_path)
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)

    b, c, h, w = input_size
    dummy = torch.randn(b, c, h, w, device=device)

    model.eval()

    result = torch.onnx.export(
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

    # Some exporter flows can return an object with .save()
    if hasattr(result, "save"):
        result.save(export_path)

    return export_path


def view_onnx_graph(
    onnx_path: str,
    host: str = "127.0.0.1",
    port: int = 8081,
    browse: bool = True,
) -> None:
    """
    View an ONNX file using Netron.

    If netron is installed, starts a local Netron server.
    Otherwise prints instructions for using https://netron.app/.

    Parameters
    ----------
    onnx_path:
        Path to the ONNX file.
    host:
        Host interface for Netron server.
    port:
        Port for Netron server.
    browse:
        Whether to automatically open a browser tab.

    Raises
    ------
    FileNotFoundError
        If `onnx_path` does not exist.
    """
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    try:
        import netron
        netron.start(onnx_path, address=(host, port), browse=browse)
        print(f"Netron started at http://{host}:{port} (showing {onnx_path})")
    except ImportError:
        print("Netron is not installed.")
        print("Option A: pip install netron  (then rerun view_onnx)")
        print("Option B: open the file in the web viewer at https://netron.app/")
