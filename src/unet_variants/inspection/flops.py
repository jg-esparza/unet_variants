from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any
import json
import os

import torch
from torch import nn

from unet_variants.utils.io import save_text, save_json

@dataclass
class FlopsResult:
    """
    Container for FLOPs profiling results.

    Attributes
    ----------
    flops:
        Estimated FLOPs for a forward pass (may be None if profiling unavailable).
    macs:
        Estimated MACs for a forward pass (may be None).
    params:
        Parameter count reported by the profiler (may be None).
    input_size:
        Input tensor shape used for profiling (B, C, H, W).
    note:
        Optional note about profiling assumptions (batch normalization, etc.).
    """
    flops: Optional[float]
    macs: Optional[float]
    params: Optional[int]
    input_size: Tuple[int, int, int, int]
    note: str = "FLOPs approximated as 2 * MACs (common convention)."


def _human_units(x: Optional[float]) -> str:
    """Convert a large scalar to human-readable units (K, M, G, T)."""
    if x is None:
        return "N/A"
    units = ["", "K", "M", "G", "T", "P"]
    v = float(x)
    i = 0
    while abs(v) >= 1000.0 and i < len(units) - 1:
        v /= 1000.0
        i += 1
    return f"{v:.3f}{units[i]}"


@torch.no_grad()
def estimate_flops_thop(
    model: nn.Module,
    input_size: Tuple[int, int, int, int],
    device: torch.device,
) -> FlopsResult:
    """
    Estimate FLOPs/MACs using THOP (if installed).

    Parameters
    ----------
    model:
        Model to profile.
    input_size:
        (B, C, H, W) input size.
    device:
        Device for dummy input.

    Returns
    -------
    FlopsResult:
        Contains flops, macs, params, input_size, and a note.
        If THOP isn't installed or profiling fails: flops/macs/params may be None.
    """
    try:
        from thop import profile
    except ImportError:
        return FlopsResult(flops=None, macs=None, params=None, input_size=input_size, note="THOP not installed.")

    b, c, h, w = input_size
    x = torch.randn(b, c, h, w, device=device)

    try:
        macs, params = profile(model, inputs=(x,), verbose=False)
        flops = 2.0 * macs
        return FlopsResult(flops=float(flops), macs=float(macs), params=int(params), input_size=(b, c, h, w))
    except Exception:
        return FlopsResult(flops=None, macs=None, params=None, input_size=(b, c, h, w), note="THOP profiling failed.")


def format_flops(result: FlopsResult) -> str:
    """
    Format FLOPs/MACs results as a readable report string.

    Parameters
    ----------
    result:
        FlopsResult instance.

    Returns
    -------
    text:
        Human-readable profiling report.
    """
    b, c, h, w = result.input_size
    lines = [
        "=== FLOPs / MACs Report ===",
        f"Input size: (B={b}, C={c}, H={h}, W={w})",
        f"MACs:  {result.macs}  ({_human_units(result.macs)} MACs)",
        f"FLOPs: {result.flops} ({_human_units(result.flops)} FLOPs)",
        f"Params (profiler): {result.params if result.params is not None else 'N/A'}",
        f"Note: {result.note}",
    ]
    return "\n".join(lines)


def save_flops_report(
    result: FlopsResult,
    txt_path: Optional[str] = None,
    json_path: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """
    Save FLOPs report as text and/or JSON for later MLflow logging.

    Parameters
    ----------
    result:
        FlopsResult to save.
    txt_path:
        Path to save a human-readable text report.
    json_path:
        Path to save machine-readable JSON.

    Returns
    -------
    paths:
        Dict with keys {"txt_path", "json_path"} mapping to saved paths (or None).
    """

    saved = {"txt_path": None, "json_path": None}

    if txt_path is not None:
        save_text(format_flops(result), txt_path)
        saved["txt_path"] = txt_path

    if json_path is not None:
        save_json(asdict(result), json_path)
        saved["json_path"] = json_path

    return saved
