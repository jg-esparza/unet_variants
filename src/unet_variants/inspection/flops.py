
from __future__ import annotations

from typing import Optional, Tuple
import torch
from torch import nn


def estimate_flops(
    model: nn.Module,
    input_size: Tuple[int, int, int, int],
    device: torch.device,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Estimate FLOPs and MACs.
    Returns (flops, macs) or (None, None) if not available.

    Notes:
    - depends on input size
    - may undercount if some ops are unsupported by the FLOPs tool
    """
    b, c, h, w = input_size
    x = torch.randn(b, c, h, w, device=device)

    # Try THOP
    try:
        from thop import profile
    except ImportError:
        return None, None

    model.eval()
    with torch.no_grad():
        try:
            macs, params = profile(model, inputs=(x,), verbose=False)
            flops = 2.0 * macs  # common approximation: FLOPs ≈ 2 * MACs
            return float(flops), float(macs)
        except Exception:
            return None, None
