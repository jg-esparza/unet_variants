
from __future__ import annotations

from typing import Tuple, Union
import torch
from torch import nn

try:
    from torchinfo import summary as torchinfo_summary
except ImportError as e:
    torchinfo_summary = None


def print_torchinfo_summary(
    model: nn.Module,
    input_size: Tuple[int, int, int, int],
    device: Union[str, torch.device] = "cpu",
    verbose: int = 1,
) -> None:
    """
    Print torchinfo summary.
    input_size: (B, C, H, W)
    """
    if torchinfo_summary is None:
        raise ImportError(
            "torchinfo is not installed. Install with: pip install torchinfo"
        )

    print("\n=== torchinfo summary ===")
    torchinfo_summary(
        model,
        input_size=input_size,
        device=str(device),
        verbose=verbose,
        col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"),
    )
