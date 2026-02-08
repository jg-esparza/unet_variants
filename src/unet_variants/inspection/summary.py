
from __future__ import annotations

from typing import Tuple, Optional
import os
import torch
from torch import nn

from unet_variants.utils.io import save_text

def get_torchinfo_summary(
    model: nn.Module,
    input_size: Tuple[int, int, int, int],
    device: torch.device,
    verbose: int = 0,
) -> str:
    """
    Generate a torchinfo summary as a string (useful for MLflow artifacts).

    Parameters
    ----------
    model:
        Model to summarize.
    input_size:
        Input tensor size as (B, C, H, W).
    device:
        Device on which to run the forward pass for summary.
    verbose:
        torchinfo verbosity level.

    Returns
    -------
    summary_text:
        Human-readable summary text.

    Raises
    ------
    ImportError:
        If torchinfo is not installed.
    """
    try:
        from torchinfo import summary as torchinfo_summary
    except ImportError as e:
        raise ImportError("Install torchinfo: pip install torchinfo") from e

    summary_obj = torchinfo_summary(
        model,
        input_size=input_size,
        device=str(device),
        verbose=verbose,
        col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"),
    )
    return str(summary_obj)


def save_torchinfo_summary(
    model: nn.Module,
    input_size: Tuple[int, int, int, int],
    device: torch.device,
    path: str,
    verbose: int = 1,
) -> str:
    """
    Generate and save torchinfo summary to a text file.

    Parameters
    ----------
    model, input_size, device, verbose:
        Same as get_torchinfo_summary().
    path:
        Where to save the summary text (e.g., "runs/.../summary.txt").

    Returns
    -------
    path:
        The saved file path.
    """
    text = get_torchinfo_summary(model, input_size, device, verbose=verbose)
    save_text(text, path)
    return path
