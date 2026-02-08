
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def preview_predictions(
    model: nn.Module,
    batch: dict,
    device: torch.device,
    save_path: Optional[str] = None,
) -> None:
    """
    (Optional) Visualize model predictions for a batch.

    Parameters
    ----------
    model:
        Segmentation model.
    batch:
        Batch dict from dataloader (e.g., {"image": ..., "mask": ...}).
    device:
        Device for inference.
    save_path:
        If provided, saves visualization image to this path.

    Notes
    -----
    Implement later using matplotlib / PIL:
    - input image
    - ground truth mask overlay
    - predicted mask overlay
    """
    raise NotImplementedError("Visualization is not implemented yet.")
