from typing import Dict

import torch
from torch.utils.data import DataLoader

from unet_variants.metrics.binary import BinarySegmentationMetrics

def validate_one_epoch(model: torch.nn.Module,
                       criterion: torch.nn.Module,
                       val_loader: torch.utils.data.DataLoader,
                       device:torch.device,
                       metrics: BinarySegmentationMetrics
                       ) -> Dict[str, float]:
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    val_loss = 0.0
    metrics.reset()
    with torch.no_grad():
        for batch in val_loader:
            images, masks = batch["image"].to(device), batch["mask"].to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            metrics.update(outputs, masks)

    val_loss /= len(val_loader)
    return {"val/loss": val_loss,
            "val/dsc": metrics.compute_f1_score(),
            "val/iou": metrics.compute_iou_score()}