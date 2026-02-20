from typing import Dict

import torch
from torch.utils.data import DataLoader

from unet_variants.metrics.binary import BinarySegmentationMetrics

def evaluate(model: torch.nn.Module,
             test_loader: torch.utils.data.DataLoader,
             device:torch.device,
             metrics: BinarySegmentationMetrics
             ) -> Dict[str, float]:
    """
    Evaluate a segmentation model on a test loader.
    """
    model.eval()
    metrics.reset()
    with torch.no_grad():
        for batch in test_loader:
            images, masks = batch["image"].to(device), batch["mask"].to(device)
            outputs = model(images)
            metrics.update(outputs, masks)

    return {"seg/acc": metrics.compute_accuracy(),
            "seg/dsc": metrics.compute_f1_score(),
            "seg/iou": metrics.compute_iou_score(),
            "seg/sen": metrics.compute_sensitivity(),
            "seg/spe": metrics.compute_specificity()
            }
