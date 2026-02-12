from __future__ import annotations

from typing import Callable, Dict, Any
import torch.nn as nn

from unet_variants.utils.registries import review_registry_availability, validate_instance

class LossFactory:
    """
    Factory to build supported loss functions.
    Add new loss function by using register function.
    """
    _BUILDERS: Dict[str, Callable[[Any], nn.Module]] = {}

    @classmethod
    def register(cls, name: str):
        def _wrap(fn: Callable[[Any], nn.Module]):
            cls._BUILDERS[name.lower()] = fn
            return fn
        return _wrap

    @classmethod
    def build(cls, cfg: Any) -> nn.Module:
        key = review_registry_availability(name=cfg.name, registry=cls._BUILDERS, kind="loss")
        model = cls._BUILDERS[key](cfg.control)
        validate_instance(name=key, obj=model, expected_type=nn.Module)
        return model

# ---------- Register Loss Functions ----------

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        return self.bce_loss(pred_, target_)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = ((2 * intersection.sum(1) + self.smooth)/
                      (pred_.sum(1) + target_.sum(1) + self.smooth))
        dice_loss = 1 - dice_score.sum()/size
        return dice_loss

@LossFactory.register("bce_dice")
class BceDiceLoss(nn.Module):
    def __init__(self, cfg):
        super(BceDiceLoss, self).__init__()
        self.wb = float(cfg.wb)
        self.wd = float(cfg.wd)
        self.bce = BCELoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        loss = self.wd * dice_loss + self.wb * bce_loss
        return loss