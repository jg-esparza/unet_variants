from __future__ import annotations

from typing import Callable, Dict, Any
import torch.nn as nn

from unet_variants.models.unet import UNet
from unet_variants.models.resnet_unet import ResNetUnet

class ModelFactory:
    """
    Factory to build supported models.
    Add new models by adding one line to _BUILDERS.
    """
    _BUILDERS: Dict[str, Callable[[Any], nn.Module]] = {
        "unet": lambda cfg: UNet(cfg),
        "resnet_unet": lambda cfg: ResNetUnet(cfg)
    }

    @classmethod
    def create(cls, name: str, cfg: Any) -> nn.Module:
        key = name.lower().strip()
        if key not in cls._BUILDERS:
            available = ", ".join(sorted(cls._BUILDERS.keys()))
            raise ValueError(f"Unknown model '{name}'. Available: [{available}]")

        model = cls._BUILDERS[key](cfg)
        if not isinstance(model, nn.Module):
            raise TypeError(f"Builder for '{name}' did not return a torch.nn.Module")
        return model
