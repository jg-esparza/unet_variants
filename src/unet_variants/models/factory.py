from __future__ import annotations

from typing import Callable, Dict, Any
import torch.nn as nn
from omegaconf import DictConfig

from unet_variants.models.unet import UNet
from unet_variants.models.resnet_unet import ResNetUnet

from unet_variants.utils.registries import review_registry_availability, validate_instance

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
    def build(cls, cfg: DictConfig) -> nn.Module:
        key = review_registry_availability(name=cfg.name, registry=cls._BUILDERS, kind="model")
        model = cls._BUILDERS[key](cfg)
        validate_instance(name=key, obj=model, expected_type=nn.Module)
        return model
