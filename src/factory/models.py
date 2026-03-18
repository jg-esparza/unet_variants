from __future__ import annotations

from typing import Callable, Dict, Any
import torch.nn as nn
from omegaconf import DictConfig

from models.cnn.unet import UNet
from models.cnn.attention_unet import AttU_Net
from models.cnn.malunet import MALUNet
from models.transformers.transunet import TransUNet
from models.transformers.swinunet import SwinUnet

from factory.registries import review_registry_availability, validate_instance

class ModelFactory:
    """
    Factory to build supported models.
    Add new models by adding one line to _BUILDERS.
    """
    _BUILDERS: Dict[str, Callable[[Any], nn.Module]] = {
        "unet": lambda cfg: UNet(cfg),
        "att_unet": lambda cfg: AttU_Net(cfg),
        "malunet": lambda cfg: MALUNet(cfg),
        "transunet": lambda cfg: TransUNet(cfg),
        "swinunet": lambda cfg: SwinUnet(cfg),
    }

    @classmethod
    def build(cls, cfg: DictConfig) -> nn.Module:
        key = review_registry_availability(name=cfg.architecture_name, registry=cls._BUILDERS, kind="model")
        model = cls._BUILDERS[key](cfg)
        validate_instance(name=key, obj=model, expected_type=nn.Module)
        return model
