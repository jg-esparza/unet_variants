from typing import Callable, Dict, Any
import torch
from torch import nn

import torch.optim as optim

from unet_variants.utils.registries import review_registry_availability, validate_instance

class OptimizerFactory:
    """
        Factory to build supported optimizers.
        Add new optimizer function by using register function.
        """
    _BUILDERS: Dict[str, Callable[[nn.Module, Any], torch.optim.Optimizer]] = {}

    @classmethod
    def register(cls, name: str):
        def _wrap(fn: Callable[[nn.Module, Any], torch.optim.Optimizer]):
            cls._BUILDERS[name.lower()] = fn
            return fn
        return _wrap

    @classmethod
    def build(cls, model: nn.Module,cfg: Any) -> torch.optim.Optimizer:
        key = review_registry_availability(name=cfg.name, registry=cls._BUILDERS, kind="optimizer")
        optimizer = cls._BUILDERS[key](model, cfg.params)
        validate_instance(name=key, obj=optimizer, expected_type=torch.optim.Optimizer)
        return optimizer

# ---------- Register Optimizers ----------

@OptimizerFactory.register("adamw")
def _adamw(model: nn.Module, cfg: Any):
    return optim.AdamW(model.parameters(),
                       lr = float(cfg.lr),
                       betas = (float(cfg.beta1), float(cfg.beta2)),
                       eps = float(cfg.eps),
                       weight_decay = float(cfg.weight_decay),
                       amsgrad = cfg.amsgrad)
