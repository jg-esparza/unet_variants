from typing import Callable, Dict, Any

import torch
import torch.optim as optim

from unet_variants.utils.registries import review_registry_availability, validate_instance

class SchedulerFactory:
    """
    Factory to build supported schedulers.
    Add new scheduler by using register function.
    """
    _BUILDERS: Dict[str, Callable[[torch.optim.Optimizer, Any], torch.optim.lr_scheduler]] = {}

    @classmethod
    def register(cls, name: str):
        def _wrap(fn: Callable[[torch.optim.Optimizer, Any], torch.optim.lr_scheduler]):
            cls._BUILDERS[name.lower()] = fn
            return fn
        return _wrap

    @classmethod
    def build(cls, optimizer: torch.optim.Optimizer, cfg: Any) -> torch.optim.lr_scheduler:
        key = review_registry_availability(name=cfg.name, registry=cls._BUILDERS, kind="scheduler")
        scheduler = cls._BUILDERS[key](optimizer, cfg.params)
        return scheduler

# ---------- Register Schedulers ----------

@SchedulerFactory.register("cosine")
def _cosine(optimizer:torch.optim.Optimizer, cfg: Any):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                T_max= int(cfg.t_max),
                                                eta_min= float(cfg.eta_min),
                                                last_epoch= int(cfg.last_epoch))
