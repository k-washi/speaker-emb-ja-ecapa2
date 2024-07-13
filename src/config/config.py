from dataclasses import dataclass, field
from omegaconf import MISSING, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
from typing import List, Any


from src.config.ml.default import MLConfig


defaults = [
    {"ml": "default"},
]

@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    ml: MLConfig = MISSING


def get_config(
    ml: MLConfig = MLConfig(),
):
    cfg = OmegaConf.create({
        "_target_": "__main__.Config",
        "ml": ml,
    })
    cfg = hydra.utils.instantiate(cfg)
    return cfg