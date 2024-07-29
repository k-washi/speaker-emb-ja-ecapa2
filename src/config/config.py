from dataclasses import dataclass, field
from omegaconf import MISSING, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
from typing import List, Any


from src.config.ml.default import MLConfig
from src.config.dataset.default import DatasetConfig
from src.config.model.default import ModelConfig

defaults = [
    {"ml": "default"},
    {"dataset": "default"},
    {"model", "default"}
]

@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    ml: MLConfig = MISSING
    dataset: DatasetConfig = MISSING
    model: ModelConfig = MISSING


def get_config(
    ml: MLConfig = MLConfig(),
    dataset: DatasetConfig = DatasetConfig(),
    model: ModelConfig = ModelConfig()
):
    cfg = OmegaConf.create({
        "_target_": "__main__.Config",
        "ml": ml,
        "dataset": dataset,
        "model": model
    })
    cfg = hydra.utils.instantiate(cfg)
    return cfg