from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import List, Optional

from src.dataset.ecapa2.dataset import Ecapa2Dataset
from src.dataset.ecapa2.collate import collate_fn
from src.utils.ml import seed_worker
from src.utils.logger import get_logger
logger = get_logger(debug=True)

from src.config.config import Config

class Ecapa2DatasetModule(LightningDataModule):
    def __init__(
        self,
        train_audio_fp_list: List[str],
        train_label_list: List[int],
        valid_audio_fp_list: List[str],
        valid_label_list: List[int],
        cfg: Config
    ) -> None:
        super().__init__()
        
        self.cfg = cfg
        self.train_audio_fp_list = train_audio_fp_list
        self.train_label_list = train_label_list
        self.valid_audio_fp_list = valid_audio_fp_list
        self.valid_label_list = valid_label_list
    
    def train_dataloader(self) -> torch.Any:
        train_dataset = Ecapa2Dataset(
            audio_fp_list=self.train_audio_fp_list,
            label_list=self.train_label_list,
            config=self.cfg.dataset,
            num_classes=self.cfg.dataset.audio.num_classes,
            is_augment=True,
            is_mixup=True,
            is_mel=True
        )
        return DataLoader(
            train_dataset,
            batch_size=self.cfg.ml.batch_size,
            shuffle=True,
            num_workers=self.cfg.ml.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker
        )
    
    def val_dataloader(self):
        valid_dataset = Ecapa2Dataset(
            audio_fp_list=self.valid_audio_fp_list,
            label_list=self.valid_label_list,
            config=self.cfg.dataset,
            num_classes=self.cfg.dataset.audio.num_classes,
            is_augment=False,
            is_mixup=False,
            is_mel=True
        )
        return DataLoader(
            valid_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.cfg.ml.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker
        )
    
    def test_dataloader(self):
        test_dataset = Ecapa2Dataset(
            audio_fp_list=self.valid_audio_fp_list,
            label_list=self.valid_label_list,
            config=self.cfg.dataset,
            num_classes=self.cfg.dataset.audio.num_classes,
            is_augment=False,
            is_mixup=False,
            is_mel=True
        )
        return DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.cfg.ml.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker
        )