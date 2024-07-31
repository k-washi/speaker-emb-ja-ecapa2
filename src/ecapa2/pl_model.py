import traceback
from typing import Any, Optional
import torch
import torch.nn.functional as F
import numpy as np
import gc
import itertools
import shutil
from pathlib import Path
import pickle
from pytorch_lightning import LightningModule
from timm.scheduler import CosineLRScheduler
import math

from src.ecapa2.ecapa2 import ECAPA2
from src.config.config import Config
from src.criteria.mixup_aamsoftmax import MixupAAMsoftmax
from src.metrics.metrics import accuracy, ComputeErrorRates, ComputeMinDcf
from src.metrics.utils import tuneThresholdfromScore

from src.utils.logger import get_logger
logger = get_logger(debug=True)

class Ecapa2ModelModule(LightningModule):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__()
        
        self.config = config
        mc = config.model
        self.model = ECAPA2(
            frequency_bins_num=mc.ecapa2.frequency_bins_num,
            speaker_emb_dim=mc.ecapa2.speaker_emb_dim,
            activation=mc.ecapa2.activation,
            lfe_fwse_hidden_dim=mc.ecapa2.lfe_fwse_hidden_dim,
            lfe_use_frequency_encoding=mc.ecapa2.lfe_use_frequency_encoding, # Frequency Encodingによる差はあるか？
            gfe_hidden_channels=mc.ecapa2.gfe_hidden_channels,
            gfe_out_channels=mc.ecapa2.gfe_out_channels,
            state_pool_hidden_channels=mc.ecapa2.state_pool_hidden_channels,
            local_feature_repeat_list=mc.ecapa2.local_feature_repeat_list
        )
        
        self.aam_loss = MixupAAMsoftmax(
            n_class=config.dataset.audio.num_classes,
            hidden_size=mc.ecapa2.speaker_emb_dim,
            m=mc.mmas.m,
            s=mc.mmas.s
        )
        
        self._val_spkemb_output_dir = Path(mc.exp.val_spkemb_output_dir)
    
    def training_step(self, batch, batch_idx):
        x, label1, label2, mixup_lambda = batch
        output = self.model(x)
        
        loss, _ = self.aam_loss(output, label1, label2, mixup_lambda)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Performs a validation step on a batch of data.
        1. calc speaker embedding

        Args:
            batch: The input batch of data.
            batch_idx: The index of the current batch.

        Returns:
            None
        """
        x, label, _, _ = batch
        spk_emb = self.model(x)
        spk_emb = F.normalize(spk_emb).detach().cpu()
        for i in range(len(x)):
            label_idx = int(label[i].item())
            if label_idx not in self.label_set:
                self.label_set.add(label_idx)
                self.embedding_fp_dict[label_idx] = []
            output_fp = self._val_spkemb_output_dir / f"{self._spkemb_index:08d}.pkl"
            assert not output_fp.exists(), f"Output file already exists: {output_fp}"
            self._spkemb_index += 1
            torch.save(spk_emb[i], str(output_fp))
            self.embedding_fp_dict[label_idx].append(output_fp)
       
    
    def on_validation_epoch_start(self):
        self.embedding_fp_dict = {}
        self.label_set = set()
        
        # validationのspkembを格納する場所を作成
        try:
            shutil.rmtree(str(self._val_spkemb_output_dir))
        except:
            pass
        self._val_spkemb_output_dir.mkdir(parents=True, exist_ok=True)
        self._spkemb_index = 0
    
    def on_validation_epoch_end(self):
        min_audio_num_by_spk = 10**8
        score_list, label_list = [], []
        # 同じ話者のspeaker embeddingをまとめる
        speaker_id_list = sorted(list(self.embedding_fp_dict.keys()))
        for label_idx in speaker_id_list:
            embedding_fp_list = self.embedding_fp_dict[label_idx]
            for i, (fp1, fp2) in enumerate(zip(embedding_fp_list, embedding_fp_list[::-1])):
                emb1 = torch.load(fp1)
                emb2 = torch.load(fp2)
                emb1, emb2 = emb1.unsqueeze(0), emb2.unsqueeze(0)
                score = torch.mean(torch.matmul(emb1, emb2.mT))                    
                score_list.append(score.item())
                label_list.append(1)
    
        # 異なる話者のspeaker embeddingを比較
        for speaker_id1, speaker_id2 in zip(speaker_id_list, speaker_id_list[1:]):
            spkemb_list1 = self.embedding_fp_dict[speaker_id1]
            spkemb_list2 = self.embedding_fp_dict[speaker_id2]
            for i, (fp1, fp2) in enumerate(itertools.product(spkemb_list1, spkemb_list2)):
                emb1 = torch.load(fp1)
                emb2 = torch.load(fp2)
                emb1, emb2 = emb1.unsqueeze(0), emb2.unsqueeze(0)
                score = torch.mean(torch.matmul(emb1, emb2.T))
                score_list.append(score.item())
                label_list.append(0)
        score_list = np.array(score_list)
        label_list = np.array(label_list)
        try:
            eer = tuneThresholdfromScore(score_list, label_list, [1, 0.1])[1]
        except Exception as e:
            logger.error(f"Error in tuneThresholdfromScore: {e}")
            eer = 1.0
        try:
            fnrs, fprs, thresholds = ComputeErrorRates(score_list, label_list)
            minDCF, _  = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        except Exception as e:
            logger.error(f"Error in ComputeErrorRates: {e}")
            minDCF = 1.0
        
        self.log('val/eer', eer, on_step=False, on_epoch=True, logger=True)
        self.log('val/minDCF', minDCF, on_step=False,on_epoch=True, logger=True)
        self.log('val_eer', eer, on_step=False, on_epoch=True, logger=True)

        del score_list, label_list
        del self.embedding_fp_dict
        gc.collect()
    
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]

        # https://tma15.github.io/blog/2021/09/17/deep-learningbert%E5%AD%A6%E7%BF%92%E6%99%82%E3%81%ABbias%E3%82%84layer-normalization%E3%82%92weight-decay%E3%81%97%E3%81%AA%E3%81%84%E7%90%86%E7%94%B1/#weight-decay%E3%81%AE%E5%AF%BE%E8%B1%A1%E5%A4%96%E3%81%A8%E3%81%AA%E3%82%8B%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.ml.optimizer.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.config.ml.optimizer.lr,
                eps=self.config.ml.optimizer.eps,
                betas=self.config.ml.optimizer.betas,
            )

        self.scheduler  = CosineLRScheduler(
            self.optimizer,
            t_initial=self.config.ml.optimizer.t_initial,
            lr_min=self.config.ml.optimizer.lr_min,
            cycle_decay=self.config.ml.optimizer.decay_rate,
            cycle_limit=math.ceil(self.config.ml.num_epochs / self.config.ml.optimizer.t_initial),
            warmup_t=self.config.ml.optimizer.warm_up_t,
            warmup_lr_init=self.config.ml.optimizer.warm_up_init,
            warmup_prefix=self.config.ml.optimizer.warmup_prefix,
        )
        return [self.optimizer], [self.scheduler]
    
    def lr_scheduler_step(self, scheduler, metric: Any | None) -> None:
        return scheduler.step(self.current_epoch)