# Speech brain like ecapa_tdnn model with classification head.
import traceback
from typing import Any, Optional
import torch
import torch.nn.functional as F
import numpy as np
import gc
import itertools
import shutil
from pathlib import Path
from tqdm import tqdm
from pytorch_lightning import LightningModule
from schedulefree import RAdamScheduleFree

import math
from speechbrain.utils.metric_stats import EER, minDCF

from src.ecapa_tdnn.ecapatdnn import ECAPA_TDNN, InputNormalization
from src.config.config import Config
from src.criteria.dinoloss import DINOLoss

from src.metrics.utils import tuneThresholdfromScore
from src.experiments.utils.plot_emb import umap_show
from src.experiments.utils.utils import cosine_scheduler

from src.utils.logger import get_logger
logger = get_logger(debug=True)

# ecapa2の一部をConvNextに変更

class EcapaTDNNModelModule(LightningModule):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__()
        
        self.config = config
        mc = config.model
        self.model = ECAPA_TDNN(
            mc.ecapa_tdnn.frequency_bins_num,
            lin_neurons=mc.ecapa_tdnn.hidden_size,
            channels=[1024, 1024, 1024, 1024, 3072]
        )
        
        self.teacher_model = ECAPA_TDNN(
            mc.ecapa_tdnn.frequency_bins_num,
            lin_neurons=mc.ecapa_tdnn.hidden_size,
            channels=[1024, 1024, 1024, 1024, 3072]
        )
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        assert config.ml.num_epochs > 0, "DINO training epochs must be greater than 0"
        assert config.ml.num_epochs >= mc.dino.warmup_teacher_temp_epochs, "DINO warmup_teacher_temp_epochs must be less than n_epochs"
        self.loss = DINOLoss(
            hidden_size=mc.ecapa_tdnn.hidden_size,
            warmup_teacher_temp=mc.dino.warmup_teacher_temp,
            teacher_temp=mc.dino.teacher_temp,
            warmup_teacher_temp_epochs=mc.dino.warmup_teacher_temp_epochs,
            n_epochs=config.ml.num_epochs,
            student_temp=mc.dino.student_temp,
            center_momentum=mc.dino.center_momentum
        )
        
        assert config.ml.len_dataloader > 0, "len_dataloader must be greater than 0"
        self.momentum_schedule = cosine_scheduler(
            mc.exp.momentum_teacher, 1, config.ml.num_epochs, config.ml.len_dataloader
        )
        
        self._val_spkemb_output_dir = Path(mc.exp.val_spkemb_output_dir)
        self._current_step = 0
        self._train_batch_idx = 0
    def training_step(self, batch, batch_idx):
        self.model.train()
        optimizer = self.optimizers()
        optimizer.train()
        
        self._train_batch_idx = batch_idx
        x1, x2, lengths1, lengths2, _ = batch
        if x1.dim() == 4:
            x1 = x1.squeeze(1)
        if x2.dim() == 4:
            x2 = x2.squeeze(1)
            
        x_1 = torch.cat([x1, x2], dim=0)
        x_2 = torch.cat([x2, x1], dim=0)
        lengths_1 = torch.cat([lengths1, lengths2], dim=0)
        lengths_2 = torch.cat([lengths2, lengths1], dim=0)
        student_out = self.model(x_1, lengths=lengths_1) # (B, time, mel) -> (B, hidden_size)
        if student_out.dim() == 3:
            student_out = student_out.squeeze(1)
        teacher_out = self.teacher_model(x_2, lengths=lengths_2)
        if teacher_out.dim() == 3:
            teacher_out = teacher_out.squeeze(1)

        loss = self.loss(student_out, teacher_out, self.current_epoch)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def on_after_backward(self):
        super().on_after_backward()
        
        # update teacher model
        m = self.momentum_schedule[self.config.ml.len_dataloader * self.current_epoch + self._train_batch_idx]
        with torch.no_grad():
            for param_s, param_t in zip(self.model.parameters(), self.teacher_model.parameters()):
                param_t.data.mul_(m).add_((1 - m) * param_s.detach().data)
    
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
        self.model.eval()
        optimizer = self.optimizers()
        optimizer.eval()
        x, _, length1, _, label1 = batch
        if x.dim() == 4:
            x = x.squeeze(1)
        with torch.no_grad():
            spk_emb = self.model(x)
            if spk_emb.dim() == 3:
                spk_emb = spk_emb.squeeze(1)
        spk_emb = F.normalize(spk_emb, p=2, dim=1).detach().cpu()
        for i in range(len(x)):
            label_idx = int(label1[i].item())
            if label_idx not in self.label_set:
                self.label_set.add(label_idx)
                self.embedding_fp_dict[label_idx] = []
            output_fp = self._val_spkemb_output_dir / f"{self._spkemb_index:08d}.pkl"
            assert not output_fp.exists(), f"Output file already exists: {output_fp}"
            self._spkemb_index += 1
            torch.save(spk_emb[i], str(output_fp))
            self.embedding_fp_dict[label_idx].append(output_fp)
       
    
    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
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
        super().on_validation_epoch_end()
        score_list, label_list = [], []
        # 同じ話者のspeaker embeddingをまとめる
        embedding_error_num = 0
        score_error_num = 0
        all_data_num = 0
        speaker_id_list = sorted(list(self.embedding_fp_dict.keys()))
        
        cos_sim = torch.nn.CosineSimilarity()
        for label_idx in tqdm(speaker_id_list, desc="Calc same speaker score", total=len(speaker_id_list)):
            embedding_fp_list = self.embedding_fp_dict[label_idx]
            for i, (fp1, fp2) in enumerate(zip(embedding_fp_list[0::2], embedding_fp_list[1::2])):
                emb1 = torch.load(fp1)
                emb2 = torch.load(fp2)
                if emb1.dim() == 1:
                    emb1 = emb1.unsqueeze(0)
                if emb2.dim() == 1:
                    emb2 = emb2.unsqueeze(0)
                all_data_num += 1
                if not (emb1.abs().max() >= 0 and emb2.abs().max() >= 0):
                    embedding_error_num += 1
                score = torch.mean(cos_sim(emb1, emb2))
                if not score.abs().max() >= 0:
                    score_error_num += 1
                    score = torch.tensor([1.0])
                score_list.append(score.item())
                label_list.append(1)
        
        self.log('val/same_score_mean', np.mean(score_list), on_step=False, on_epoch=True, logger=True)
        
        if embedding_error_num > 0:
            print(f"Embedding error: {embedding_error_num}/{all_data_num}")
        if score_error_num > 0:
            print(f"Score error: {score_error_num}/{all_data_num}")
        # 異なる話者のspeaker embeddingを比較
        
        diff_score_list, diff_label_list = [], []
        for speaker_id1, speaker_id2 in tqdm(zip(speaker_id_list[0::2], speaker_id_list[1::2]),
                                             desc="Calc diff speaker score",
                                             total=len(speaker_id_list)//2
                                            ):
            spkemb_list1 = self.embedding_fp_dict[speaker_id1]
            spkemb_list2 = self.embedding_fp_dict[speaker_id2]
            for i, (fp1, fp2) in enumerate(zip(spkemb_list1, spkemb_list2)):
                emb1 = torch.load(fp1)
                emb2 = torch.load(fp2)
                if emb1.dim() == 1:
                    emb1 = emb1.unsqueeze(0)
                if emb2.dim() == 1:
                    emb2 = emb2.unsqueeze(0)
                score = torch.mean(cos_sim(emb1, emb2))
                if not score.abs().max() >= 0:
                    score = torch.tensor([1.0])
                
                diff_score_list.append(score.item())
                diff_label_list.append(0)
        self.log('val/diff_score_mean', np.mean(diff_score_list), on_step=False, on_epoch=True, logger=True)
        eer, _ = EER(torch.FloatTensor(score_list), torch.FloatTensor(diff_score_list))
        if np.isnan(eer):
            eer = 1.
        eer *= 100
        min_dcf, _ = minDCF(torch.FloatTensor(score_list), torch.FloatTensor(diff_score_list))
        if np.isnan(min_dcf):
            min_dcf = 1.
        
        self.log('val/eer', eer, on_step=False, on_epoch=True, logger=True)
        self.log('val/minDCF', min_dcf, on_step=False,on_epoch=True, logger=True)
        self.log('val_eer', eer, on_step=False, on_epoch=True, logger=True)

        del score_list, label_list, diff_score_list, diff_label_list
        del self.embedding_fp_dict
        gc.collect()
    
    def on_test_epoch_start(self):
        self.on_validation_epoch_start()
    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)
    def on_test_epoch_end(self):
        super().on_test_epoch_end()
        speaker_id_list = sorted(list(self.embedding_fp_dict.keys()))
        
        # show embedding
        embedding_list, label_list = [], []
        for label_idx in speaker_id_list:
            embedding_fp_list = self.embedding_fp_dict[label_idx]
            for fp in embedding_fp_list:
                emb = torch.load(fp)
                embedding_list.append(emb)
                label_list.append(label_idx)
        show_output_fp = Path(self.config.ml.show_output_dir) / "umap.png"
        show_output_fp.parent.mkdir(parents=True, exist_ok=True)
        umap_show(embedding_list, label_list, show_output_fp)
        
        
        # calc score
        score_list, label_list = [], []
        # 同じ話者のspeaker embeddingをまとめる
        embedding_error_num = 0
        score_error_num = 0
        all_data_num = 0
        cos_sim = torch.nn.CosineSimilarity()
        for label_idx in tqdm(speaker_id_list, desc="Calc same speaker score", total=len(speaker_id_list)):
            embedding_fp_list = self.embedding_fp_dict[label_idx]
            for i, (fp1, fp2) in enumerate(zip(embedding_fp_list[0::2], embedding_fp_list[1::2])):
                emb1 = torch.load(fp1)
                emb2 = torch.load(fp2)
                if emb1.dim() == 1:
                    emb1 = emb1.unsqueeze(0)
                if emb2.dim() == 1:
                    emb2 = emb2.unsqueeze(0)
                
                all_data_num += 1
                if not (emb1.abs().max() >= 0 and emb2.abs().max() >= 0):
                    embedding_error_num += 1
                score = torch.mean(cos_sim(emb1, emb2))
                
                if not score.abs().max() >= 0:
                    score_error_num += 1
                    score = torch.tensor([1.0])
                score_list.append(score.item())
                label_list.append(1)
        if embedding_error_num > 0:
            print(f"Embedding error: {embedding_error_num}/{all_data_num}")
        if score_error_num > 0:
            print(f"Score error: {score_error_num}/{all_data_num}")
        self.log('test/score_mean', np.mean(score_list), on_step=False, on_epoch=True, logger=True)
        # 異なる話者のspeaker embeddingを比較
        diff_score_list, diff_label_list = [], []
        for speaker_id1, speaker_id2 in tqdm(zip(speaker_id_list[0::2], speaker_id_list[1::2]),
                                             desc="Calc diff speaker score",
                                             total=len(speaker_id_list)//2
                                            ):
            spkemb_list1 = self.embedding_fp_dict[speaker_id1]
            spkemb_list2 = self.embedding_fp_dict[speaker_id2]
            for i, (fp1, fp2) in enumerate(zip(spkemb_list1, spkemb_list2)):
                emb1 = torch.load(fp1)
                emb2 = torch.load(fp2)
                if emb1.dim() == 1:
                    emb1 = emb1.unsqueeze(0)
                if emb2.dim() == 1:
                    emb2 = emb2.unsqueeze(0)
                score = torch.mean(cos_sim(emb1, emb2))
                if not score.abs().max() >= 0:
                    score = torch.tensor([1.0])
                
                diff_score_list.append(score.item())
                diff_label_list.append(0)
        self.log('test/diff_score_mean', np.mean(diff_score_list), on_step=False, on_epoch=True, logger=True)
        eer, _ = EER(torch.FloatTensor(score_list), torch.FloatTensor(diff_score_list))
        if np.isnan(eer):
            eer = 1.
        eer *= 100
        min_dcf, _ = minDCF(torch.FloatTensor(score_list), torch.FloatTensor(diff_score_list))
        if np.isnan(min_dcf):
            min_dcf = 1.
        
        self.log('test/eer', eer, on_step=False, on_epoch=True, logger=True)
        self.log('test/minDCF', min_dcf, on_step=False,on_epoch=True, logger=True)

        del score_list, label_list, diff_score_list, diff_label_list
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
        optimizer = RAdamScheduleFree(
            optimizer_grouped_parameters,
            lr=self.config.ml.optimizer.lr,
            betas=self.config.ml.optimizer.betas
        )
        return [optimizer], []