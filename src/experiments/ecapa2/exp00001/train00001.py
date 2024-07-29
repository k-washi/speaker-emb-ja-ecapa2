import torch
from pathlib import Path
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from dataclasses import asdict

from src.ecapa2.pl_model_e2next import Ecapa2ModelModule
from src.dataset.ecapa2.pl_dataset import Ecapa2DatasetModule
from src.experiments.utils.pl_callbacks import CheckpointEveryEpoch
from src.experiments.utils.utils import get_audiofp_and_label_list_from_userlist_file

from src.utils.logger import get_logger
logger = get_logger(debug=True)

from src.config.config import Config, get_config
cfg:Config = get_config()

seed_everything(cfg.ml.seed)

##########
# PARAMS #
##########

VERSION = "00001"
EXP_ID = "ecapa2"
WANDB_PROJECT_NAME = "speaker_verfication_ecapa2"
IS_LOGGING = True
FAST_DEV_RUN = False
LOGGING_INTERVAL_STEP = 1000

LOG_SAVE_DIR = f"logs/{EXP_ID}_{VERSION}"
model_save_dir = f"{LOG_SAVE_DIR}/ckpt"

train_userlist_fp = "data/users/train_userlist.txt"
test_userlist_fp = "data/users/test_userlist.txt"

train_audiofp_list, train_label_list = get_audiofp_and_label_list_from_userlist_file(train_userlist_fp)
# train_audiofp_list, train_label_list = train_audiofp_list[:2000],train_label_list[:2000] 
num_classes = len(set(train_label_list))
valid_audiofp_list, valid_label_list = get_audiofp_and_label_list_from_userlist_file(test_userlist_fp)
# valid_audiofp_list, valid_label_list = valid_audiofp_list[:1000],valid_label_list[:1000]

############
############

cfg.ml.num_epochs = 50
cfg.ml.batch_size = 22
cfg.ml.num_workers = 8
cfg.ml.accumulate_grad_batches = 23 # batch_size * accumulate_grad_batches = 506 ~ 512
cfg.ml.grad_clip_val = 100
cfg.ml.check_val_every_n_epoch = 1
cfg.ml.early_stopping.patience = 500
cfg.ml.early_stopping.mode = "min"
cfg.ml.early_stopping.monitor = "val_eer"
cfg.ml.mix_precision = 16 # 16 or 32, bf16



cfg.ml.optimizer.lr = 1e-3 # ft: 1e-5
cfg.ml.optimizer.lr_min = 1e-8
cfg.ml.optimizer.t_initial = 10
cfg.ml.optimizer.warm_up_init = 0 # pretrained modelの場合は0
cfg.ml.optimizer.warm_up_t = 0 # pretrained modelの場合は0
cfg.ml.optimizer.warmup_prefix = True # pretrained modelの場合はFalse

# model
cfg.model.mmas.m = 0.2 # ft: 0.4
cfg.model.ecapa2.lfe_use_frequency_encoding = False # Falseでも性能良いかも
cfg.model.ecapa2.gfe_hidden_channels = 512
cfg.model.ecapa2.gfe_out_channels = 768
cfg.model.ecapa2.local_feature_repeat_list = [2, 2, 2, 2, 2]
cfg.model.ecapa2.activation = "gelu"

# dataset

cfg.dataset.audio.sample_rate = 16000
cfg.dataset.audio.max_length = int(2 * cfg.dataset.audio.sample_rate)
cfg.dataset.audio.num_classes = num_classes

# augment
cfg.dataset.augment.maxlen.prob = 0 # ft: 0.4
cfg.dataset.augment.time_stretch.prob = 0.6 # ft: 0.2
cfg.dataset.augment.noise.prob = 0.8 # ft: 0
cfg.dataset.augment.rir.prob = 0.8 # ft: 0
cfg.dataset.augment.tfmask.prob = 0.8 # ft: 0
cfg.dataset.augment.codec.prob = 0.5 # ft: 0.2


def train():
    logger.info(f"Config: {cfg}")

    ################################
    # データセットとモデルの設定
    ################################
    dataset = Ecapa2DatasetModule(
        train_audio_fp_list=train_audiofp_list,
        train_label_list=train_label_list,
        valid_audio_fp_list=valid_audiofp_list,
        valid_label_list=valid_label_list,
        cfg=cfg
    )
    model = Ecapa2ModelModule(config=cfg)
    ################################
    # コールバックなど訓練に必要な設定
    ################################
    wandb_logger = None
    if IS_LOGGING:
        wandb_logger = WandbLogger(name=f"{EXP_ID}_{VERSION}", project=WANDB_PROJECT_NAME, config=asdict(cfg))
        wandb_logger.log_hyperparams(asdict(cfg))
    
    checkpoint_callback = CheckpointEveryEpoch(
        save_dir=model_save_dir,
        every_n_epochs=cfg.ml.check_val_every_n_epoch
    )
    callback_list = [
        checkpoint_callback, 
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(
            monitor=cfg.ml.early_stopping.monitor,
            patience=cfg.ml.early_stopping.patience,
            mode=cfg.ml.early_stopping.mode
        )
    ]
    
    ################################
    # 訓練
    ################################
    device = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(
            precision=cfg.ml.mix_precision,
            accelerator=device,
            devices=cfg.ml.gpu_devices,
            max_epochs=cfg.ml.num_epochs,
            accumulate_grad_batches=cfg.ml.accumulate_grad_batches,
            gradient_clip_val=cfg.ml.grad_clip_val,
            profiler=cfg.ml.profiler,
            fast_dev_run=FAST_DEV_RUN,
            check_val_every_n_epoch=cfg.ml.check_val_every_n_epoch,
            log_every_n_steps=LOGGING_INTERVAL_STEP,
            logger=wandb_logger,
            callbacks=callback_list,
            num_sanity_val_steps=2
        )
    logger.debug("START TRAIN")
    trainer.fit(model, dataset)

if __name__ == "__main__":
    train()