import torch
import pytorch_lightning as pl
from pathlib import Path

class CheckpointEveryEpoch(pl.Callback):
    def __init__(self, save_dir, start_epoch=0, every_n_epochs=1):
        self.start_epoch = start_epoch
        self.save_dir = save_dir
        self.epochs = 0
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train epoch """
        self.epochs += 1
        if self.epochs == 1 or self.epochs >= self.start_epoch and self.epochs % self.every_n_epochs == 0:
            save_dir = Path(f"{self.save_dir}") / f"ckpt-{self.epochs}"
            save_dir.mkdir(exist_ok=True, parents=True)
            net_g_save_path = save_dir / "ecapa2.ckpt"
            if hasattr(trainer, "model"):
                if hasattr(trainer.model, "module"):
                    torch.save(trainer.model.module.model.state_dict(), net_g_save_path)
                    
                    if hasattr(trainer.model.module, "input_normalization"):
                        torch.save(trainer.model.module.input_normalization.get_stete_dict(), save_dir / "normalize.ckpt")
                else:
                    torch.save(trainer.model.model.state_dict(), net_g_save_path)
                    if hasattr(trainer.model, "input_normalization"):
                        torch.save(trainer.model.input_normalization.get_state_dict(), save_dir / "normalize.ckpt")
                return None
            