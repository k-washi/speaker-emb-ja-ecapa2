from dataclasses import dataclass, field

@dataclass
class WandbConfig:
    project_name: str = "speakeremb-ecapa2"

@dataclass()
class SaveConfig:
    top_k: int = -1
    monitor: str = 'val_eer'
    mode: str = "min"

@dataclass
class EearlyStoppingConfig:
    patience: int = 5
    mode: str = "max"
    monitor: str = "val_eer"
    

    
@dataclass()
class Optimizer:
    optimizer: str = "adamw"
    scheduler: str = "cosine_with_warmup"
    lr: float = 2e-4
    eps: float = 1e-5
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 1e-4
    #lr_min: float = 1e-7
    mode: str = "min"
    #t_initial: int = 25 # 1 cycle epochs
    t_mul: int = 1
    #decay_rate:float = 0.5
    monitor:str = "val_eer"
    #warm_up_init:float = 1e-7
    warm_up_t:int = 2
    #warmup_prefix:bool =  True

@dataclass()
class MLConfig:
    seed:int =  3407
    num_epochs: int = 50
    batch_size: int = 32
    val_batch_size: int = 12
    num_workers: int = 4
    accumulate_grad_batches: int = 1
    grad_clip_val: float = 100
    
    check_val_every_n_epoch: int = 10
    mix_precision: str = 32 # 16 or 32, bf16
    gpu_devices: int = 1
    profiler: str = "simple"
    checkpoint: SaveConfig = field(default_factory=lambda: SaveConfig())
    early_stopping: EearlyStoppingConfig = field(default_factory=lambda: EearlyStoppingConfig())
    optimizer: Optimizer = field(default_factory=lambda: Optimizer())
    wandb: WandbConfig = field(default_factory=lambda: WandbConfig())