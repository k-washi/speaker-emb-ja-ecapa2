from dataclasses import dataclass, field

@dataclass
class Ecapa2Config:
    frequency_bins_num: int = 256
    speaker_emb_dim: int = 192
    activation: str = "relu"
    lfe_fwse_hidden_dim: int = 128
    lfe_use_frequency_encoding: bool = False
    gfe_hidden_channels: int = 1024
    gfe_out_channels: int = 1536
    state_pool_hidden_channels:int = 256
    local_feature_repeat_list: list[int] = field(default_factory=lambda: [3, 4, 4, 4, 5])

@dataclass
class EcapaTDNNConfig:
    frequency_bins_num: int = 80
    channel_size: int = 1024
    hidden_size: int = 192

@dataclass
class AAMSoftmaxConfig:
    m: float = 0.2
    s: float = 30.0
    k: int = 1
    elastic: bool = False
    elastic_std: float = 0.0125
    elastic_plus: bool = False
    focal_loss: bool = False
    focal_loss_gamma: int = 2
    label_smoothing: float = 0
@dataclass
class ExpConfig:
    val_spkemb_output_dir: str = "/data/val_spkemb"

@dataclass
class ModelConfig:
    ecapa2: Ecapa2Config = field(default_factory=lambda: Ecapa2Config())
    ecapa_tdnn: EcapaTDNNConfig = field(default_factory=lambda: EcapaTDNNConfig())
    mmas: AAMSoftmaxConfig = field(default_factory=lambda: AAMSoftmaxConfig())
    exp: ExpConfig = field(default_factory=lambda: ExpConfig())