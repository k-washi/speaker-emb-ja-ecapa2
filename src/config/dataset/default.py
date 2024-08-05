from dataclasses import dataclass, field

@dataclass
class AugTimeStretchConfig:
    use: bool = True
    min_rate: float = 0.9
    max_rate: float = 1.1
    prob: float = 0.6
    #update_prob: float = 0.2

@dataclass
class AugVolumeConfig:
    use: bool = True
    volume_mul_params: list=field(default_factory=lambda: [0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.999])
    volume_aug_rate: float=0.8

@dataclass
class AugNoiseConfig:
    use: bool = True
    noise_dir: str = "/data/kwnoise"
    min_snr: float = 0
    max_snr: float = 20.0
    max_noise_num: int = 2
    prob: float = 0.8
    #update_prob: float = 0.0

@dataclass
class AugRirConfig:
    use: bool = True
    rir_dir: str = "/data/kwriris"
    prob: float = 0.8
    #update_prob: float = 0.0
    max_rir_sec: int = 2

@dataclass
class AugTimeFreqMaskingConfig:
    use: bool = True
    freq_mask_max: int = 32
    time_mask_max: int = 5
    prob: float = 0.8
    #update_prob: float = 0.0

@dataclass
class AugCodecConfig:
    use: bool = True
    prob: float = 0.5
    #update_prob: float = 0.2

@dataclass
class AugMixupConfig:
    alpha: float = 0.2
    beta: float = 0.2
    spec_normalize: bool = True

@dataclass
class AugAudioMaxLengthConfig:
    prob: float = 0 #
    min_sec: int = 2
    max_sec: int = 5

@dataclass
class AugmentConfig:
    time_stretch: AugTimeStretchConfig = field(default_factory=lambda: AugTimeStretchConfig())
    volume: AugVolumeConfig = field(default_factory=lambda: AugVolumeConfig())
    noise: AugNoiseConfig = field(default_factory=lambda: AugNoiseConfig())
    rir: AugRirConfig = field(default_factory=lambda: AugRirConfig())
    tfmask: AugTimeFreqMaskingConfig = field(default_factory=lambda: AugTimeFreqMaskingConfig())
    codec: AugCodecConfig = field(default_factory=lambda: AugCodecConfig())
    mixup: AugMixupConfig = field(default_factory=lambda: AugMixupConfig())
    maxlen: AugAudioMaxLengthConfig = field(default_factory=lambda: AugAudioMaxLengthConfig())

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_fft: int = 510
    win_length: int = 400
    hop_length: int = 160
    max_length: int = int(16000 * 2)
    num_classes: int = 7839
    n_mels: int = 80
    fmin: int = 80
    fmax: int = 7600


@dataclass
class DatasetConfig:
    audio: AudioConfig = field(default_factory=lambda: AudioConfig())
    augment: AugmentConfig = field(default_factory=lambda: AugmentConfig())