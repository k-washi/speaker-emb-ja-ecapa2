import torch
import numpy as np
from audiomentations import Compose, TimeStretch


from src.dataset.augment.volume import VolumeAugment
from src.dataset.augment.noise import NoiseAugment
from src.dataset.augment.rir import RIRAugment
from src.dataset.augment.tfmasking import TimeFreqMasking
from src.dataset.augment.codec import CodecAugment
from src.config.dataset.default import AugmentConfig



class AugmentManager():
    def __init__(self, config: AugmentConfig, sample_rate:int=16000) -> None:
        super().__init__()
        self.config = config
        self.sample_rate = sample_rate
        
        # Initialize augmentations
        # Time stretch
        self.time_stretch = None
        if config.time_stretch.use:
            self.time_stretch = Compose([
                TimeStretch(
                    min_rate=config.time_stretch.min_rate,
                    max_rate=config.time_stretch.max_rate,
                    p=config.time_stretch.prob
                )
            ])
        
        # Volume
        self.volume = None
        if config.volume.use:
            self.volume = VolumeAugment(
                volume_mul_params=config.volume.volume_mul_params,
                volume_aug_rate=config.volume.volume_aug_rate
            )
        
        # Noise
        self.noise = None
        if config.noise.use:
            self.noise = NoiseAugment(config.noise, sample_rate=sample_rate)

        # RIR
        self.rir = None
        if config.rir.use:
            self.rir = RIRAugment(config.rir, sample_rate=sample_rate)
            
        # Time-Frequency masking
        self.tfmask = None
        if config.tfmask.use:
            self.tfmask = TimeFreqMasking(config.tfmask)
            
        # Codec
        self.codec = None
        if config.codec.use:
            self.codec = CodecAugment(config.codec, sample_rate=sample_rate)
            
    def np_wav_process(self, x):
        assert isinstance(x, np.ndarray), f"Input should be np.ndarray, but got {type(x)}"
        # Time stretch
        if self.time_stretch is not None:
            x = self.time_stretch(samples=x, sample_rate=self.sample_rate)
        return x
    
    def pt_wav_process(self, x):
        assert isinstance(x, torch.Tensor), f"Input should be torch.Tensor, but got {type(x)}"
        aug_type = torch.randint(0, 4, (1,)).item()
        if aug_type == 0:
            if self.noise is not None:
                x = self.noise(x)
        elif aug_type == 1:
            if self.rir is not None:
                x = self.rir(x)
        elif aug_type == 2:
            if self.noise is not None:
                x = self.noise(x)
            if self.rir is not None:
                x = self.rir(x)
        elif aug_type == 3:
            if self.rir is not None:
                x = self.rir(x)
            if self.noise is not None:
                x = self.noise(x)
        if self.volume is not None:
            x = self.volume(x)
        if self.codec is not None:
            x = self.codec(x)
        # stereo to mono and channel size (1, length)
        if x.dim() == 2 and x.size(0) == 2:
            x = x[:1, :]
        return x
    
    def pt_spec_process(self, x):
        assert isinstance(x, torch.Tensor), f"Input should be torch.Tensor, but got {type(x)}"
        if self.tfmask is not None:
            x = self.tfmask(x)
        return x