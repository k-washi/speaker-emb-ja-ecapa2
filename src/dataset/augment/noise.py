import random
from pathlib import Path
import torch
import torch.nn.functional as F
import torchaudio.functional as aF

from src.utils.audio import load_wave
from src.config.dataset.default import AugNoiseConfig

class NoiseAugment(torch.nn.Module):
    def __init__(
        self, 
        cfg: AugNoiseConfig,
        sample_rate: int=16000
    ):
        super().__init__()
        self.cfg = cfg
        self.noise_dir = Path(cfg.noise_dir)
        assert self.noise_dir.exists(), f"Directory not found: {self.noise_dir}"
        self.audio_fp_list = list(self.noise_dir.glob("**/*"))
        self.prob = cfg.init_prob
        self.sample_rate = sample_rate
    
    def update_prob(self):
        self.prob = self.cfg.update_prob
            
        
    def forward(self, audio: torch.Tensor):
        if torch.rand((1)).item() > self.prob:
            return audio
        
        unsqueeze = False
        if audio.dim() == 1:
            unsqueeze = True
            audio = audio.unsqueeze(0)
            
        audio_length = audio.size(-1)
        noise_fp_list = random.sample(self.audio_fp_list, random.randint(1, self.cfg.max_noise_num))
        for noise_fp in noise_fp_list:
            noise, _ = load_wave(noise_fp, sample_rate=self.sample_rate)
            noise_length = noise.size(-1)
            if noise_length > audio_length:
                noise = noise[:, :audio_length]
            elif noise_length < audio_length:
                noise = F.pad(noise, (0, audio_length-noise_length), mode="circular")
            
            assert noise.size(-1) == audio_length, f"Size mismatch: {noise.size(-1)} != {audio_length}"
            snr = torch.tensor([random.uniform(self.cfg.min_snr, self.cfg.max_snr)])
            audio = aF.add_noise(audio, noise, snr=snr)
        if unsqueeze:
            audio = audio.squeeze(0)
        return audio

if __name__ == "__main__":
    from src.config.dataset.default import AugNoiseConfig
    from src.utils.audio import save_wave
    noise_aug = NoiseAugment(AugNoiseConfig())
    audio, _ = load_wave("src/__example/sample.wav", sample_rate=16000, mono=True)
    output_dir = Path("data") / "augment" / "noise"
    output_dir.mkdir(exist_ok=True, parents=True)
    for i in range(10):
        augmented_audio = noise_aug(audio)
        output_fp = output_dir / f"sample_{i}.wav"
        save_wave(augmented_audio, str(output_fp))
        print(augmented_audio.size())
    
