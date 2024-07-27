import random
from pathlib import Path
import torch
import torch.nn.functional as F
import torchaudio.functional as aF

from src.utils.audio import load_wave
from src.config.dataset.default import AugRirConfig

class RIRAugment(torch.nn.Module):
    def __init__(
        self, 
        cfg: AugRirConfig,
        sample_rate: int=16000
    ):
        super().__init__()
        self.cfg = cfg
        self.rir_dir = Path(cfg.rir_dir)
        assert self.rir_dir.exists(), f"Directory not found: {self.rir_dir}"
        self.audio_fp_list = list(self.rir_dir.glob("**/*"))
        self.prob = cfg.init_prob
        self.sample_rate = sample_rate
        self.max_length = int(sample_rate * cfg.max_rir_sec)
    
    def update_prob(self):
        self.prob = self.cfg.update_prob
    
    def forward(self, audio: torch.Tensor):
        if torch.rand((1)).item() > self.prob:
            return audio
        
        unsqueeze = False
        if audio.dim() == 1:
            unsqueeze = True
            audio = audio.unsqueeze(0)
            
        rir_fp = random.choice(self.audio_fp_list)
        rir, _ = load_wave(rir_fp, sample_rate=self.sample_rate)
        rir = rir[:, :self.max_length]
        rir = rir / torch.linalg.vector_norm(rir, ord=2)
        audio = aF.fftconvolve(audio, rir)
        if unsqueeze:
            audio = audio.squeeze(0)
        return audio

if __name__ == "__main__":
    from src.config.dataset.default import AugRirConfig
    from src.utils.audio import save_wave
    rir_aug = RIRAugment(AugRirConfig())
    audio, _ = load_wave("src/__example/sample.wav", sample_rate=16000, mono=True)
    output_dir = Path("data") / "augment" / "rir"
    output_dir.mkdir(exist_ok=True, parents=True)
    for i in range(10):
        augmented_audio = rir_aug(audio)
        output_fp = output_dir / f"sample_{i}.wav"
        save_wave(augmented_audio, str(output_fp))
        print(augmented_audio.size())
    