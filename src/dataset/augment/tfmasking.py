import torch
import torchaudio.transforms as T
from src.config.dataset.default import AugTimeFreqMaskingConfig

class TimeFreqMasking(torch.nn.Module):
    def __init__(
        self, 
        cfg: AugTimeFreqMaskingConfig
    ):
        super().__init__()
        self.cfg = cfg
        self.time_masking = T.TimeMasking(cfg.time_mask_max)
        self.freq_masking = T.FrequencyMasking(cfg.freq_mask_max)
        self.prob = cfg.prob
    
    def forward(self, x: torch.Tensor):
        if torch.rand((1)).item() > self.prob:
            return x
        
        x = self.time_masking(x)
        x = self.freq_masking(x)
        return x

if __name__ == "__main__":
    from pathlib import Path
    from src.config.dataset.default import AugTimeFreqMaskingConfig
    from src.utils.audio import save_wave, load_wave
    from src.modules.stft import OnnxSTFT
    import matplotlib.pyplot as plt
    from torchaudio.functional import amplitude_to_DB
    stft = OnnxSTFT(filter_length=510, hop_length=160, win_length=400)
    time_freq_mask = TimeFreqMasking(AugTimeFreqMaskingConfig())
    audio, _ = load_wave("src/__example/sample.wav", sample_rate=16000, mono=True)
    output_dir = Path("data") / "augment" / "time_freq_mask"
    output_dir.mkdir(exist_ok=True, parents=True)
    print(audio.size())
    spec, phase = stft.transform(audio.unsqueeze(0))
    
    print(spec.shape)
    for i in range(10):
        aug_spec = time_freq_mask(spec)
        print(aug_spec.shape)
        aug_spec = aug_spec[0]
        aug_spec /= aug_spec.max()
        aug_spec = torch.clamp(aug_spec, min=1e-5, max=0.5)
        fig, axes = plt.subplots(1, 1, sharex=True, sharey=True)
        
        axes.imshow(aug_spec.numpy(), origin="lower", aspect="auto")
        fig.tight_layout()
        plt.savefig(output_dir / f"sample_{i}_before.png")