import random
from pathlib import Path
import torch
import torchaudio

from src.config.dataset.default import AugCodecConfig

CODEC_FORMAT_LIST = ["wav", "g722", "ogg", "webm", "mp3"]

MP3_CODEC_QSCALE = [None] + list(range(0, 10))

#EFFECT_LIST = [
#    "aecho=0.8:0.88:60:0.4",
#    "aecho=0.8:0.88:6:0.4",
#    "aecho=0.8:0.9:1000:0.3",
#    "aecho=0.8:0.9:1000|1800:0.3|0.25",
#    "acrossfade=d=10:c1=exp:c2=exp",
#    "acrossfade=d=10:o=0:c1=exp:c2=exp",
#    "acrossover=split=1500[LOW][HIGH]",
#    "acrossover=split=1500:order=8th[LOW][HIGH]",
#    "acrossover=split=1500 8000:order=8th[LOW][MID][HIGH]",
#    "adelay=1500|0|500",
#    "adelay=0|500S|700S",
#    "adelay=delays=64S:all=1",
#    "adrc=transfer='if(gt(p,-50),-50+(p-(-50))/6,p)':attack=50:release=100",
#    "adrc=transfer='if(gt(p,-50),-50+(p-(-50))/2,p)':attack=50:release=100:channels=FC",
#    "adrc=transfer='if(lte(p,-85),p-800,p)':attack=1:release=5",
#    "adrc=transfer='if(lt(p,-10),-10+(p-(-10))*2,p)':attack=50:release=100"
#    "adrc=transfer='min(p,-60)':attack=2:release=10",
#    "aeval=val(ch)/2:c=same",
#    "aeval=val(0)|-val(1)",
#    "afade=t=in:ss=0:d=15",
#    "afade=t=out:st=875:d=25",
#    "afftdn=nr=10:nf=-40",
#    "afftdn=nr=10:nf=-80:tn=1",
#    "asendcmd=0.0 afftdn sn start,asendcmd=0.4 afftdn sn stop,afftdn=nr=20:nf=-40",
#]
#

def apply_codec(waveform, sample_rate, format, encoder=None, codec_config=None):
    if codec_config is None:
        encoder = torchaudio.io.AudioEffector(format=format, encoder=encoder)
    else:
        encoder = torchaudio.io.AudioEffector(format=format, encoder=encoder, codec_config=codec_config)
    return encoder.apply(waveform, sample_rate)

class CodecAugment(torch.nn.Module):
    def __init__(
        self, 
        cfg: AugCodecConfig,
        sample_rate: int=16000
    ):
        super().__init__()
        self.cfg = cfg
        self.prob = cfg.prob
        self.sample_rate = sample_rate
    
    def forward(self, audio: torch.Tensor):
        if torch.rand((1)).item() > self.prob:
            return audio
        
        unsqueeze = False
        if audio.dim() == 1:
            unsqueeze = True
            audio = audio.unsqueeze(0)
        audio = audio.T
        # audio codecは入力が(time, channel)
        format = random.choice(CODEC_FORMAT_LIST)
        encoder = None
        if format == "wav":
            encoder = random.choice(["pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_f32le", "pcm_f64le", "pcm_mulaw"])
        elif format in ["ogg", "webm"]:
            encoder = random.choice(["libvorbis", "libopus"])
        codec_config = None
        if format == "mp3":
            qscale = random.choice(MP3_CODEC_QSCALE)
            if qscale is not None:
                codec_config = torchaudio.io.CodecConfig(qscale=qscale)

       
        audio = apply_codec(audio, self.sample_rate, format, encoder, codec_config=codec_config)
      
        audio = audio.T
        if unsqueeze:
            audio = audio.squeeze(0)
        return audio

if __name__ == "__main__":
    from src.config.dataset.default import AugCodecConfig
    from src.utils.audio import save_wave, load_wave
    cfg = AugCodecConfig()
    cfg.init_prob = 1.0
    codec_aug = CodecAugment(cfg)
    audio, _ = load_wave("src/__example/sample.wav", sample_rate=16000, mono=True)
    output_dir = Path("data") / "augment" / "codec"
    output_dir.mkdir(exist_ok=True, parents=True)
    for i in range(10):
        
        augmented_audio = codec_aug(audio)
        output_fp = output_dir / f"sample_{i}.wav"
        save_wave(augmented_audio, str(output_fp))