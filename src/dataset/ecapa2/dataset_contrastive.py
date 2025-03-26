import numpy as np
import torch
import gc
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import Dataset
import random
from typing import Tuple

from src.utils.audio import load_wave, save_wave

from src.dataset.augment.augment import AugmentManager
from src.dataset.augment.mixup import (
    get_mixup_lambda,
    mixup
)
from src.config.dataset.default import DatasetConfig
from src.modules.stft import (
    OnnxSTFT,
    spec_max_random_normalization
)
from src.utils.audio import spec_to_mel_torch


class Ecapa2Dataset(Dataset):
    def __init__(
        self, 
        audio_fp_list: List[Path],
        label_list: List[int],
        config: DatasetConfig,
        num_classes: int,
        is_augment: bool=False,
        is_mel: bool=False
    ):
        super().__init__()
        self.audio_fp_list = audio_fp_list
        self.label_list = label_list
        
        self.label_to_index_list = {}
        for index, label in enumerate(label_list):
            if label not in self.label_to_index_list:
                self.label_to_index_list[label] = []
            self.label_to_index_list[label].append(index)
        
        self.cfg = config
        self.num_classes = num_classes
        self.aug = None
        if is_augment:
            self.aug = AugmentManager(config.augment, sample_rate=config.audio.sample_rate)
        
        self.stft = OnnxSTFT(
            filter_length=config.audio.n_fft,
            hop_length=config.audio.hop_length,
            win_length=config.audio.win_length,
        )
        
        self.is_mel = is_mel
        self.audio_config = config.audio
        
    def __len__(self):
        return len(self.audio_fp_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Retrieves the item at the given index.

            Args:
                idx (int): The index of the item to retrieve.

            Returns:
                Tuple[torch.Tensor, torch.Tensor]: A tuple containing the mixed spectrogram, mixed one-hot label,
                mixup lambda, and a tuple of the original audio samples.
            >>  print(mixed_spec.size(), label1.size(), label2.sizse(), mixup_lambda.size())
            >> torch.Size([2, 1, 257, 201]) torch.Size([2, 1]) torch.Size([2, 1]) torch.Size([2, 1]) 
            """
            while True:
                try:
                    audio1, label_id1, spec1 = self.process(idx, same_label=False)
                    assert spec1.abs().sum() > 0, f"spec1 all 0: {spec1}"
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"Error fp: {self.audio_fp_list[idx]}")
                    idx = torch.randint(0, len(self), (1,)).item()
                    continue
        
            idx = random.sample(self.label_to_index_list[label_id1], 1)[0]
            while True:
                try:
                    audio2, label_id2, spec2 = self.process(idx, same_label=True)
                    assert spec2.abs().sum() > 0, f"spec1 all 0: {spec2}"
                    assert label_id1 == label_id2, f"label_id1: {label_id1}, label_id2: {label_id2}"
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"Error fp: {self.audio_fp_list[idx]}")
                    idx = random.sample(self.label_to_index_list[label_id1], 1)[0]
                    continue
            if spec1.dim() == 3:
                spec1 = spec1.squeeze(0)
            if spec2.dim() == 3:
                spec2 = spec2.squeeze(0)
            if self.is_mel:
                spec1 = spec_to_mel_torch(
                    spec1,
                    n_fft=self.audio_config.n_fft,
                    num_mels=self.audio_config.n_mels,
                    sampling_rate=self.audio_config.sample_rate, 
                    fmin=self.audio_config.fmin, 
                    fmax=self.audio_config.fmax
                )
                spec2 = spec_to_mel_torch(
                    spec2,
                    n_fft=self.audio_config.n_fft,
                    num_mels=self.audio_config.n_mels,
                    sampling_rate=self.audio_config.sample_rate, 
                    fmin=self.audio_config.fmin, 
                    fmax=self.audio_config.fmax
                )
            if self.aug is not None:
                spec1 = self.aug.pt_spec_process(spec1)
                spec2 = self.aug.pt_spec_process(spec2)
            #spec1 = spec_max_random_normalization(spec1).unsqueeze(0)
            if spec1.dim() == 2:
                spec1 = spec1.unsqueeze(0)
            if spec2.dim() == 2:
                spec2 = spec2.unsqueeze(0)
            #spec1 = spec1 - torch.mean(spec1, dim=-1, keepdim=True)
            
            return spec1, spec2, label_id1, audio1, audio2
                
            
    
    
    def process(self, index: torch.Tensor, same_label:bool=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process the audio data at the given index.

        Args:
            index (torch.Tensor): The index of the audio data to process.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the processed audio, label ID, and spectrogram.
            audio: (batch_size, audio_length)
            spec: (batch_size, n_freq_bins, time_steps)
        """
        # 音声の読み込み (エラーが発生した場合は再度読み込みを行う)
        while True:
            try:
                audio_fp = self.audio_fp_list[index]
                label_id = self.label_list[index]
                audio, _ = load_wave(audio_fp, sample_rate=self.cfg.audio.sample_rate, is_torch=False, mono=True)
                # audio = audio / np.abs(audio).max()*0.999 # 1以下に正規化
                if self.aug is not None:
                    audio = self.aug.np_wav_process(audio)
                
                break
            except Exception as e:
                print(f"Error: {e}")
                print(f"Error fp: {audio_fp}")
                if same_label:
                    index = random.sample(self.label_to_index_list[label_id], 1)[0]
                else:
                    index = torch.randint(0, len(self), (1,)).item()
                continue
        # 音声ファイルの切り出し
        # 音声ファイルが長過ぎる場合は短くする
        max_length = self.cfg.audio.max_length
        audio_length = audio.shape[0]
        if self.cfg.augment.maxlen.prob > torch.rand((1)).item():
            if audio_length > max_length:
                max_length = int(
                    random.randint(
                        int(self.cfg.augment.maxlen.min_sec * self.cfg.audio.sample_rate),
                        max_length
                    ))
            else:
                if audio_length > int(self.cfg.augment.maxlen.max_sec * self.cfg.audio.sample_rate):
                    max_length = int(
                        random.randint(
                            int(self.cfg.augment.maxlen.min_sec * self.cfg.audio.sample_rate),
                            audio_length
                        ))
                else:
                    max_length = audio_length
        
        if audio_length > max_length:
            start = random.randint(0, audio_length - max_length)
            audio = audio[start:start+max_length]
        # pytorchに変更
        audio = torch.from_numpy(audio)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if self.aug is not None:
            audio = self.aug.pt_wav_process(audio)
            # rirのデータ拡張で音声が長くなっている可能性がある
            if audio.size(-1) > max_length:
                start = random.randint(0, audio.size(-1) - max_length)
                audio = audio[:, start:start+max_length]
        spec, _ = self.stft.transform(audio)

        return audio, label_id, spec

if __name__ == "__main__":
    from src.config.dataset.default import DatasetConfig
    from src.utils.audio import save_wave
    import matplotlib.pyplot as plt
    cfg = DatasetConfig()
    cfg.augment.maxlen.prob = 0.5
    dataset = Ecapa2Dataset(
        [Path("src/__example/sample.wav"), Path("src/__example/sample.wav"), Path("src/__example/sample.wav"), Path("src/__example/sample.wav")],
        [0, 0, 2, 2],
        cfg,
        num_classes=4,
        is_augment=True,
        is_mixup=True,
        is_mel=True
    )
    output_dir = Path("data") / "augment" / "dataset"
    output_dir.mkdir(exist_ok=True, parents=True)
    for j in range(20):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        for i, (spec1, spec2, label, audio1, audio2) in enumerate(dataloader):
            fig, axes = plt.subplots(1, 1, sharex=True, sharey=True)
            
            axes.imshow(spec2.squeeze(0).squeeze(0).numpy(), origin="lower", aspect="auto")
            fig.tight_layout()
            plt.savefig(output_dir / f"sample_{j}_{i}_before.png")
            plt.close()
            
            save_wave(audio1[0, 0], str(output_dir / f"sample_{j}_{i}_audio1.wav"))
            save_wave(audio2[0, 0], str(output_dir / f"sample_{j}_{i}_audio2.wav"))
