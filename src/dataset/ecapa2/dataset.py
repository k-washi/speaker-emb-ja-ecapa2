import numpy as np
import torch
import gc
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import Dataset
import random
from typing import Tuple

from src.utils.audio import load_wave

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



class Ecapa2Dataset(Dataset):
    def __init__(
        self, 
        audio_fp_list: List[Path],
        label_list: List[int],
        config: DatasetConfig,
        num_classes: int,
        is_augment: bool=False,
        is_mixup: bool=False
    ):
        super().__init__()
        self.audio_fp_list = audio_fp_list
        self.label_list = label_list
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
        self.is_mixup = is_mixup
        
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
                    audio1, label_id1, spec1 = self.process(idx)
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"Error fp: {self.audio_fp_list[idx]}")
                    idx = torch.randint(0, len(self), (1,)).item()
                    continue
            if not self.is_mixup:
                if spec1.dim() == 3:
                    spec1 = spec1.squeeze(0)
                # spec1 = (spec1 / torch.linalg.norm(spec1, ord=2)).unsqueeze(0)
                spec1 = spec_max_random_normalization(spec1).unsqueeze(0)
                return spec1, label_id1, label_id1, torch.tensor([1.0]), (audio1, audio1)
                
            # Mixup
            mixup_lambda = get_mixup_lambda(alpha=self.cfg.augment.mixup.alpha, beta=self.cfg.augment.mixup.beta)
            
            while True:
                try:
                    random_index = torch.randint(0, len(self), (1,)).item()
                    audio2, label_id2, spec2 = self.process(random_index)
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"Error fp: {self.audio_fp_list[random_index]}")
                    continue
            mixed_spec, _ = mixup(
                spec1, label_id1, spec2, label_id2, mixup_lambda, self.num_classes,
                spec_normalize=self.cfg.augment.mixup.spec_normalize
            )
            if mixed_spec.dim() == 2:
                mixed_spec = mixed_spec.unsqueeze(0) # (1, freq_bins, time_steps)
            mixed_spec = spec_max_random_normalization(mixed_spec)
            return mixed_spec, label_id1, label_id2, mixup_lambda, (audio1, audio2)
    
    
    def process(self, index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
                index = torch.randint(0, len(self), (1,)).item()
                continue
        # 音声ファイルの切り出し
        # 音声ファイルが長過ぎる場合は短くする
        max_length = self.cfg.audio.max_length
        if self.cfg.augment.maxlen.prob > torch.rand((1)).item():
            max_length = int(random.randint(self.cfg.augment.maxlen.min_sec, self.cfg.augment.maxlen.max_sec) * self.cfg.audio.sample_rate)
        audio_length = audio.shape[0]
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
        if self.aug is not None:
            spec = self.aug.pt_spec_process(spec)
        return audio, label_id, spec

if __name__ == "__main__":
    from src.config.dataset.default import DatasetConfig
    from src.utils.audio import save_wave
    import matplotlib.pyplot as plt
    dataset = Ecapa2Dataset(
        [Path("src/__example/sample.wav"), Path("src/__example/sample.wav"), Path("src/__example/sample.wav"), Path("src/__example/sample.wav")],
        [0, 1, 2, 3],
        DatasetConfig(),
        num_classes=4,
        is_augment=True,
        is_mixup=True
    )
    output_dir = Path("data") / "augment" / "dataset"
    output_dir.mkdir(exist_ok=True, parents=True)
    for j in range(20):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        for i, (mixed_spec, label1, label2, mixup_lambda, (audio1, audio2)) in enumerate(dataloader):
            print(mixed_spec.size(), label1.size(), label2.size(), mixup_lambda.size())
            print(i, mixup_lambda, label1, label2)
            fig, axes = plt.subplots(1, 1, sharex=True, sharey=True)
            
            axes.imshow(mixed_spec.squeeze(0).squeeze(0).numpy(), origin="lower", aspect="auto")
            fig.tight_layout()
            plt.savefig(output_dir / f"sample_{j}_{i}_before.png")
            
            save_wave(audio1[0, 0], str(output_dir / f"sample_{j}_{i}_audio1.wav"))
            save_wave(audio2[0, 0], str(output_dir / f"sample_{j}_{i}_audio2.wav"))
    
    #print("Validation----------------")
    #dataset = Ecapa2Dataset(
    #    [Path("src/__example/sample.wav"), Path("src/__example/sample.wav"), Path("src/__example/sample.wav"), Path("src/__example/sample.wav")],
    #    [0, 1, 2, 3],
    #    DatasetConfig(),
    #    num_classes=4,
    #    is_augment=False, 
    #    is_mixup=False
    #)
    #output_dir = Path("data") / "augment" / "dataset"
    #output_dir.mkdir(exist_ok=True, parents=True)
    #for _ in range(10):
    #    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    #    for i, (mixed_spec, label1, label2, mixup_lambda, (audio1, audio2)) in enumerate(dataloader):
    #        print(mixed_spec.size(), label1.size(), label2.size(), mixup_lambda.size())
    #        print(i, mixup_lambda, label1, label2)
    #        #fig, axes = plt.subplots(1, 1, sharex=True, sharey=True)
    #        #
    #        #axes.imshow(mixed_spec.squeeze(0).squeeze(0).numpy(), origin="lower", aspect="auto")
    #        #fig.tight_layout()
    #        #plt.savefig(output_dir / f"sample_{i}_before.png")
    #        #
    #        #save_wave(audio1[0, 0], str(output_dir / f"sample_{i}_audio1.wav"))
    #        #save_wave(audio2[0, 0], str(output_dir / f"sample_{i}_audio2.wav"))