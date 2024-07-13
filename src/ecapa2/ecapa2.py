import torch
import torch.nn as nn

def get_activation(activation: str):
    if activation == "relu":
        return nn.ReLU(inplace=True)
    else:
        raise NotImplementedError(f"activation {activation} is not implemented")

class Conv2dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: tuple[int, int] = (1, 1),
        activation: str = "relu",
        bias: bool = False
    ) -> None:
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            padding=[kernel_size//2, kernel_size//2]
        )
        
        self.activation = get_activation(activation)
        self.batch_norm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv2d(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        return x

class FrequencyWiseSqueezeExcitationBlock(nn.Module):
    def __init__(
        self,
        frequency_bins_num: int,
        hidden_dim: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1x1_1 = nn.Conv2d(
            in_channels=frequency_bins_num,
            out_channels=hidden_dim,
            kernel_size=1,
            bias=True
        )
        
        self.activation = get_activation(activation)
        self.conv1x1_2 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=frequency_bins_num,
            kernel_size=1,
            bias=True
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (B, C, F, T)
        x = x.permute(0, 2, 1, 3) # (B, F, T, C)
        w = self.pool(x)
        w = self.conv1x1_1(w)
        w = self.activation(w)
        w = self.conv1x1_2(w)
        w = self.sigmoid(w)
        x = x * w
        x = x.permute(0, 2, 1, 3) # (B, C, F, T)
        return x

class LocalFeatureExtracterBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        first_stride: tuple[int, int] = (1, 1),
        frequency_bins_num: int = 256,
        fwse_hidden_dim: int = 128,
        activation: str = "relu",
        use_frequency_encoding: bool = False,
    ) -> None:
        super().__init__()
        self.conv2d_1 = Conv2dBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=first_stride,
            activation=activation
        )
        self.conv2d_2 = Conv2dBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            activation=activation
        )
        self.conv2d_3 = Conv2dBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            activation=activation
        )
        frequency_bins_num = frequency_bins_num // first_stride[0]
        self.downsample = None
        if first_stride[0] > 1:
            self.downsample = nn.MaxPool2d(kernel_size=kernel_size, stride=first_stride, padding=kernel_size//2)
        self.fwse = FrequencyWiseSqueezeExcitationBlock(
            frequency_bins_num=frequency_bins_num,
            hidden_dim=fwse_hidden_dim,
            activation=activation
        )
        
        self.freq_encoding = None
        if use_frequency_encoding:
            self.freq_encoding = nn.Parameter(torch.randn(1, 1, frequency_bins_num, 1))
        
    def forward(self, x):
        w = x
        if self.freq_encoding is not None:
            w = w + self.freq_encoding
        w = self.conv2d_1(w)
        w = self.conv2d_2(w)
        w = self.conv2d_3(w)
        w = self.fwse(w)
        
        if self.downsample is not None:
            x = self.downsample(x)
        x = x + w
        return x
        


if __name__ == "__main__":
    from src.utils.audio import load_wave, spectrogram_torch
    audio, sr = load_wave("src/__example/sample.wav", sample_rate=16000, is_torch=True, mono=True)
    audio = audio.unsqueeze(0)
    print("#(batch, audio length)", audio.shape)
    spec = spectrogram_torch(
        audio,
        n_fft=510, # freq = n_fft//2 + 1
        hop_size=160,
        win_size=400
    )
    print("#(batch, freq, frame)", spec.shape)
    spec = spec.unsqueeze(1) # add channel
    print("#(batch, channel, freq, frame)", spec.shape)
    
    local_feature_extracter01 = LocalFeatureExtracterBlock(
        1, 164, 3, first_stride=(1, 1), frequency_bins_num=256, fwse_hidden_dim=128, activation="relu"
    )
    out = local_feature_extracter01(spec)
    print("lfe01", out.shape)
    
    local_feature_extracter02 = LocalFeatureExtracterBlock(
        164, 164, 3, first_stride=(2, 1), frequency_bins_num=256, fwse_hidden_dim=128, activation="relu"
    )
    out = local_feature_extracter02(out)
    print("lfe02", out.shape)