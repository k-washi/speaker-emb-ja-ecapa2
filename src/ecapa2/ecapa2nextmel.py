import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation(activation: str):
    if activation == "relu":
        return nn.ReLU(inplace=True)
    elif activation == "gelu":
        return nn.GELU()
    else:
        raise NotImplementedError(f"activation {activation} is not implemented")

class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)
    

class DepthwiseConv2dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: tuple[int, int] = (1, 1),
        bias: bool = False
    ) -> None:
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            padding=[kernel_size//2, kernel_size//2],
            groups=in_channels
        )
    
    def forward(self, x):
        x = self.conv2d(x)
        return x

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
        #self.batch_norm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv2d(x)
        x = self.activation(x)
        #x = self.batch_norm(x)
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
    
    def forward(self, x: torch.Tensor):
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
        self.conv2d_1 = DepthwiseConv2dBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=first_stride,
            bias=True
        )
        self.ln = LayerNorm(in_channels, eps=1e-4)
        self.conv2d_2 = Conv2dBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            activation=activation,
            bias=True
        )
        self.conv2d_3 = Conv2dBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            activation=activation,
            bias=True
        )
        self.out_frequency_bins_num = frequency_bins_num // first_stride[0]
        self.downsample = None
        if first_stride[0] > 1:
            self.downsample = nn.Sequential(*[
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=first_stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            ])
        self.fwse = FrequencyWiseSqueezeExcitationBlock(
            frequency_bins_num=self.out_frequency_bins_num,
            hidden_dim=fwse_hidden_dim,
            activation=activation
        )
        
        self.freq_encoding = None
        if use_frequency_encoding:
            self.freq_encoding = nn.Parameter(torch.randn(1, 1, frequency_bins_num, 1) * frequency_bins_num ** -0.5)
          
    def forward(self, x):
        w = x
        if self.freq_encoding is not None:
            w = w + self.freq_encoding
        w = self.conv2d_1(w)
        w = self.ln(w)
        w = self.conv2d_2(w)
        w = self.conv2d_3(w)
        w = self.fwse(w)
        if self.downsample is not None:
            x = self.downsample(x)
        x = x + w
        return x

class LocalFeatureExtractorModule(nn.Module):
    def __init__(
        self,
        frequency_bins_num: int,
        activation: str = "relu",
        fwse_hidden_dim: int = 128,
        use_frequency_encoding: bool = False,
        out_channels_list: list[int] = [164, 164, 192, 192],
        stride_list: list[tuple[int, int]] = [(1, 1), (2, 1), (2, 1), (2, 1)],
        repeat_num_list: list[int] = [3, 4, 4, 5]
    ):
        super().__init__()
        
        block_list:list = []
        in_channels = 1
        out_frequency_bins_num = frequency_bins_num
        for out_channels, stride, repeat_num in zip(out_channels_list, stride_list, repeat_num_list):
            lfe_blocks, out_frequency_bins_num = self.make_blocks(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                first_stride=stride,
                frequency_bins_num=out_frequency_bins_num,
                fwse_hidden_dim=fwse_hidden_dim,
                activation=activation,
                use_frequency_encoding=use_frequency_encoding,
                repeat_num=repeat_num,
            )
            in_channels = out_channels
            block_list.extend(lfe_blocks)
        self.blocks = nn.ModuleList(block_list)
        self.last_out_channels = out_channels
        self.last_frequency_bins_num = out_frequency_bins_num
        
    def make_blocks(
        self, 
        in_channels: int,
        out_channels: int,
        kernel_size: int=7,
        first_stride: tuple[int, int] = (1, 1),
        frequency_bins_num: int = 256,
        fwse_hidden_dim: int = 128,
        activation: str = "relu",
        use_frequency_encoding: bool = False,
        repeat_num: int = 4,
    ):
        blocks: list[LocalFeatureExtracterBlock] = []
        for _ in range(repeat_num):
            blocks.append(
                LocalFeatureExtracterBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    first_stride=first_stride,
                    frequency_bins_num=frequency_bins_num,
                    fwse_hidden_dim=fwse_hidden_dim,
                    activation=activation,
                    use_frequency_encoding=use_frequency_encoding
                )
            )
            first_stride = (1, 1) # 最初だけstrideを変える
            in_channels = out_channels
            frequency_bins_num = blocks[-1].out_frequency_bins_num
        return blocks, frequency_bins_num
    
    def forward(self, x):
        for i, block in enumerate(self.blocks):
            x = block(x)
        return x

class Conv1dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        activation: str = "relu",
        bias: bool = False
    ) -> None:
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            padding=kernel_size//2
        )
        
        self.activation = get_activation(activation)
        self.batch_norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        return x

class Res2ConvNetModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: str = "relu",
        stride: int = 1,
        split_num: int = 4
    ):
        super().__init__()
        
        #self.pre_conv = Conv1dBlock(
        #    in_channels=in_channels,
        #    out_channels=out_channels,
        #    kernel_size=1,
        #    stride=stride,
        #    activation=activation,
        #    bias=False
        #)
        assert in_channels == out_channels, f"in_channels must be equal to out_channels, but got {in_channels} != {out_channels}"
        assert out_channels % split_num == 0, f"split_num must be divisible by in_channels, but got {out_channels} % {split_num}"
        self.width = out_channels // split_num
        self.split_num = split_num
        
        self.conv_list = nn.ModuleList([nn.Sequential(
            nn.Conv1d(
                in_channels=self.width,
                out_channels=self.width,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size//2,
                bias=False
            ),
            nn.BatchNorm1d(
                self.width
            )) for _ in range(split_num - 1)])
        self.activation = get_activation(activation)
        self.bn = nn.BatchNorm1d(out_channels)
        
        
    def forward(self, x):
        spx = torch.split(x, self.width, dim=1)
        for i in range(self.split_num - 1):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.conv_list[i](sp)
            sp = self.activation(sp)
            if i == 0:
                spo = sp
            else:
                spo = torch.cat([spo, sp], dim=1)
        spo = torch.cat([spo, spx[-1]], dim=1)

        x = self.activation(x)
        x =self.bn(x)
        return x

class Conv1dSEBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        activation: str = "relu",
    ):
        super().__init__()
        self.pre_conv = Conv1dBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            activation=activation,
        )
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.conv1x1_1 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=True
        )
        self.activation = get_activation(activation)
        self.conv1x1_2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=True
        )
        
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.pre_conv(x)
        w = self.pool(x)
        w = self.conv1x1_1(w)
        w = self.activation(w)
        w = self.conv1x1_2(w)
        w = self.sigmoid(w)
        x = x * w
        return x
        


class GlobalFeatureExtractorModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        frequency_bins_num: int = 16,
        activation: str = "relu",
        res2convnet_split_num: int = 4
    ):
        super().__init__()
        in_channels = in_channels * frequency_bins_num
        print(f"#(in_channels, frequency_bins_num)", in_channels)
        self.conv1d_seq = nn.Sequential(*[
            Conv1dBlock(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                stride=1,
                activation=activation,
                bias=False
            ),
            Conv1dBlock(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                stride=1,
                activation=activation,
                bias=False
            ),
        ])
        
        self.res2net_block = Res2ConvNetModule(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            activation=activation,
            stride=1,
            split_num=res2convnet_split_num
        )
        
        self.se_conv1d = Conv1dSEBlock(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
            activation=activation
        )
        self.post_conv = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=True
        )
        self.activation = get_activation(activation)

        
    def forward(self, x: torch.Tensor):
        # (batch, channel, freq, frame) -> (batch, freq*channel, freame)
        x = torch.flatten(x.permute(0, 2, 1, 3), start_dim=1, end_dim=2)
        x = self.conv1d_seq(x)
        x = self.res2net_block(x)
        x = self.se_conv1d(x)
        x = self.post_conv(x)
        x = self.activation(x)
        return x

class ChannelDependentStaticsPooling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        activation: str = "relu"
    ) -> None:
        super().__init__()
        
        self.attention = nn.Sequential(*[
            Conv1dBlock(
                in_channels=in_channels*3,
                out_channels=hidden_channels,
                kernel_size=1,
                activation=activation
            ),
            nn.Tanh(),
            nn.Conv1d(hidden_channels, in_channels, kernel_size=1),
            nn.Softmax(dim=2) # 時間方向に対してsoftmax
            
        ])
    
    def forward(self, x):
        t = x.size()[-1]
        globalx = torch.cat((
            x,
            torch.mean(x, dim=2, keepdim=True).expand(-1, -1, t), # 時間方向に圧縮した平均をt個拡張
            torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).expand(-1, -1, t), # 時間方向に圧縮した分散をt個拡張
        ), dim=1) # (B, C*3, T)
        w = self.attention(globalx)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4))
        return mu, sg

class ECAPA2(nn.Module):
    def __init__(
        self,
        frequency_bins_num: int = 256,
        speaker_emb_dim: int = 192,
        activation: str = "relu",
        lfe_fwse_hidden_dim: int = 128,
        lfe_use_frequency_encoding: bool = False,
        gfe_hidden_channels: int = 1024,
        gfe_out_channels: int = 1536,
        state_pool_hidden_channels:int = 256,
        local_feature_repeat_list: list[int] = [3, 4, 4, 4, 5]
    ):
        super().__init__()
        self.lfe = LocalFeatureExtractorModule(
            frequency_bins_num=frequency_bins_num,
            activation=activation,
            fwse_hidden_dim=lfe_fwse_hidden_dim,
            use_frequency_encoding=lfe_use_frequency_encoding,
            out_channels_list=[164, 164, 192],
            stride_list=[(1, 1), (1, 1), (2, 1)],
            repeat_num_list=local_feature_repeat_list # [3, 4, 4, 4, 5]
        )
        self.gfe = GlobalFeatureExtractorModule(
            in_channels=self.lfe.last_out_channels,
            hidden_channels=gfe_hidden_channels,
            out_channels=gfe_out_channels,
            frequency_bins_num=self.lfe.last_frequency_bins_num,
            activation=activation,
            res2convnet_split_num=4
        )
        self.cd_stats_pooling = ChannelDependentStaticsPooling(
            in_channels=gfe_out_channels,
            hidden_channels=state_pool_hidden_channels,
            activation=activation
        )
        
        self.speaker_emb = nn.Linear(gfe_out_channels*2, speaker_emb_dim)
        
    def forward(self, x):
        x = self.lfe(x)
        x = self.gfe(x)
        mu, sg = self.cd_stats_pooling(x)
        x = torch.cat((mu, sg), dim=1)
        x = self.speaker_emb(x)
        return x

if __name__ == "__main__":
    from src.utils.audio import load_wave, spectrogram_torch, mel_spectrogram_torch
    audio, sr = load_wave("src/__example/sample.wav", sample_rate=16000, is_torch=True, mono=True)
    audio = audio.unsqueeze(0)
    print("#(batch, audio length)", audio.shape)
    spec = mel_spectrogram_torch(
        audio,
        n_fft=512, # freq = n_fft//2 + 1
        hop_size=160,
        win_size=400,
        sampling_rate=16000,
        fmin=80,
        fmax=7600,
        num_mels=80
    )
    print("#(batch, freq, frame)", spec.shape)
    spec = spec.unsqueeze(1) # add channel
    print("#(batch, channel, freq, frame)", spec.shape)
    model = ECAPA2(
        frequency_bins_num=80,
        speaker_emb_dim=192,
        activation="gelu",
        lfe_fwse_hidden_dim=128,
        lfe_use_frequency_encoding=True,
        gfe_hidden_channels=512,
        gfe_out_channels=756,
        state_pool_hidden_channels=256,
        local_feature_repeat_list=[2, 2, 2]
    )
    model.eval()
    out = model(spec)
    print("#ecapa2 (batch, speaker_emb_dim)", out.shape)
    print(out)
    print(spec.min(), spec.max())