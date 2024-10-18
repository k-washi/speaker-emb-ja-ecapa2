import math
from typing import Mapping
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any

class InputNormalization(torch.nn.Module):
    """Performs mean and variance normalization of the input tensor.

    Arguments
    ---------
    mean_norm : True
         If True, the mean will be normalized.
    std_norm : True
         If True, the standard deviation will be normalized.
    norm_type : str
         It defines how the statistics are computed ('sentence' computes them
         at sentence level, 'batch' at batch level, 'speaker' at speaker
         level, while global computes a single normalization vector for all
         the sentences in the dataset). Speaker and global statistics are
         computed with a moving average approach.
    avg_factor : float
         It can be used to manually set the weighting factor between
         current statistics and accumulated ones.
    requires_grad : bool
        Whether this module should be updated using the gradient during training.
    update_until_epoch : int
        The epoch after which updates to the norm stats should stop.

    Example
    -------
    >>> import torch
    >>> norm = InputNormalization()
    >>> inputs = torch.randn([10, 101, 20])
    >>> inp_len = torch.ones([10])
    >>> features = norm(inputs, inp_len)
    """

    from typing import Dict

    def __init__(
        self,
        mean_norm=True,
        std_norm=True,
        norm_type="global",
        avg_factor=None,
        requires_grad=False,
        update_until_epoch=3,
    ):
        super().__init__()
        self.mean_norm = mean_norm
        self.std_norm = std_norm
        self.norm_type = norm_type
        self.avg_factor = avg_factor
        self.requires_grad = requires_grad
        self.glob_mean = torch.tensor([0.], dtype=torch.float32)
        self.glob_std = torch.tensor([0.], dtype=torch.float32)
        self.weight = 1.0
        self.count = 0
        self.eps = 1e-10
        self.update_until_epoch = update_until_epoch

    def forward(self, x, lengths=None, epoch=0):
        """Returns the tensor with the surrounding context.

        Arguments
        ---------
        x : torch.Tensor
            A batch of tensors.
            (B, 1, Mel_bins, T)
        lengths : torch.Tensor
            The length of each sequence in the batch.
        spk_ids : torch.Tensor containing the ids of each speaker (e.g, [0 10 6]).
            It is used to perform per-speaker normalization when
            norm_type='speaker'.
        epoch : int
            The epoch count.

        Returns
        -------
        x : torch.Tensor
            The normalized tensor.
        """
        N_batches = x.shape[0]

        current_means = []
        current_stds = []
        
        if lengths is None:
            lengths = torch.ones(N_batches) * x.shape[-1]
        lengths = lengths.long()

        x = x.transpose(-1, -2)
        for snt_id in range(N_batches):
            # Avoiding padded time steps
            actual_size = lengths[snt_id]

            # computing statistics
            current_mean, current_std = self._compute_current_stats(
                x[snt_id, 0, :actual_size, :]
            )
            current_means.append(current_mean)
            current_stds.append(current_std)


            current_mean = torch.mean(torch.stack(current_means), dim=0)
            current_std = torch.mean(torch.stack(current_stds), dim=0)

        
        if self.norm_type == "batch":
            out = (x - current_mean.data) / (current_std.data)

        if self.norm_type == "global":
            if self.training:
                if self.count == 0:
                    self.glob_mean = current_mean
                    self.glob_std = current_std

                elif epoch is None or epoch < self.update_until_epoch:
                    if self.avg_factor is None:
                        self.weight = 1 / (self.count + 1)
                    else:
                        self.weight = self.avg_factor

                    self.glob_mean = (1 - self.weight) * self.glob_mean.to(
                        current_mean
                    ) + self.weight * current_mean

                    self.glob_std = (1 - self.weight) * self.glob_std.to(
                        current_std
                    ) + self.weight * current_std

                self.glob_mean.detach()
                self.glob_std.detach()

                self.count = self.count + 1

            out = (x - self.glob_mean.data.to(x)) / (
                self.glob_std.data.to(x)
            )
        out = out.transpose(-1, -2)
        return out

    def _compute_current_stats(self, x):
        """Computes mean and std

        Arguments
        ---------
        x : torch.Tensor
            A batch of tensors.

        Returns
        -------
        current_mean : torch.Tensor
            The average of x along dimension 0
        current_std : torch.Tensor
            The standard deviation of x along dimension 0
        """
        # Compute current mean
        if self.mean_norm:
            current_mean = torch.mean(x, dim=0).detach().data
        else:
            current_mean = torch.tensor([0.0], device=x.device)

        # Compute current std
        if self.std_norm:
            current_std = torch.std(x, dim=0).detach().data
        else:
            current_std = torch.tensor([1.0], device=x.device)

        # Improving numerical stability of std
        current_std = torch.max(
            current_std, self.eps * torch.ones_like(current_std)
        )

        return current_mean, current_std
    
    def get_state_dict(self):
        """Returns the state of the module as a dictionary."""
        return {
            "glob_mean": self.glob_mean,
            "glob_std": self.glob_std,
            "count": self.count,
        }
    def set_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        """Loads the module state."""
        self.glob_mean = state_dict["glob_mean"]
        self.glob_std = state_dict["glob_std"]
        self.count = state_dict["count"]
        return self

def get_activation(activation: str):
    if activation == "relu":
        return nn.ReLU(inplace=True)
    elif activation == "gelu":
        return nn.GELU()
    else:
        raise NotImplementedError(f"activation {activation} is not implemented")

def length_to_mask(length, max_len=None, dtype=None, device=None):
    """Creates a binary mask for each sequence.

    Reference: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397/3

    Arguments
    ---------
    length : torch.LongTensor
        Containing the length of each sequence in the batch. Must be 1D.
    max_len : int
        Max length for the mask, also the size of the second dimension.
    dtype : torch.dtype, default: None
        The dtype of the generated mask.
    device: torch.device, default: None
        The device to put the mask variable.

    Returns
    -------
    mask : tensor
        The binary mask.

    Example
    -------
    >>> length=torch.Tensor([1,2,3])
    >>> mask=length_to_mask(length)
    >>> mask
    tensor([[1., 0., 0.],
            [1., 1., 0.],
            [1., 1., 1.]])
    """
    assert len(length.shape) == 1

    if max_len is None:
        max_len = length.max().long().item()  # using arange to generate mask
    mask = torch.arange(
        max_len, device=length.device, dtype=length.dtype
    ).expand(len(length), max_len) < length.unsqueeze(1)

    if dtype is None:
        dtype = length.dtype

    if device is None:
        device = length.device

    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask

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
        #self.pool = nn.AdaptiveAvgPool2d(output_size=1)
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
    
    def forward(self, x: torch.Tensor, m=None):
        # x: (B, C, F, T)
        x = x.permute(0, 2, 1, 3) # (B, F, T, C)
        if m is None:
            w = x.mean((2, 3), keepdim=True)
        else:
            m = m.permute(0, 2, 1, 3)
            total = m.sum(dim=(2, 3), keepdim=True)
            w = (x * m).sum(dim=(2, 3), keepdim=True) / total
        
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
          
    def forward(self, x, m=None):
        w = x
        if self.freq_encoding is not None:
            w = w + self.freq_encoding
        if m is not None:
            w = w * m # mask
        w = self.conv2d_1(w)
        w = self.ln(w)
        w = self.conv2d_2(w)
        w = self.conv2d_3(w)
        w = self.fwse(w, m=m)
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
    
    def forward(self, x, m=None):
        for i, block in enumerate(self.blocks):
            x = block(x, m=m)
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
    def forward(self, x, m=None):
        x = self.pre_conv(x)
        if m is None:
            w = x.mean(dim=2, keepdim=True)
        else:
            total = m.sum(dim=2, keepdim=True).float()
            w = (x * m).sum(dim=2, keepdim=True) / total
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

        
    def forward(self, x: torch.Tensor, m=None):
        # (batch, channel, freq, frame) -> (batch, freq*channel, freame)
        x = torch.flatten(x.permute(0, 2, 1, 3), start_dim=1, end_dim=2)
        if m is not None:
            assert m.dim() == 3, f"mask must be 3D tensor, but got {m.dim()}"
        x = self.conv1d_seq(x)
        x = self.res2net_block(x)
        x = self.se_conv1d(x, m=m)
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
            nn.Conv1d(hidden_channels, in_channels, kernel_size=1)
            
        ])
    
    def forward(self, x, m=None):
        t = x.size()[-1]
        if m is None:
            u = torch.mean(x, dim=2, keepdim=True)
            v = torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4))
        else:
            total = m.sum(dim=2, keepdim=True).float()
            u = (x * m).sum(dim=2, keepdim=True) / total
            v = torch.sqrt(((m * (x - u).pow(2)).sum(dim=2, keepdim=True) / total).clamp(min=1e-8))
        globalx = torch.cat((
            x,
            u.expand(-1, -1, t), # 時間方向に圧縮した平均をt個拡張
            v.expand(-1, -1, t), # 時間方向に圧縮した分散をt個拡張
        ), dim=1) # (B, C*3, T)
        w = self.attention(globalx)
        w = w.masked_fill(m == 0, float("-inf"))
        w = F.softmax(w, dim=2)
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
        local_feature_repeat_list: list[int] = [3, 4, 4, 4, 5],
        dropout_rate: float = 0.0
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
        
        self.cd_bn = nn.BatchNorm1d(gfe_out_channels*2)
        self.cd_do = nn.Dropout(p=dropout_rate)
        self.speaker_emb = nn.Linear(gfe_out_channels*2, speaker_emb_dim)
        self.out_bn = nn.BatchNorm1d(speaker_emb_dim)
        
    def forward(self, x, lengths=None):
        
        B, _, _, L = x.shape
        if lengths is None:
            lengths = torch.ones(B, device=x.device) * L
        m = length_to_mask(lengths, max_len=L, device=x.device)
        m3d = m.unsqueeze(1)
        m4d = m3d.unsqueeze(1)
        # x: (batch, 1, freq, frame)
        x = self.lfe(x, m=m4d) # (batch, 192, freq/2, frame)
        x = self.gfe(x, m=m3d) # (batch, 756, frame)
        mu, sg = self.cd_stats_pooling(x, m=m3d) # (batch, 756), (batch, 756)
        x = torch.cat((mu, sg), dim=1) # (batch, 756*2)
        x = self.cd_bn(x)
        x = self.cd_do(x)
        x = self.speaker_emb(x) # (batch, speaker_emb_dim)
        x = self.out_bn(x)
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
        local_feature_repeat_list=[2, 3, 3, 4]
    )
    model.eval()
    out = model(spec)
    print("#ecapa2 (batch, speaker_emb_dim)", out.shape)
    #print(out)
    print(spec.min(), spec.max())