'''
This is the ECAPA-TDNN model.
This model is modified and combined based on the following three projects:
  1. https://github.com/clovaai/voxceleb_trainer/issues/86
  2. https://github.com/lawlict/ECAPA-TDNN/blob/master/ecapa_tdnn.py
  3. https://github.com/speechbrain/speechbrain/blob/96077e9a1afff89d3f5ff47cab4bca0202770e4f/speechbrain/lobes/models/ECAPA_TDNN.py

References from https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/model.py
'''

import math, torch
import torch.nn as nn


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        
        out = self.se(out)
        out += residual
        return out 



    
class ECAPA_TDNN(nn.Module):

    def __init__(
        self, 
        frequency_bins_num=80,
        channel_size=1024,
        hidden_size=192
    ):
        """
        Args:
            channel_size (int): channel size. Defaults to 1000.
            hidden_size (int): output hidden size. Defaults to 64.
        """

        super(ECAPA_TDNN, self).__init__()
        self.conv1  = nn.Conv1d(frequency_bins_num, channel_size, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(channel_size)
        self.layer1 = Bottle2neck(channel_size, channel_size, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(channel_size, channel_size, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(channel_size, channel_size, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*channel_size, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, hidden_size)
        self.bn6 = nn.BatchNorm1d(hidden_size)


    def forward(self, x) -> torch.Tensor:
        """音声から特徴抽出 (time_indexは、可変でOK)
        Args:
            x (torch.Tensor): メルスペクトロうグラム (batch_size, n_mels, time_index)

        Returns:
            torch.Tensor: 特徴ベクトル (batch_size, hidden_size)
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4))

        x = torch.cat((mu,sg), 1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)
        return x

if __name__ == "__main__":
    from src.utils.audio import load_wave, mel_spectrogram_torch
    audio, sr = load_wave("src/__example/sample.wav", sample_rate=16000, is_torch=True, mono=True)
    audio = audio.unsqueeze(0)
    print("#(batch, audio length)", audio.shape)
    
    mel = mel_spectrogram_torch(
        audio,
        num_mels=80,
        sampling_rate=16000,
        n_fft=512,
        win_size=400,
        hop_size=160,
        fmax=8000,
        fmin=0,
    )
    print("#(batch, n_mels, time_index)", mel.shape)
    model = ECAPA_TDNN()
    model.eval()
    output = model(mel)
    print("#(batch, hidden size)", output.shape)