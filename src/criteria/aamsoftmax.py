'''
AAMsoftmax loss function copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
@ reference
https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/loss.py
'''
from typing import List
import torch, math
import torch.nn as nn
import torch.nn.functional as F

class AAMsoftmax(nn.Module):
    def __init__(self, n_class, hidden_size=64, m=0.2, s=30):
        """AAMsoftmax loss function

        Args:
            n_class (_type_): class num
            hidden_size (int): hidden size (model output). Defaults to 64.
            m (float, optional): loss margin. Defaults to 0.2.
            s (int, optional): loss scale. Defaults to 30.
        """
        
        super(AAMsoftmax, self).__init__()
        self.m = m
        self.s = s
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, hidden_size), requires_grad=True) # (out_features, in_features)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        # self.kl = torch.nn.KLDivLoss(reduction="sum")

    def forward(self, x, label=None):
        
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        # cos(theta + m) < cos(theta) 
        # mがあることで、対象クラスのcosineが小さくなり、その分、xのクラス内分散を小さく、クラス間分散を大きくするように学習)
        # しかし、theta = piを境に減少から増加に変わるため、
        # cosine(theta) > cosine(pi - m) では影響無しで, phiを使用
        # cosine(theta) < cosine(pi - m) では影響有りで, cosine(theta)に対して直接マージンを作成 (CosFace)によるマージン
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        # マージンの作成
        # >>> a = torch.zeros((2,4))
        # >>> label = torch.tensor([2, 3])
        # >>> label.view(-1, 1)
        # tensor([[2],
        #         [3]])
        # >>> a.scatter_(1, label.view(-1, 1).long(), 1)
        # tensor([[0., 0., 1., 0.],
        #         [0., 0., 0., 1.]])
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        # 正解ラベルの部分のみphiを適用
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        
        output = F.log_softmax(output, dim=1)
        loss = self.ce(output, label)

        return loss, output