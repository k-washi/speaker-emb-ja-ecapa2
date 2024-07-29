'''
AAMsoftmax loss function copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
@ reference
https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/loss.py
'''
from typing import List
import torch, math
import torch.nn as nn
import torch.nn.functional as F

class MixupAAMsoftmax(nn.Module):
    def __init__(self, n_class, hidden_size=64, m=0.2, s=30):
        """AAMsoftmax loss function

        Args:
            n_class (_type_): class num
            hidden_size (int): hidden size (model output). Defaults to 64.
            m (float, optional): loss margin. Defaults to 0.2.
            s (int, optional): loss scale. Defaults to 30.
        """
        
        super(MixupAAMsoftmax, self).__init__()
        self.m = m
        self.s = s
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, hidden_size), requires_grad=True) # (out_features, in_features)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

    def create_phi(self, cosine, sine, mixup_lambda):
        # params
        cos_m = torch.cos(self.m * mixup_lambda)
        sin_m = torch.sin(self.m * mixup_lambda)
        th = torch.cos(math.pi - self.m * mixup_lambda)
        mm = torch.sin(math.pi - self.m * mixup_lambda) * self.m * mixup_lambda
        
        phi = cosine * cos_m - sine * sin_m # cos(theta + m * lamba) = cos(theta)cos(m*lamda) - sin(theta)sin(m*lambda)
        phi = torch.where((cosine - th) > 0, phi, cosine - mm)
        return phi
    
    def multi_label_forward(self, x, label1, label2, mixup_lambda):
        """
        Forward pass of the Mixup AAMSoftmax model.

        Args:
            x (torch.Tensor): Input tensor. (B, 192)
            label1 (torch.Tensor): First label tensor. (B)
            label2 (torch.Tensor): Second label tensor. (B)
            mixup_lambda (float): Mixup lambda value. (B)

        Returns:
            tuple: A tuple containing the loss tensor and the output tensor.
        """
        #phiは、マージンを適応したcosine (cos(theta + m))
        cosine = F.linear(F.normalize(x), F.normalize(self.weight)) # (B, n_class:7836)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi1 = self.create_phi(cosine, sine, mixup_lambda)
        phi2 = self.create_phi(cosine, sine, 1 - mixup_lambda)
        
        one_hot1 = torch.zeros_like(cosine)
        one_hot1.scatter_(1, label1.view(-1, 1), 1)
        one_hot2 = torch.zeros_like(cosine)
        one_hot2.scatter_(1, label2.view(-1, 1), 1)
        # 正解ラベルの部分のみphiを適用
        other_one_hot = torch.ones_like(cosine) - one_hot1 - one_hot2
        output = (one_hot1 * phi1) + (one_hot2 * phi2) + (other_one_hot * cosine)
        output = output * self.s
        
        # create label
        one_hot_label = mixup_lambda * one_hot1 + (1 - mixup_lambda) * one_hot2
        #
        output = F.log_softmax(output, dim=1)
        loss = self.ce(output, one_hot_label)

        return loss, output

    def one_label_forward(self, x, label1):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi1 = self.create_phi(cosine, sine, 1)
        one_hot1 = torch.zeros_like(cosine)
        one_hot1.scatter_(1, label1.view(-1, 1), 1)
        output = one_hot1 * phi1 + (1 - one_hot1) * cosine
        output = output * self.s
        
        output = F.log_softmax(output, dim=1)
        loss = self.ce(output, label1)
        return loss, output
        
    def forward(self, x, label1, label2=None, mixup_lambda=None):
        """
        Forward pass of the Mixup AAMSoftmax model.

        Args:
            x (torch.Tensor): Input tensor.
            label1 (torch.Tensor): First label tensor.
            label2 (torch.Tensor): Second label tensor.
            mixup_lambda (float): Mixup lambda value.

        Returns:
            tuple: A tuple containing the loss tensor and the output tensor.
        """
        if mixup_lambda is not None:
            mixup_lambda = mixup_lambda.reshape(-1, 1)
        label1 = label1.long()
        if label2 is None:
            return self.one_label_forward(x, label1)
        
        label2 = label2.long()
        return self.multi_label_forward(x, label1, label2, mixup_lambda)       
        