'''
AAMsoftmax loss function copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
@ reference
https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/loss.py
'''
from typing import List
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from src.criteria.focal_loss import FocalLoss

class MixupAAMsoftmax(nn.Module):
    def __init__(
        self, 
        n_class, 
        hidden_size=64, 
        m=0.2, 
        s=30,
        k=1,
        elastic=False,
        elastic_std=0.0125,
        elastic_plus=False,
        focal_loss=False,
        focal_loss_gamma=2
    ):
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
        if s is None or s <= 0:
            self.s = math.sqrt(2) * math.log(n_class - 1)
        self.k = k
        self.n_class = n_class
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class*k, hidden_size), requires_grad=True) # (out_features, in_features)
        nn.init.xavier_normal_(self.weight, gain=1)
        
        if focal_loss:
            self.ce = FocalLoss(gamma=focal_loss_gamma)
        else:
            self.ce = nn.CrossEntropyLoss()
        
        self.elastic = elastic
        self.elastic_std = elastic_std
        self.elastic_plus = elastic_plus

    def create_phi(self, cosine, sine, mixup_lambda, m):
        # params
        cos_m = torch.cos(m * mixup_lambda)
        sin_m = torch.sin(m * mixup_lambda)
        th = torch.cos(math.pi - m * mixup_lambda)
        mm = torch.sin(math.pi - m * mixup_lambda) * m * mixup_lambda
        
        phi = cosine * cos_m - sine * sin_m # cos(theta + m * lamba) = cos(theta)cos(m*lamda) - sin(theta)sin(m*lambda)
        phi = torch.where(cosine > th, phi, cosine - mm)
        return phi
    
    def elastic_margin(self, cosine, m):
        # https://zenn.dev/primenumber/articles/5a6b6da01aaafb
        m = torch.normal(mean=m, std=self.elastic_std, size=cosine.size()).to(cosine.device)
        if self.training and self.elastic_plus:
            with torch.no_grad():
                distmat = cosine.detach().clone()
                _, ori_indices = torch.sort(distmat, dim=0, descending=True)
                m, _ = torch.sort(m, dim=0)
                m = torch.gather(m, 0, ori_indices) # marginをcosineの降順に並べ替え
        return m
    
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
        if self.k > 1:
            cosine = torch.reshape(cosine, (-1, self.n_class, self.k)) # (B, n_class, k)
            cosine, _ = torch.max(cosine, axis=2) # (B, n_class)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        m = self.m
        if self.elastic:
            m = self.elastic_margin(cosine, m)
        phi1 = self.create_phi(cosine, sine, mixup_lambda, m)
        phi2 = self.create_phi(cosine, sine, 1 - mixup_lambda, m)
        one_hot1 = torch.zeros_like(cosine)
        one_hot1 = one_hot1.scatter_(1, label1.view(-1, 1).long(), 1)
        one_hot2 = torch.zeros_like(cosine)
        one_hot2 = one_hot2.scatter_(1, label2.view(-1, 1).long(), 1)
        # 正解ラベルの部分のみphiを適用
        other_one_hot = (torch.ones_like(cosine).long() - one_hot1 - one_hot2).clamp(0, 1)
        output = one_hot1 * phi1 + one_hot2 * phi2 + other_one_hot * cosine
        output = output * self.s
        
        # create label
        one_hot_label = mixup_lambda * one_hot1 + (1 - mixup_lambda) * one_hot2
        loss = self.ce(output, one_hot_label)
        return loss, output

    def one_label_forward(self, x, label1):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        if self.k > 1:
            cosine = torch.reshape(cosine, (-1, self.n_class, self.k)) # (B, n_class, k)
            cosine, _ = torch.max(cosine, axis=2) # (B, n_class)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(1e-4, 1))
        m = self.m
        if self.elastic:
            m = self.elastic_margin(cosine, m)
        phi1 = self.create_phi(cosine, sine, torch.Tensor([1.]).to(cosine.device), m)
        one_hot1 = torch.zeros_like(cosine)
        one_hot1.scatter_(1, label1.view(-1, 1), 1)
        output = one_hot1 * phi1 + (1 - one_hot1) * cosine
        output = output * self.s
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
        