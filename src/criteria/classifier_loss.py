from typing import List
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from src.criteria.focal_loss import FocalLoss

class ClassifierLoss(nn.Module):
    def __init__(
        self,
        n_class,
        hidden_size=64,
        focal_loss=False,
        focal_loss_gamma=2,
        label_smoothing=0
    ) -> None:
        super().__init__()
        
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, hidden_size), requires_grad=True) # (out_features, in_features)
        nn.init.xavier_normal_(self.weight, gain=1)
        if focal_loss:
            self.ce = FocalLoss(gamma=focal_loss_gamma)
        else:
            self.ce = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing
            )
        
    def forward(self, x, y):
        output = F.linear(F.normalize(x), F.normalize(self.weight))
        loss = self.ce(output, y)
        return loss, output