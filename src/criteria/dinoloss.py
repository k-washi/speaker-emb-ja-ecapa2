from typing import List
import torch, math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DINOHead(nn.Module):
    def __init__(
        self,
        feature_dim:int,
        hidden_dim:int,
        projection_dim:int
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),  # First FC layer
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # Second FC layer
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim, bias=False),  # Weight normalized layer
            nn.LayerNorm(projection_dim)  # ℓ2 Normalization
        )
    
    def forward(self, x):
        return self.head(x)

class DINOLoss(nn.Module):
    def __init__(
        self,
        hidden_size=64,
        warmup_teacher_temp=0.04,
        teacher_temp=0.04,
        warmup_teacher_temp_epochs=30,
        n_epochs=30,
        student_temp=0.1,
        center_momentum=0.9,
    ) -> None:
        """_summary_

        Args:
            hidden_size (int, optional): _description_. Defaults to 64.
            warmup_teacher_temp (float, optional): 
                Initial value for the teacher temperature: 0.04 works well in most cases.
                Try decreasing it if the training loss does not decrease.
            teacher_temp (float, optional):
                Final value (after linear warmup)
                of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
                starting with the default value of 0.04 and increase this slightly if needed.
            warmup_teacher_temp_epochs (int, optional):
                'Number of warmup epochs for the teacher temperature
            n_epochs (int, optional): Number of epochs for the training. Defaults to 30.
            student_temp (float, optional): Temperature for the student. Defaults to 0.1.
            center_momentum (float, optional): Momentum for the center update. Defaults to 0.9.
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.n_epochs = n_epochs
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        
        self.register_buffer("center", torch.zeros(1, self.hidden_size))
        
        self.teacher_tmp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(n_epochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        
    
    def forward(self, student_output, teacher_output, epoch):
        """_summary_

        Args:
            student_output (torch.Tensor): Student output
            teacher_output (torch.Tensor): Teacher output
            epoch (int): Current epoch

        Returns:
            torch.Tensor: Loss
        """
        if student_output.dim() == 3:
            student_output = student_output.squeeze(1)
        if teacher_output.dim() == 3:
            teacher_output = teacher_output.squeeze(1)
        student_output = student_output / self.student_temp
        
        temp = self.teacher_tmp_schedule[epoch] if epoch < self.n_epochs else self.teacher_tmp_schedule[-1]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach()
        
        loss = -torch.mean(torch.sum(teacher_out * F.log_softmax(student_output, dim=-1), dim=-1))
        
        self.update_center(teacher_output)
        return loss
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """_summary_

        Args:
            teacher_output (torch.Tensor): Teacher output
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True) / len(teacher_output)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        

if __name__ == "__main__":
    student_output = torch.randn(10, 1, 64)
    teacher_output = torch.randn(10, 1, 64)
    epoch = 0
    criterion = DINOLoss()
    loss = criterion(student_output, teacher_output, epoch)
    print(loss)
    print(criterion.center)
    print(criterion.center.shape)