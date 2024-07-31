import numpy as np
import torch
def get_mixup_lambda(alpha:float, beta:float):
    """
    Get mixup lambda value from beta distribution.

    Args:
        alpha (float): Alpha parameter of the beta distribution.
        beta (float): Beta parameter of the beta distribution.

    Returns:
        float: Mixup lambda value.
    """
    return np.random.beta(alpha, beta)


def mixup(
    x0:torch.Tensor,
    y0:int, 
    x1:torch.Tensor,
    y1:int,
    mixup_lambda:float, 
    num_classes:int,
    spec_normalize:bool=True
):
    """
    Mixup the input data.

    Args:
        x (torch.Tensor): Input data.
        y (torch.Tensor): Labels.
        mixup_lambda (float): Mixup lambda value.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Mixed data and labels.
    """
    is_squeeze = False
    if x0.dim() == 3:
        x0 = x0.squeeze(0)
        x1 = x1.squeeze(0)
        is_squeeze = True
    x0_length = x0.size(-1)
    x1_length = x1.size(-1)
    if x0_length > x1_length:
        x0 = x0[:, :x1_length]
    elif x0_length < x1_length:
        x1 = x1[:, :x0_length]
    if spec_normalize:
        if x0.abs().sum() > 0:
            x0 = x0 / torch.mean(torch.linalg.norm(x0, ord=2, dim=0))

        if x1.abs().sum() > 0:
            x1 = x1 / torch.mean(torch.linalg.norm(x1, ord=2, dim=0))

    mixed_x = mixup_lambda * x0 + (1 - mixup_lambda) * x1
    y0 = torch.nn.functional.one_hot(torch.tensor(y0), num_classes=num_classes)
    y1 = torch.nn.functional.one_hot(torch.tensor(y1), num_classes=num_classes)
    mixed_y = mixup_lambda * y0 + (1 - mixup_lambda) * y1
    if is_squeeze:
        mixed_x = mixed_x.unsqueeze(0)
        mixed_y = mixed_y.unsqueeze(0)
    return mixed_x, mixed_y