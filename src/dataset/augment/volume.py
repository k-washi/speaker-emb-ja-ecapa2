import torch
import numpy as np

class VolumeAugment(torch.nn.Module):
    def __init__(
        self, 
        volume_mul_params: list=[0.25, 0.5, 0.75, 0.95], 
        volume_aug_rate: float=0.8
    ) -> None:
        """
        Initialize the VolumeAugment module.

        Args:
            volume_mul_params (list, optional): List of volume multiplier parameters. Defaults to [0.25, 0.5, 0.75, 0.95].
            volume_aug_rate (float, optional): Volume augmentation rate. Defaults to 0.8.
        """
        super().__init__()
        
        self.volume_mul_params = volume_mul_params
        self.volume_aug_rate = volume_aug_rate
        
    def forward(self, x: torch.Tensor) -> torch:
        """
        Apply volume augmentation to the input waveform.

        Args:
            x (np.ndarray): Input waveform (audio length).

        Returns:
            np.ndarray: Augmented waveform.
        """
        if torch.rand((1)).item() > self.volume_aug_rate:
            return x
        
        mul_param_index = torch.randint(0, len(self.volume_mul_params), size=(1,)).item()
        mul_param = self.volume_mul_params[mul_param_index]
        assert np.abs(x).max() > 0, "volume_max should be greater than 0"
        x = x / np.abs(x).max() * mul_param
        return x