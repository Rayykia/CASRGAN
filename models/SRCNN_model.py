"""SRCNN model.

Date: 3/14/2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from . import utils

from torch import Tensor


class SRCNNModel(nn.Module):
    """
    This class implements the SRCNN model.
    """
    def __init__(self, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)
        self.patch_extraction = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=9, stride=1, padding=4
        )

        self.non_linear_mapping = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=1, stride=1
        )

        self.reconstruction = nn.Conv2d(
            in_channels=32, out_channels=1, kernel_size=5, stride=1, padding=2
        )
    
    def forward(
            self,
            x: Tensor
    ) -> Tensor:
        """Standard forward.

        Parameters:
            x (Tensor)      -- the input image
        
        Returns:
            x               -- the reconstructed image
        
        ..notes:
            For SRCNN, the layers are organized as fallow:
            F1(Y) = max(0, W1*Y+B1)
            F2(Y) = max(0, W2*F1(Y)+B2)
            F3(Y) = W3*F2(Y) + B3
            Activation function is not used in the last layer in the original
            SRCNN paper.
        """
        x = F.relu(self.patch_extraction(x), inplace=True)
        x = F.relu(self.non_linear_mapping(x), inplace=True)
        x = self.reconstruction(x)
        return x
    