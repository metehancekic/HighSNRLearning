"""
Attention Layers
"""

import numpy as np

import torch.nn as nn
import torch
import torch.nn.functional as F


class Normalize(nn.Module):
    """Data normalizing class as torch.nn.Module

    Attributes:
        mean (float): Mean value of the training dataset.
        std (float): Standard deviation value of the training dataset.

    """

    def __init__(self, mean, std):
        """

        Args:
            mean (float): Mean value of the training dataset.
            std (float): Standard deviation value of the training dataset.

        """
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        """
        Args:
            x (tensor batch): Input tensor.
        Returns:
            Normalized data.
        """
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]
