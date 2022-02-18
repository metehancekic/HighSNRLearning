import torch
from torch import nn
from torch.nn import Module
from torch import Tensor

class Threshold(Module):
    r"""
    Thresholds values x[x>threshold]
    """

    def __init__(self, threshold: float) -> None:
        super(Threshold, self).__init__()

        self.threshold = threshold

    def _thresholding(self, x):
        return x*(x > self.threshold)
        
    def forward(self, input: Tensor) -> Tensor:
        return self._thresholding(input)

    def __repr__(self) -> str:
        s = f"Threshold(threshold={self.threshold})"
        return s

class AdaptiveThreshold(Module):
    r"""
    Thresholds values x[x>threshold]
    """

    def __init__(self, mean_scalar: float=1.0) -> None:
        super(AdaptiveThreshold, self).__init__()

        self.mean_scalar = mean_scalar

    def _thresholding(self, x, threshold):
        return x*(x > threshold)
        
    def forward(self, input: Tensor) -> Tensor:
        means = input.mean(dim=(2,3), keepdim=True)
        return self._thresholding(input, means*self.mean_scalar)

    def __repr__(self) -> str:
        s = f"AdaptiveThreshold(mean_scalar={self.mean_scalar})"
        return s