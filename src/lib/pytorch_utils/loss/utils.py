from typing import Dict, Union, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["hoyer"]


def hoyer(x: torch.Tensor, dim: Union[int, Tuple[int]] = None, epsilon: float = 0.000000000001) -> torch.float:
    """
        Takes a tensor and dimensions to compute Hoyer

    returns |x|_1/(|x|_2 + epsilon)
    """

    l1 = torch.sum(torch.abs(x), dim=dim, keepdim=True)
    l2 = torch.sqrt(torch.sum(x**2, dim=dim, keepdim=True))
    if torch.any(torch.isnan(l1/(l2+epsilon))):
        breakpoint()
    return l1/(l2+epsilon)
