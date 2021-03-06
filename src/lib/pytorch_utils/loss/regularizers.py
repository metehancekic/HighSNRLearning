from typing import Dict, Union, Tuple, Optional
import numpy as np
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import hoyer


__all__ = ["hoyer", "hoyer_loss", "l1_loss", "l2_loss"]

def hoyer(x: torch.Tensor, dim: Union[int, Tuple[int]] = None, epsilon: float = 0.000000000001) -> torch.float:
    """
        Takes a tensor and dimensions to compute Hoyer

    returns |x|_1/(|x|_2 + epsilon)
    """

    l1 = torch.sum(torch.abs(x), dim=dim, keepdim=True)
    l2 = torch.sqrt(torch.sum(x**2, dim=dim, keepdim=True))
    return l1/(l2+epsilon)

def hoyer_loss(features: Dict[str, torch.Tensor] = {}, dim: Union[int, Tuple[int]] = None, epsilon: float = 0.000000000001) -> torch.float:
    """
    Hoyer square loss: https://arxiv.org/pdf/1908.09979.pdf

    Takes a dictionary of tensors

    returns (|feature|_1)^2/((|feature|_2)^2 + epsilon)
    """
    loss = 0
    for feature in features:
        loss += torch.mean(hoyer(x=features[feature], dim=dim, epsilon=epsilon)**2)
    return loss


def l1_loss(features: Dict[str, torch.Tensor] = {}, dim: Union[int, Tuple[int]] = None) -> torch.float:
    """
    L1 loss: L1 norm of the given tensors disctionary

    Takes a dictionary of tensors

    returns |feature|_1
    """
    loss = 0
    for feature in features:
        loss += torch.mean(torch.sum(torch.abs(features[feature]), dim=dim))
    return loss


def l2_loss(features: Dict[str, torch.Tensor] = {}, dim: Union[int, Tuple[int]] = None) -> torch.float:
    """
    L2 loss: L2 norm of the given tensors disctionary

    Takes a dictionary of tensors

    returns |feature|_2
    """
    loss = 0
    for feature in features:
        loss += torch.mean(torch.sqrt(torch.sum(features[feature]**2, dim=dim)))
    return loss


def matching_K(activations: torch.Tensor, K: int, demote: float = 1.0, dim: int = 1, patches: Optional[torch.Tensor] = None, weights: Optional[torch.Tensor] = None, **kwargs):
    
    activations = F.relu(activations)
    
    if weights is not None:
        weight_norms = torch.norm(
            weights, p=2, dim=(1, 2, 3), keepdim=True).transpose(0, 1)
        activations = activations/(weight_norms+1e-6)

    sorted = torch.sort(activations, dim=dim, descending=True)[0]
    top_K_avg = sorted[:, :K].mean(dim=1)
    bottom_avg = sorted[:, K:].mean(dim=1)
    
    if patches is not None:
        patch_norms = torch.norm(patches, p=2, dim=(1, 2, 3))

        if patches.ndim == 4:
            activations_shape = list(activations.shape)
            activations_shape[1] = 1
            patch_norms = patch_norms.view(activations_shape)

        activations = activations/(patch_norms+1e-6)

        sorted = torch.sort(activations, dim=dim, descending=True)[0]
        top_K_avg_normalized = sorted[:, :K].mean(dim=1)
        bottom_avg_normalized = sorted[:, K:].mean(dim=1)

        return torch.mean(top_K_avg_normalized), torch.mean(bottom_avg_normalized), torch.mean(top_K_avg-demote*bottom_avg)
    
    else:
        return torch.mean(top_K_avg), torch.mean(bottom_avg), torch.mean(top_K_avg-demote*top_K_avg)


def matching_loss(activations: torch.Tensor, ratio: float, demote: float = 1.0, dim: int = 1, patches: Optional[torch.Tensor] = None, weights: Optional[torch.Tensor] = None, **kwargs):
    n_filters = activations.shape[1]
    K = ceil(ratio*n_filters)
    return matching_K(activations=activations, K=K, demote=demote, dim=dim, patches=patches, weights=weights)


