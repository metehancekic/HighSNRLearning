import torch
from torch import nn
import torch.nn.functional as F

from itertools import product


class Binarization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.sign(x-0.5)/2. + 0.5

    @staticmethod
    def backward(ctx, grad_wrt_output):
        return grad_wrt_output, None

