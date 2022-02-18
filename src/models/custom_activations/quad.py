import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch import nn


class Quad(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.sigmoid(x+1.)-F.sigmoid(x-1.)
