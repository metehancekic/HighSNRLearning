"""
Neural Network models for training and testing implemented in PyTorch
"""
from typing import TypeVar, Union, Tuple, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class LpConv2d(nn.Module):
    r"""Applies Lp norm as 2d convolutional operation input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`
    and output :math:`(N, C, H, W)`.

    Args:
        in_channels: Type of suppressin field, must be one of (`linf`, `l1`, `l2`).
        out_channels: The size of the suppression field, must be > 0.
        kernel_size: Constant added to suppression field, must be > 0.

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)`

    Examples::

        >>> # suppression of size=3, sigma=1
        >>> d = DivisiveNormalization2d(b_size=3, sigma=1)
        >>> input = torch.randn(20, 16, 50, 50)
        >>> output = d(input)
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int = 1,
            kernel_size: Union[int, Tuple[int, int]] = (5, 5),
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = False,
            padding_mode: str = "zeros",
            p_norm: Union[str, int] = 1,
            ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.p_norm = p_norm
        self.bias = None

        self.weight = Parameter(torch.ones((out_channels, in_channels//groups,
                                            self.kernel_size[0], self.kernel_size[1])), requires_grad=False)

        self.register_parameter("weight", self.weight)

    def forward(self, input_tensor):
        if self.p_norm == 1:
            o = torch.abs(input_tensor)
            o = F.conv2d(input=o, weight=self.weight, bias=None, stride=self.stride,
                         padding=self.padding, dilation=self.dilation, groups=self.groups)
        elif self.p_norm == 2:
            o = input_tensor*input_tensor
            o = torch.sqrt(F.conv2d(input=o, weight=self.weight, bias=None, stride=self.stride,
                                    padding=self.padding, dilation=self.dilation, groups=self.groups))
        elif self.p_norm == "inf":
            o = torch.abs(input_tensor)
            o, _ = torch.max(F.max_pool2d(torch.abs(o), self.kernel_size,
                                          self.stride, self.padding, self.dilation), dim=1, keepdim=True)
        else:
            raise NotImplementedError(f"P_norm should be one of 1, 2, 'inf', however given as {self.p_norm}.")
        return o

    def __repr__(self):
        s = f"L{self.p_norm}Conv2d("
        s += f'{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}'
        if self.padding != 0:
            s += f', padding={self.padding}'
        if self.dilation != 1:
            s += f', dilation={self.dilation}'
        if self.groups != 1:
            s += f', groups={self.groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += f', padding_mode={self.padding_mode}'
        s += ")"
        return s.format(**self.__dict__)
