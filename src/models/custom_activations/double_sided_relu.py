import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch import nn


class DReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def bias_calculator(self, filters: Parameter) -> torch.Tensor:
        bias = torch.sum(torch.abs(filters), dim=(1, 2, 3)).unsqueeze(dim=0)
        bias = bias.unsqueeze(dim=2)
        bias = bias.unsqueeze(dim=3)
        return bias

    def forward(self, x: torch.Tensor, filters: Parameter, alpha: float = 8./255) -> torch.Tensor:
        bias = alpha * self.bias_calculator(filters)
        return F.relu(x - bias) - F.relu(-x - bias)

    def __repr__(self) -> str:
        s = f"DReLU()"
        return s.format(**self.__dict__)


class DTReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def bias_calculator(self, filters: Parameter) -> torch.Tensor:
        bias = torch.sum(torch.abs(filters), dim=(1, 2, 3)).unsqueeze(dim=0)
        bias = bias.unsqueeze(dim=2)
        bias = bias.unsqueeze(dim=3)
        return bias

    def forward(self, x: torch.Tensor, filters: Parameter, alpha: float = 8./255) -> torch.Tensor:
        bias = alpha * self.bias_calculator(filters)
        return F.relu(x - bias) + bias * torch.sign(F.relu(x - bias)) - F.relu(-x - bias) - bias * torch.sign(F.relu(-x - bias))

    def __repr__(self) -> str:
        s = f"DReLU()"
        return s.format(**self.__dict__)


class TReLU(nn.Module):

    def __init__(self, in_channels: int, layer_type: str = "conv2d"):
        super().__init__()
        self.in_channels = in_channels
        self.layer_type = layer_type
        if layer_type == "conv2d":
            self.bias = Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=False)
        elif layer_type == "conv1d":
            self.bias = Parameter(torch.zeros((1, in_channels, 1)), requires_grad=False)
        elif layer_type == "linear":
            self.bias = Parameter(torch.zeros((1, in_channels)), requires_grad=False)
        else:
            raise NotImplementedError("layer_type should be one of ['conv2d', 'conv1d', 'linear']")

        self.register_parameter("bias", self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x - self.bias) + self.bias * (torch.sign(F.relu(x - self.bias))+1)/2
        # return F.relu(x)

    def __repr__(self) -> str:
        s = f"TReLU({self.in_channels})"
        return s.format(**self.__dict__)


class TReLU_with_trainable_bias(nn.Module):
    def __init__(self, in_channels, layer_type="conv2d"):
        super().__init__()
        self.in_channels = in_channels
        self.layer_type = layer_type
        if layer_type == "conv2d":
            self.bias = Parameter(torch.randn((1, in_channels, 1, 1))/50., requires_grad=True)
        elif layer_type == "conv1d":
            self.bias = Parameter(torch.randn((1, in_channels, 1))/50., requires_grad=True)
        elif layer_type == "linear":
            self.bias = Parameter(torch.randn((1, in_channels))/50., requires_grad=True)
        else:
            raise NotImplementedError
        # torch.nn.init.xavier_normal_(self.bias)
        # self.bias = self.bias/10.
        self.register_parameter("bias", self.bias)

    def forward(self, x):
        return F.relu(x - torch.abs(self.bias)) + torch.abs(self.bias) * torch.sign(F.relu(x - torch.abs(self.bias)))

    def __repr__(self):
        s = f"TReLU_with_trainable_bias(in_channels = {self.in_channels}, layer_type = {self.layer_type})"
        return s.format(**self.__dict__)


def test_DReLU():
    import numpy as np
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(np.arange(-10, 10, 0.1), DReLU(torch.Tensor(np.arange(-10, 10, 0.1)), bias=5))
    plt.savefig("double_sided_relu")


if __name__ == '__main__':
    test_DReLU()
