'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

from ..custom_activations import TReLU, Quad, TReLU_with_trainable_bias
from ..custom_layers import Normalize, take_top_k


class VGGblock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.conv = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(self.out_channel, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        o = self.conv(x)
        o = self.bn(o)
        o = self.relu(o)
        return o

    def __repr__(self):
        s = f"[conv({self.in_channel}, {self.out_channel}), top({self.out_channel//32}), bn, relu]"
        return s.format(**self.__dict__)


class VGGblock_without_bn(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.conv = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, padding=1)
        # self.bn = nn.BatchNorm2d(self.out_channel, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        o = self.conv(x)
        # o = self.bn(o)
        o = self.relu(o)
        return o

    def __repr__(self):
        s = f"[conv({self.in_channel}, {self.out_channel}), top({self.out_channel//32}), relu]"
        return s.format(**self.__dict__)


class Topkblock(nn.Module):
    def __init__(self, in_channel, out_channel, k):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.k = k

        self.conv = nn.Conv2d(self.in_channel, self.out_channel,
                              kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(self.out_channel, affine=True)
        self.relu = TReLU_with_trainable_bias(self.out_channel)

    def bias_calculator(self, filters):
        bias = torch.sum(torch.abs(filters), dim=(1, 2, 3)).unsqueeze(dim=0)
        bias = bias.unsqueeze(dim=2)
        bias = bias.unsqueeze(dim=3)
        return bias

    def forward(self, x, gamma=1.):
        o = self.conv(x)
        bias = self.bias_calculator(self.conv.weight)
        if self.training:
            noise = bias * 8./255 * (torch.rand_like(o, device=o.device) * gamma * 2 - gamma)
            o = o + noise
            # breakpoint()
        o = take_top_k(o, self.k)
        # o = self.bn(o)
        o = self.relu(o)
        return o

    def __repr__(self):
        s = f"[conv({self.in_channel}, {self.out_channel}), top({self.k}), bn, relu]"
        return s.format(**self.__dict__)


class topk_VGG(nn.Module):
    def __init__(self, vgg_name="VGG11", gamma=1., k=20):
        super().__init__()
        self.norm = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])

        self.gamma = gamma
        self.k = k

        self.block1 = Topkblock(3, 640, k=k)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block2 = VGGblock(640, 128)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block3 = VGGblock(128, 256)
        self.block4 = VGGblock(256, 256)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block5 = VGGblock(256, 512)
        self.block6 = VGGblock(512, 512)
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block7 = VGGblock(512, 512)
        self.block8 = VGGblock(512, 512)
        self.mp5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.linear = nn.Linear(512, 10)

    def forward(self, x):
        o = self.norm(x)
        o = self.block1(o, gamma=self.gamma)
        o = self.mp1(o)
        o = self.block2(o)
        o = self.mp2(o)
        o = self.block3(o)
        o = self.block4(o)
        o = self.mp3(o)
        o = self.block5(o)
        o = self.block6(o)
        o = self.mp4(o)
        o = self.block7(o)
        o = self.block8(o)
        o = self.mp5(o)
        o = o.view(o.size(0), -1)
        o = self.linear(o)
        return o

    def model_name(self):
        return f"VGG_top_{int(self.k)}_gamma_{self.gamma}"


class VGG11_basis(nn.Module):
    def __init__(self, vgg_name="VGG11", gamma=1., k=20):
        super().__init__()
        self.norm = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])

        self.gamma = gamma
        self.k = k

        self.block1 = VGGblock_without_bn(3, 640)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block2 = VGGblock(640, 128)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block3 = VGGblock(128, 256)
        self.block4 = VGGblock(256, 256)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block5 = VGGblock(256, 512)
        self.block6 = VGGblock(512, 512)
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block7 = VGGblock(512, 512)
        self.block8 = VGGblock(512, 512)
        self.mp5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.linear = nn.Linear(512, 10)

    def forward(self, x):
        o = self.norm(x)
        o = self.block1(o)
        o = self.mp1(o)
        o = self.block2(o)
        o = self.mp2(o)
        o = self.block3(o)
        o = self.block4(o)
        o = self.mp3(o)
        o = self.block5(o)
        o = self.block6(o)
        o = self.mp4(o)
        o = self.block7(o)
        o = self.block8(o)
        o = self.mp5(o)
        o = o.view(o.size(0), -1)
        o = self.linear(o)
        return o

    def model_name(self):
        return f"VGG11_basis"
