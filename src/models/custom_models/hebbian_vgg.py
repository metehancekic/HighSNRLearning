'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from ..custom_layers import Normalize

from pytorch_utils.layers import DivisiveNormalization2d, Threshold, AdaptiveThreshold

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def _thresholding(x, threshold):
    return x*(x > threshold)

class ImplicitNormalizationConv_v1(nn.Conv2d):
    # this version does not take absolute value
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_norms = (self.weight**2).sum(dim=(1, 2, 3),
                                            keepdim=True).transpose(0, 1).sqrt()

        conv = super().forward(x)
        return conv/(weight_norms+1e-6)

class Implicit_Divisive_VGG(nn.Module):
    def __init__(self, divisive_sigma=0.5, vgg_name="VGG16", dropout: float = 0.5):
        super(Implicit_Divisive_VGG, self).__init__()
        self.norm = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
            0.2471, 0.2435, 0.2616])
        self.vgg_name = vgg_name
        self.divisive_sigma = divisive_sigma

        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.norm(x)
        out = self.features(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        counter = 0
        for x in cfg:
            counter += 1
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if counter < 9:
                    layers += [ImplicitNormalizationConv_v1(in_channels, x, kernel_size=3, padding=1, bias=False),
                                nn.ReLU(inplace=False),
                                DivisiveNormalization2d(sigma=self.divisive_sigma)
                                ]
                else:
                    layers += [ImplicitNormalizationConv_v1(in_channels, x, kernel_size=3, padding=1, bias=False),
                            nn.ReLU(inplace=False),
                            nn.BatchNorm2d(x)
                            ]
                in_channels = x
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        return nn.Sequential(*layers)

    @property
    def name(self) -> str:
        return f"Implicit_Divisive_{self.vgg_name}_div_{self.divisive_sigma}"


class Implicit_Divisive_Threshold_VGG(nn.Module):
    def __init__(self, divisive_sigma=0.5, threshold=0.0, vgg_name="VGG16", dropout: float = 0.5):
        super(Implicit_Divisive_Threshold_VGG, self).__init__()
        self.norm = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
            0.2471, 0.2435, 0.2616])
        self.vgg_name = vgg_name
        self.divisive_sigma = divisive_sigma
        self.threshold = threshold

        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 1024),
        #     nn.ReLU(True),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(True),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(1024, 10),
        # )
        


    def forward(self, x):
        out = self.norm(x)
        out = self.features(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        counter = 0
        for x in cfg:
            counter += 1
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if counter < 9:
                    layers += [ImplicitNormalizationConv_v1(in_channels, x, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True),
                                DivisiveNormalization2d(sigma=self.divisive_sigma),
                                Threshold(self.threshold)
                                ]
                else:
                    layers += [ImplicitNormalizationConv_v1(in_channels, x, kernel_size=3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.BatchNorm2d(x)
                            ]
                in_channels = x
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        return nn.Sequential(*layers)

    @property
    def name(self) -> str:
        return f"Implicit_Divisive_{self.vgg_name}_div_{self.divisive_sigma}_threshold_{self.threshold}"

class Implicit_Divisive_Adaptive_Threshold_VGG(nn.Module):
    def __init__(self, divisive_sigma=0.5, threshold=1.0, vgg_name="VGG16", dropout: float = 0.5):
        super(Implicit_Divisive_Adaptive_Threshold_VGG, self).__init__()
        self.norm = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
            0.2471, 0.2435, 0.2616])
        self.vgg_name = vgg_name
        self.divisive_sigma = divisive_sigma
        self.threshold = threshold

        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

        # self.features[0].weight.data = torch.ones_like(self.features[0].weight.data)
        # self.features[0].weight.requires_grad = False
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 1024),
        #     nn.ReLU(True),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(True),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(1024, 10),
        # )
        


    def forward(self, x):
        out = self.norm(x)
        out = self.features(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        counter = 0
        for x in cfg:
            counter += 1
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if counter < 9:
                    layers += [ImplicitNormalizationConv_v1(in_channels, x, kernel_size=3, padding=1, bias=False),
                                nn.ReLU(inplace=False),
                                DivisiveNormalization2d(sigma=self.divisive_sigma),
                                AdaptiveThreshold(mean_scalar=self.threshold)
                                ]
                else:
                    layers += [ImplicitNormalizationConv_v1(in_channels, x, kernel_size=3, padding=1, bias=False),
                            nn.ReLU(inplace=False),
                            nn.BatchNorm2d(x)
                            ]
                in_channels = x
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        return nn.Sequential(*layers)

    @property
    def name(self) -> str:
        return f"Implicit_Divisive_{self.vgg_name}_div_{self.divisive_sigma}_adaptive_threshold_{self.threshold}"


class Implicit_VGG(nn.Module):
    def __init__(self, divisive_sigma=0.5, vgg_name="VGG16", dropout: float = 0.5):
        super(Implicit_VGG, self).__init__()
        self.norm = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
            0.2471, 0.2435, 0.2616])
        self.vgg_name = vgg_name
        self.divisive_sigma = divisive_sigma

        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 1024),
        #     nn.ReLU(True),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(True),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(1024, 10),
        # )
        


    def forward(self, x):
        out = self.norm(x)
        out = self.features(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [ImplicitNormalizationConv_v1(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True),
                           ]
                in_channels = x
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        return nn.Sequential(*layers)

    @property
    def name(self) -> str:
        return f"Implicit_{self.vgg_name}"

class Divisive_VGG(nn.Module):
    def __init__(self, divisive_sigma=0.5, vgg_name="VGG16", dropout: float = 0.5):
        super(Divisive_VGG, self).__init__()
        self.vgg_name = vgg_name
        self.divisive_sigma = divisive_sigma
        self.norm = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
            0.2471, 0.2435, 0.2616])

        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
        
    def forward(self, x):
        out = self.norm(x)
        out = self.features(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU(inplace=True),
                           DivisiveNormalization2d(sigma=self.divisive_sigma)]
                in_channels = x
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        return nn.Sequential(*layers)

    @property
    def name(self) -> str:
        return f"Divisive_{self.vgg_name}"


class Standard_VGG(nn.Module):
    def __init__(self, vgg_name="VGG16", dropout: float = 0.5):
        super(Standard_VGG, self).__init__()
        self.vgg_name = vgg_name
        self.norm = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
            0.2471, 0.2435, 0.2616])

        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 1024),
        #     nn.ReLU(True),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(True),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(1024, 10),
        # )
        
    def forward(self, x):
        out = self.norm(x)
        out = self.features(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        return nn.Sequential(*layers)

    @property
    def name(self) -> str:
        return f"{self.vgg_name}"

class Standard_Nobias_VGG(nn.Module):
    def __init__(self, vgg_name="VGG16", dropout: float = 0.5):
        super(Standard_Nobias_VGG, self).__init__()
        self.vgg_name = vgg_name
        self.norm = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
            0.2471, 0.2435, 0.2616])

        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
        self._initialize_weights()
        
    def forward(self, x):
        out = self.norm(x)
        out = self.features(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                           nn.ReLU(inplace=False),
                           nn.BatchNorm2d(x)]
                in_channels = x
        # layers += [nn.AdaptiveAvgPool2d((1, 1))]
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    @property
    def name(self) -> str:
        return f"Nobias_{self.vgg_name}"