import torch
import torch.nn as nn
import numpy as np

from .resnet import PreActBlock


class PolicyNetV1(nn.Module):
    def __init__(self, forward_settings, nmodels=2):
        super().__init__()
        self.nmodels = nmodels
        planes = [None for _ in range(self.nmodels)]
        self.out_plane = 512
        for ind, in_plane in enumerate(forward_settings["in_planes"]):
            if in_plane == 64:  # Width of each layer
                planes[ind] = (64, 128, 256, 512)
            elif in_plane == 160:
                planes[ind] = (160, 160, 256, 512)
            elif in_plane == 256:
                planes[ind] = (256, 512, 512, 512)
            else:
                raise ValueError("Unknown in_plane.")

        if forward_settings["normalization_type"] == "batch":
            self.n1 = nn.ModuleList([nn.BatchNorm2d(plane[1]) for plane in planes])
            self.n2 = nn.ModuleList([nn.BatchNorm2d(plane[2]) for plane in planes])
            self.n3 = nn.ModuleList([nn.BatchNorm2d(plane[3]) for plane in planes])
            self.nl = nn.BatchNorm1d(self.out_plane)
        elif forward_settings["normalization_type"] == "layer":
            self.n1 = nn.ModuleList([nn.LayerNorm([plane[1], 32, 32]) for plane in planes])
            self.n2 = nn.ModuleList([nn.LayerNorm([plane[2], 32, 32]) for plane in planes])
            self.n3 = nn.ModuleList([nn.LayerNorm([plane[3], 32, 32]) for plane in planes])
            self.nl = nn.LayerNorm(self.out_plane)
        else:
            raise ValueError("Unknown normalization type.")

        self.conv1 = nn.ModuleList([nn.Conv2d(plane[0], plane[1], kernel_size=3, stride=1,
                                              padding=1, bias=False) for plane in planes])
        self.conv2 = nn.ModuleList([nn.Conv2d(plane[1], plane[2], kernel_size=3, stride=1,
                                              padding=1, bias=False) for plane in planes])
        self.conv3 = nn.ModuleList([nn.Conv2d(plane[2], plane[3], kernel_size=3, stride=1,
                                              padding=1, bias=False) for plane in planes])

        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(self.out_plane * self.nmodels, self.out_plane, bias=True)
        self.linear2 = nn.Linear(self.out_plane, 1, bias=True)
        self.mp = torch.nn.MaxPool2d(2)

        if forward_settings["nonlinearity"] == "gelu":
            self.nonlin = torch.nn.GELU()
        elif forward_settings["nonlinearity"] == "relu":
            self.nonlin = torch.nn.ReLU()

    def forward(self, feats, feats2):
        for ind in range(self.nmodels):
            feats[ind] = self.n1[ind](self.nonlin(self.mp(self.conv1[ind](feats[ind]))))
            feats[ind] = self.n2[ind](self.nonlin(self.mp(self.conv2[ind](feats[ind]))))
            feats[ind] = self.n3[ind](self.nonlin(self.mp(self.conv3[ind](feats[ind]))))
            feats[ind] = self.global_avg_pool(feats[ind])

        feats = torch.cat(feats, dim=1).reshape(-1, self.out_plane * self.nmodels)
        feats = self.nl(self.nonlin(self.linear1(feats)))
        feats = self.linear2(feats)
        return feats


class PolicyNetV3(nn.Module):
    def __init__(self, forward_settings, nmodels=2):
        super().__init__()
        print("Initializing PolicyNet V3.")
        self.nmodels = nmodels

        planes = [None for _ in range(self.nmodels)]
        for ind, in_plane in enumerate(forward_settings["in_planes"]):
            if in_plane == 64:  # Width of each layer
                planes[ind] = (64, 128, 256, 512)  # (64, 128, 256, 320)
            elif in_plane == 160:
                planes[ind] = (160, 320, 512, 768)  # (160, 320, 320, 320)
            else:
                raise ValueError("Unknown in_plane.")
        self.planes = np.array(planes).sum(axis=0)

        self.in_planes = self.planes[0]
        self.layers = nn.ModuleList([self._make_layer(self.planes[l + 1], stride=2)
                                     for l in range(len(self.planes) - 1)])

        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(self.planes[-1], 1, bias=False)

    def _make_layer(self, planes, stride, num_blocks=2):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(PreActBlock(self.in_planes, planes, stride))
            self.in_planes = planes * PreActBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, feats1, feats2):
        feats1, feats2 = torch.cat(feats1, dim=1), torch.cat(feats2, dim=1)
        feats = self.layers[0](feats1) + feats2
        for l in range(1, len(self.layers)):
            feats = self.layers[l](feats)

        feats = self.global_avg_pool(feats).reshape(-1, self.planes[-1])
        feats = self.linear(feats)
        return feats


class PolicyNetV4(nn.Module):
    def __init__(self, forward_settings, nmodels=2):
        super().__init__()
        print("Initializing PolicyNet V4.")
        self.nmodels = nmodels

        planes = [None for _ in range(self.nmodels)]
        for ind, in_plane in enumerate(forward_settings["in_planes"]):
            if in_plane == 512:  # Width of each layer
                planes[ind] = (512, 256, 384, 512)
            elif in_plane == 256:
                planes[ind] = (256, 256, 384, 512)
            else:
                raise ValueError("Unknown in_plane.")
        self.planes = np.array(planes).sum(axis=0)

        self.in_planes = self.planes[0]
        in_1x1, out_1x1 = (planes[0][0] + planes[1][0]) * 2, planes[0][1] + planes[1][1]
        self.conv1x1 = nn.Conv2d(in_1x1, out_1x1, kernel_size=1, stride=1, bias=False)
        self.layers = nn.ModuleList([self._make_layer(self.planes[l + 1], stride=2)
                                     for l in range(len(self.planes) - 1)])

        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(self.planes[-1], 1, bias=False)

    def _make_layer(self, planes, stride, num_blocks=2):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(PreActBlock(self.in_planes, planes, stride))
            self.in_planes = planes * PreActBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, feats1, feats2):
        feats1, feats2 = torch.cat(feats1, dim=1), torch.cat(feats2, dim=1)
        feats = self.layers[0](feats1) + self.conv1x1(feats2)
        for l in range(1, len(self.layers)):
            feats = self.layers[l](feats)

        feats = self.global_avg_pool(feats).reshape(-1, self.planes[-1])
        feats = self.linear(feats)
        return feats
