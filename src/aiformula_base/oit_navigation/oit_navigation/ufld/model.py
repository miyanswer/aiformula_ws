"""
model.py - Inference-only re-implementation of the official Ultra-Fast-Lane-Detection
ResNet-18 "parsingNet" (cfzd/Ultra-Fast-Lane-Detection). Only the layers needed to load the
official pretrained checkpoint and run forward inference are included (no aux segmentation
head, no training-only code), so the full training repo does not need to be vendored.

State-dict key layout matches the official checkpoint exactly:
  model.<resnet18 layers>, pool.{weight,bias}, cls.{0,2}.{weight,bias}
"""

import torch
import torch.nn as nn
import torchvision


class _ResNet18Backbone(nn.Module):
    """Wraps torchvision resnet18 to expose the layer4 feature map, matching the
    official backbone.py key names (self.model.conv1, self.model.layer1, ...)."""

    def __init__(self):
        super().__init__()
        net = torchvision.models.resnet18(weights=None)
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ParsingNet(nn.Module):
    """UFLD row-anchor classification head (use_aux=False inference path only)."""

    def __init__(self, cls_dim=(101, 56, 4)):
        super().__init__()
        self.cls_dim = cls_dim  # (griding_num + 1, num_row_anchors, num_lanes)
        total_dim = cls_dim[0] * cls_dim[1] * cls_dim[2]

        self.model = _ResNet18Backbone()
        self.pool = nn.Conv2d(512, 8, 1)
        self.cls = nn.Sequential(
            nn.Linear(1800, 2048),
            nn.ReLU(),
            nn.Linear(2048, total_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fea = self.model(x)
        fea = self.pool(fea).view(fea.shape[0], -1)
        group_cls = self.cls(fea).view(-1, *self.cls_dim)
        return group_cls
