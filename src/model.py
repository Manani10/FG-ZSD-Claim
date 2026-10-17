"""
Backbone + classifier head.

The paper's detector backbone is Faster R-CNN with a ResNet-101 trunk
(Sec. 5.1). Our scoped-down claim is about the *classification* effect of
MSA-hicl, not detection, so we keep the ResNet-101 trunk (ImageNet
pretrained, matching the paper) but drop Faster R-CNN's region-proposal
machinery entirely and attach a plain linear classifier head instead. This
is the single biggest scope reduction versus the original paper and is
called out in the README/write-up, not buried here.
"""

import torch.nn as nn
from torchvision import models


class BirdClassifier(nn.Module):
    def __init__(self, num_classes: int = 200, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet101(weights=weights)
        feature_dim = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_features: bool = False):
        features = self.backbone(x)       # (B, 2048) -- this is what msa_hicl_loss pulls/pushes
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


def make_model(num_classes: int = 200, pretrained: bool = True) -> BirdClassifier:
    return BirdClassifier(num_classes=num_classes, pretrained=pretrained)
