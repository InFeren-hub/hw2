import torch.nn as nn
from torchvision.models import ResNet34_Weights, resnet34


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 注意力模块：全局池化 → FC → ReLU → FC → Sigmoid → 通道加权"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch, channels, _, _ = x.shape
        weights = self.avg_pool(x).view(batch, channels)  
        weights = self.fc(weights).view(batch, channels, 1, 1)  
        return x * weights  


class ResNet34PetClassifier(nn.Module):

    def __init__(self, num_classes: int = 37, pretrained: bool = True, use_attention: bool = False):
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet34(weights=weights)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

        features = base.fc.in_features
        self.attn = SEBlock(features) if use_attention else nn.Identity()  # 无注意力时退化为恒等映射
        self.fc = nn.Linear(features, num_classes)  # 替换1000类为37类

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attn(x)   # SE-Block
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.fc(x)


def build_model(num_classes: int = 37, pretrained: bool = True, use_attention: bool = False):
    return ResNet34PetClassifier(num_classes=num_classes, pretrained=pretrained, use_attention=use_attention)


def parameter_groups(model: nn.Module, lr_backbone: float, lr_head: float):
    """分层学习率：fc 和 attn 层用 lr_head，其余 backbone 用 lr_backbone"""
    backbone, head = [], []
    for name, param in model.named_parameters():
        if name.startswith("fc.") or name.startswith("attn."):
            head.append(param)
        else:
            backbone.append(param)
    return [
        {"params": backbone, "lr": lr_backbone},
        {"params": head, "lr": lr_head},
    ]
