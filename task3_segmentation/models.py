import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):

    def __init__(self, num_classes=3, base=32):
        super().__init__()

        self.pool = nn.MaxPool2d(2)
        self.e1 = DoubleConv(3, base)
        self.e2 = DoubleConv(base, base * 2)
        self.e3 = DoubleConv(base * 2, base * 4)
        self.e4 = DoubleConv(base * 4, base * 8)
        self.mid = DoubleConv(base * 8, base * 16)

        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.d4 = DoubleConv(base * 16, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.d3 = DoubleConv(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.d2 = DoubleConv(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.d1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        # 下采样
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        x = self.mid(self.pool(e4))
        # 上采样 + skip connection
        x = self.d4(torch.cat([self.u4(x), e4], dim=1))
        x = self.d3(torch.cat([self.u3(x), e3], dim=1))
        x = self.d2(torch.cat([self.u2(x), e2], dim=1))
        x = self.d1(torch.cat([self.u1(x), e1], dim=1))
        return self.out(x)


class DiceLoss(nn.Module):
    def __init__(self, num_classes=3, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, masks):
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(masks, self.num_classes).permute(0, 3, 1, 2).float()
        inter = (probs * one_hot).sum(dim=(0, 2, 3))
        union = (probs + one_hot).sum(dim=(0, 2, 3))
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


def build_loss(name, num_classes):
    ce = nn.CrossEntropyLoss()
    dice = DiceLoss(num_classes)
    if name == "ce":
        return ce
    if name == "dice":
        return dice
    if name == "ce_dice":
        return lambda logits, masks: ce(logits, masks) + dice(logits, masks)
    raise ValueError(name)
