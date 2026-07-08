import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d( in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)
    

class BottleneckResidualBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        width = out_channels
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d( width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        return F.relu(out)
    

class WideResidualBlock(nn.Module):
    def __init__( self, in_channels, out_channels, stride=1, widen_factor=2):
        super().__init__()
        width = out_channels * widen_factor
        self.conv1 = nn.Conv2d( in_channels, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(width)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)
    

class ResidualCNN(nn.Module):
    def __init__(self,block,num_classes=10):
        super().__init__()
        self.block = block
        self.conv = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())

        if block == BottleneckResidualBlock:
            self.layer1 = block(32, 32)
            self.layer2 = block(128, 64, stride=2)
            final_channels = 256
        elif block == WideResidualBlock:
            self.layer1 = block(32, 32)
            self.layer2 = block(64, 64, stride=2)
            final_channels = 128
        else:
            self.layer1 = block(32, 32)
            self.layer2 = block(32, 64, stride=2)
            final_channels = 64

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(final_channels, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)