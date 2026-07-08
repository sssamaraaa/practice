import torch
import torch.nn as nn
import torch.nn.functional as F


class SwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        sigmoid = torch.sigmoid(x)
        ctx.save_for_backward(x, sigmoid)
        return x * sigmoid
    
    @staticmethod
    def backward(ctx, grad_output):
        x, sigmoid = ctx.saved_tensors
        grad = sigmoid + x * sigmoid * (1 - sigmoid)
        return grad_output * grad

class Swish(nn.Module):
    def forward(self, x):
        return SwishFunction.apply(x)


class L2PoolingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # x: (N,C,H,W)
        y = torch.sqrt(torch.mean(x ** 2, dim=(2, 3), keepdim=True) + 1e-8)
        ctx.save_for_backward(x, y)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        h = x.size(2)
        w = x.size(3)
        grad = x / (h * w * y)
        return grad * grad_output

class L2Pooling(nn.Module):
    def forward(self, x):
        return L2PoolingFunction.apply(x)


class CustomConvLayer(nn.Module):
    """
    Conv -> BN -> Learnable Scale -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = x * self.scale
        x = F.relu(x)
        return x


class SEAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x)
        w = w.view(b, c)
        w = self.fc(w)
        w = w.view(b, c, 1, 1)
        return x * w


class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            CustomConvLayer(3, 32),
            SEAttention(32),
            nn.MaxPool2d(2),
            CustomConvLayer(32, 64),
            SEAttention(64),
            nn.MaxPool2d(2)
        )

        self.pool = L2Pooling()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            Swish(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x