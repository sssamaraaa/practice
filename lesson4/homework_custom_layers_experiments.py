import torch
import torch.nn as nn
from lesson4.models.custom_layers import CustomCNN, CustomConvLayer, L2Pooling, SEAttention, Swish
from lesson4.utils.experiment_utils import count_parameters


def test_custom_layer(layer, input_shape):
    print(layer.__class__.__name__)
    
    x = torch.randn(*input_shape, requires_grad=True)
    y = layer(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)

    loss = y.mean()
    loss.backward()

    print("Forward OK")
    print("Backward OK")

    if x.grad is not None:
        print("Gradient mean:", x.grad.abs().mean().item())

def run_layer_tests():
    test_custom_layer(Swish(), (4, 16))
    test_custom_layer(L2Pooling(), (2, 32, 8, 8))
    test_custom_layer(CustomConvLayer(3, 16), (2, 3, 32, 32))
    test_custom_layer(SEAttention(32), (2, 32, 16, 16))

def compare_standard_layers():
    print("\nComparison with standard layers")

    x = torch.randn(4, 3, 32, 32)
    custom_conv = CustomConvLayer(3, 32)
    standard_conv = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU()
    )

    y1 = custom_conv(x)
    y2 = standard_conv(x)

    print("Custom Conv output :", y1.shape)
    print("Standard Conv output:", y2.shape)

    custom_pool = L2Pooling()
    max_pool = nn.AdaptiveMaxPool2d(1)
    avg_pool = nn.AdaptiveAvgPool2d(1)

    print("L2 Pool:", custom_pool(y1).shape)
    print("Max Pool:", max_pool(y1).shape)
    print("Avg Pool:", avg_pool(y1).shape)

    swish = Swish()
    relu = nn.ReLU()
    t = torch.randn(8)

    print("\nActivation comparison")
    print("Input :", t)
    print("ReLU  :", relu(t))
    print("Swish :", swish(t))

if __name__ == "__main__":
    run_layer_tests()
    compare_standard_layers()
    model = CustomCNN()
    x = torch.randn(4, 3, 32, 32)
    y = model(x)

    print("\nCustom CNN")
    print(model)
    print("Output shape:", y.shape)

    params = count_parameters(model)

    print("Trainable parameters:", params)