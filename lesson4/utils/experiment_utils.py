import torch
import time
import torch.nn as nn


def count_parameters(model):
    """Подсчитывает количество параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, path):
    """Сохраняет модель"""
    torch.save(model.state_dict(), path)


def load_model(model, path):
    """Загружает модель"""
    model.load_state_dict(torch.load(path))
    return model


def inference_time(model, loader, device):
    model.eval()
    start = time.time()

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            model(x)

    return time.time() - start


def compute_receptive_field(model):
    rf = 1          # receptive field
    jump = 1        # шаг между соседними пикселями предыдущего слоя

    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            k = layer.kernel_size[0]
            s = layer.stride[0]
            rf = rf + (k - 1) * jump
            jump *= s

        elif isinstance(layer, nn.MaxPool2d):
            k = layer.kernel_size
            s = layer.stride
            if isinstance(k, tuple):
                k = k[0]
            if isinstance(s, tuple):
                s = s[0]

            rf = rf + (k - 1) * jump
            jump *= s

    return rf