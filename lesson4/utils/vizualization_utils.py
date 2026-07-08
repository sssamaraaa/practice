import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import torch
import torch.nn as nn


def plot_training_history(history, exp_name="", save_path=""):
    """Визуализирует историю обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax2.plot(history['train_accs'], label='Train Acc')
    ax2.plot(history['test_accs'], label='Test Acc')
    ax2.set_title('Accuracy')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f"{save_path}/{exp_name}", dpi=200)
    plt.close()


def compare_models(fc_history, cnn_history, exp_name="", save_path=""):
    """Сравнивает результаты полносвязной и сверточной сетей"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(fc_history['test_accs'], label='FC Network', marker='o')
    ax1.plot(cnn_history['test_accs'], label='CNN', marker='s')
    ax1.set_title('Test Accuracy Comparison')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(fc_history['test_losses'], label='FC Network', marker='o')
    ax2.plot(cnn_history['test_losses'], label='CNN', marker='s')
    ax2.set_title('Test Loss Comparison')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{exp_name}", dpi=200)
    plt.close()

def plot_confusion(model, loader, device, exp_name="", save_path=""):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).argmax(1).cpu()
            y_true.extend(y.numpy())
            y_pred.extend(pred.numpy())

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")
    plt.savefig(f"{save_path}/{exp_name}", dpi=200)
    plt.close


def gradient_flow(model, exp_name="", save_path=""):
    layers = []
    grad_norms = []
    grad_ratios = []

    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grad_norm = p.grad.norm().item()
            weight_norm = p.data.norm().item()
            ratio = grad_norm / (weight_norm + 1e-12)
            layers.append(name)
            grad_norms.append(grad_norm)
            grad_ratios.append(ratio)

            # проверка на затухание/взрыв
            if torch.isnan(p.grad).any():
                print(f"NaN gradient detected: {name}")

            if torch.isinf(p.grad).any():
                print(f"Inf gradient detected: {name}")

    if not grad_norms:
        print("No gradients found")
        return

    # Gradient norm
    plt.figure(figsize=(12, 5))
    plt.plot(grad_norms, marker='o', linewidth=2)
    plt.axhline(y=1e-4, color='orange', linestyle=':', alpha=0.5, label='Very small (1e-4)')
    plt.axhline(y=1e+1, color='red', linestyle=':', alpha=0.5, label='Large (1e+1)')
    plt.xticks(range(len(layers)), layers, rotation=90, fontsize=7)
    plt.yscale("log")
    plt.ylabel("Gradient norm")
    plt.title(f"Gradient Flow: {exp_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_path}/{exp_name}_norm.png", dpi=200)
    plt.close()

    # Relative gradient 
    plt.figure(figsize=(12, 5))
    plt.plot(grad_ratios, marker='o', linewidth=2)
    plt.xticks(range(len(layers)), layers, rotation=90, fontsize=7)
    plt.yscale("log")
    plt.ylabel("Gradient / Weight norm")
    plt.title(f"Relative Gradient Flow: {exp_name}")
    plt.tight_layout()
    plt.savefig(f"{save_path}/{exp_name}_ratio.png", dpi=200)
    plt.close()

    min_grad = min(grad_norms)
    max_grad = max(grad_norms)

    print(f"\n{exp_name}")
    print(f"Min gradient: {min_grad:.2e}")
    print(f"Max gradient: {max_grad:.2e}")

    if min_grad < 1e-5:
        idx = grad_norms.index(min_grad)
        print(f"Possible vanishing gradient: {layers[idx]}")

    if max_grad > 1e+1:
        idx = grad_norms.index(max_grad)
        print(f"Possible exploding gradient: {layers[idx]}")


def visualize_first_activation(model, image, device, exp_name="", save_path=""):
    activation = {}

    def hook(module, inp, out):
        activation["feat"] = out.detach().cpu()

    relu_layer = None
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            relu_layer = m
            break

    handle = relu_layer.register_forward_hook(hook)
    model.eval()

    with torch.no_grad():
        model(image.unsqueeze(0).to(device))

    handle.remove()
    fmap = activation["feat"][0]
    n = min(16, fmap.shape[0])
    fig, axes = plt.subplots(4, 4, figsize=(8,8))

    for i in range(n):
        axes[i//4][i%4].imshow(fmap[i], cmap="viridis")
        axes[i//4][i%4].axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_path}/{exp_name}", dpi=200)
    plt.close()


def visualize_feature_maps(model, image, device, layer_index, exp_name="", save_path=""):
    activation = {}
    conv_layers = []

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            conv_layers.append(m)

    def hook(module, inp, out):
        activation["feat"] = out.detach().cpu()

    handle = conv_layers[layer_index].register_forward_hook(hook)
    model.eval()

    with torch.no_grad():
        model(image.unsqueeze(0).to(device))

    handle.remove()
    fmap = activation["feat"][0]
    n = min(16, fmap.shape[0])
    fig, axes = plt.subplots(4,4,figsize=(8,8))

    for i in range(n):
        axes[i//4][i%4].imshow(fmap[i], cmap="viridis")
        axes[i//4][i%4].axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_path}/{exp_name}", dpi=200)
    plt.close()