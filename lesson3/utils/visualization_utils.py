import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from math import ceil


def plot_training_curve(epoches, train, test, metric_type="Loss", exp_name="", save_path=""):
    plt.figure(figsize=(10, 5))
    plt.plot(range(epoches), train, label=f"Training {metric_type}")
    plt.plot(range(epoches), test, label=f"Validation {metric_type}")
    plt.xlabel("Epochs")
    plt.ylabel(metric_type)
    plt.title(f"Training and Validation {metric_type} Curves")
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_path}/{exp_name}", dpi=200, bbox_inches="tight")
    plt.close()

def plot_heatmap(df, exp_name="", save_path=""):
    pivot = df.pivot(index="kind", columns="base", values="accuracy")
    plt.figure(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".4f")
    plt.title("Grid Search: Accuracy by Architecture")
    plt.xlabel("Base width")
    plt.ylabel("Architecture type")
    plt.savefig(f"{save_path}/{exp_name}", dpi=200, bbox_inches="tight")
    plt.close()

def plot_weights_distributions(model, exp_name="", save_path=""):
    linear_layers = [l for l in model.layers if isinstance(l, nn.Linear)]

    n = len(linear_layers)
    cols = 2
    rows = ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows))
    axes = axes.flatten()

    for i, layer in enumerate(linear_layers):
        weights = layer.weight.detach().cpu().numpy().ravel()

        axes[i].hist(weights, bins=50)
        axes[i].set_title(
            f"{layer.in_features} → {layer.out_features}"
        )
        axes[i].set_xlabel("Weight")
        axes[i].set_ylabel("Count")
        axes[i].grid(True)

    for i in range(n, len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle("Weight distributions", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{exp_name}", dpi=200)
    plt.close(fig)