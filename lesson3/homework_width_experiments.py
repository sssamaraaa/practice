import torch
import numpy as np
import pandas as pd
import itertools
import time

from model import FullyConnectedModel
from lesson3.trainer import train_model
from datasets import get_cifar_loaders, get_mnist_loaders
from utils.visualization_utils import plot_heatmap
from utils.experiment_utils import build_architecture, build_config


def grid_search(train_loader, test_loader, device, epochs=5, lr=0.001):
    kinds = ["constant", "expanding", "shrinking"]
    bases = [64, 256, 1024, 2048] # ширина первого скрытого слоя
    results = []

    for kind, base in itertools.product(kinds, bases):
        print(f"\nTraining: {kind}, base={base}")
        hidden = build_architecture(kind, base)
        config = build_config(hidden)
        model = FullyConnectedModel(
            input_size=784,
            num_classes=10,
            **config
        ).to(device)

        start = time.time()

        train_losses, train_accs, test_losses, test_accs = train_model(
            model,
            train_loader,
            test_loader,
            epochs=epochs,
            lr=lr,
            device=device
        )

        end = time.time()
        best_acc = max(test_accs)
        results.append({
            "kind": kind,
            "base": base,
            "accuracy": best_acc,
            "time_sec": end - start
        })

        print(f"Accuracy={best_acc:.4f} ; time={end-start:.1f}s")

    return pd.DataFrame(results)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)
    exp_name = "grid_search_mnist"
    save_path = "lesson3/results/width_experiments/plots_mnist"

    train_loader, test_loader = get_mnist_loaders(batch_size=64)

    df = grid_search(
        train_loader,
        test_loader,
        device=device,
        epochs=5,   
        lr=0.001
    )

    print(df)
    plot_heatmap(df, exp_name=exp_name, save_path=save_path)
    df.to_csv(f"{save_path}/grid_search_mnist_results.csv", index=False)