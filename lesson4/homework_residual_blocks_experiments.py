import time
import torch
from base.trainer import train_model
from base.datasets import get_cifar_loaders
from lesson4.utils.experiment_utils import count_parameters
from lesson4.utils.vizualization_utils import plot_training_history
from lesson4.models.custom_residual import ResidualCNN, BasicResidualBlock, BottleneckResidualBlock, WideResidualBlock


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = get_cifar_loaders(batch_size=64)
    models = {"Basic": ResidualCNN(BasicResidualBlock), "Bottleneck": ResidualCNN(BottleneckResidualBlock), "Wide": ResidualCNN(WideResidualBlock)}
    results = {}

    for name, model in models.items():
        print(name)
        model = model.to(device)
        params = count_parameters(model)
        start = time.time()
        train_losses, train_accs, test_losses, test_accs = train_model(model, train_loader, test_loader, epochs=10, lr=0.001, device=device)
        train_time = time.time() - start
        history = {"train_losses": train_losses, "train_accs": train_accs, "test_losses": test_losses, "test_accs": test_accs,}
        plot_training_history( history, exp_name=f"{name}.png", save_path="lesson4/plots3/")

        results[name] = {
            "parameters": params,
            "time": train_time,
            "train_acc": train_accs[-1],
            "test_acc": test_accs[-1],
            "train_loss": train_losses[-1],
            "test_loss": test_losses[-1],
        }


    print("\n")
    print("Residual Block Comparison")

    print(f"{'Model':<15}"
        f"{'Params':>12}"
        f"{'Time(s)':>12}"
        f"{'Train Acc':>12}"
        f"{'Test Acc':>12}"
    )

    for name, r in results.items():
        print(
            f"{name:<15}"
            f"{r['parameters']:>12}"
            f"{r['time']:>12.2f}"
            f"{r['train_acc']:>12.3f}"
            f"{r['test_acc']:>12.3f}"
        )