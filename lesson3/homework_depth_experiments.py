import torch
import time
import numpy as np
from lesson3.model import FullyConnectedModel
from lesson3.utils.visualization_utils import plot_training_curve, plot_weights_distributions
from base.trainer import train_model
from base.datasets import get_cifar_loaders, get_mnist_loaders


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(43)
    torch.manual_seed(43)
    epochs = 10
    batch_size = 64
    lr=0.001
    exp_name = "4_layers"
    exp_num = "8"

    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)

    model = FullyConnectedModel(input_size=784, num_classes=10, config_path=f"lesson3/model_configs/{exp_name}.json").to(device)

    start_time = time.time()
    train_losses, train_accs, test_losses, test_accs = train_model(model, train_loader, test_loader, epochs=epochs, lr=lr, device=device)
    end_time = time.time()
    plot_training_curve(epochs, train_losses, test_losses, metric_type="Loss", exp_name=f"{exp_name}_{exp_num}_loss", save_path="lesson3/results/regularization_experiments/plots_mnist/losses")
    plot_training_curve(epochs, train_accs, test_accs, metric_type="Accuracy", exp_name=f"{exp_name}_{exp_num}_accuracy", save_path="lesson3/results/regularization_experiments/plots_mnist/accs")
    plot_weights_distributions(model, exp_name=f"{exp_name}_{exp_num}_weights_distributions", save_path="lesson3/results/regularization_experiments/plots_mnist/weights_distributions")
    print(f"TRAIN LOSS: {min(train_losses)}, TRAIN ACC: {max(train_accs)}, TEST LOSS: {min(test_losses)}, TEST ACC: {max(test_accs)}", sep='\n')
    print(end_time - start_time)