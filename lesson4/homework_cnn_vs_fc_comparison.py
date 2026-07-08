import torch
import numpy as np
import time
import os
from base.datasets import get_cifar_loaders, get_mnist_loaders
from base.trainer import train_model
from lesson4.models.cnn import CIFARCNN, CNNWithResidual, RegularizedResidualCNN, SimpleCNN
from lesson4.utils.experiment_utils import count_parameters, inference_time
from lesson4.utils.vizualization_utils import gradient_flow, plot_confusion, plot_training_history
from lesson4.models.fc import FCModel, DeepFCModel


def train_and_analyze(name, model, train_loader, test_loader, device, epochs, lr):
    save_dir = f"lesson4/plots1/{name.replace(' ', '_')}"
    os.makedirs(save_dir, exist_ok=True)

    print(name)
    print("Parameters:", count_parameters(model))

    model.to(device)
    start = time.time()
    history = train_model( model, train_loader, test_loader, epochs=epochs, lr=lr, device=device)
    history = {"train_losses": history[0], "train_accs": history[1], "test_losses": history[2], "test_accs": history[3]}
    train_time = time.time() - start
    infer = inference_time(model, test_loader, device)

    print(f"Training time: {train_time:.2f} sec")
    print(f"Inference time: {infer:.2f} sec")

    best_epoch = np.argmax(history["test_accs"])
    best_test_acc = history["test_accs"][best_epoch]
    train_acc_at_best = history["train_accs"][best_epoch]

    print(f"Best epoch: {best_epoch + 1}")
    print(f"Train accuracy: {train_acc_at_best:.4f}")
    print(f"Test accuracy:  {best_test_acc:.4f}")
    print(f"Gap: {train_acc_at_best - best_test_acc:.4f}")
    
    plot_training_history(history, exp_name="history", save_path=save_dir)
    gradient_flow(model, exp_name="gradient_flow", save_path=save_dir)
    plot_confusion(model, test_loader, device, exp_name="confusion_matrix", save_path=save_dir)
    return model, history


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(43)
    np.random.seed(43)
    epochs = 10
    lr = 0.001
    batch_size = 64

    # MNIST

    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)

    fc_model = FCModel()
    cnn_model = SimpleCNN()
    res_model = CNNWithResidual()

    fc_model, _ = train_and_analyze("FC", fc_model, train_loader, test_loader, device, epochs, lr)
    cnn_model, _ = train_and_analyze("Simple CNN", cnn_model, train_loader, test_loader, device, epochs, lr)
    res_model, _ = train_and_analyze("CNN With Residual", res_model, train_loader, test_loader, device, epochs, lr)

    # CIFAR10

    train_loader, test_loader = get_cifar_loaders(batch_size=batch_size)

    fc_model = DeepFCModel() 
    cnn_model = CNNWithResidual(input_channels=3)
    reg_model = RegularizedResidualCNN()

    fc_model, _ = train_and_analyze("Deep FC", fc_model, train_loader, test_loader, device, epochs, lr) 
    cnn_model, _ = train_and_analyze("CNN With Residual CIFAR", cnn_model, train_loader, test_loader, device, epochs, lr)
    reg_model, _ = train_and_analyze("Regularized Residual CNN", reg_model, train_loader, test_loader, device, epochs, lr) 