import os
import time
import torch
import numpy as np
from base.datasets import get_cifar_loaders
from base.trainer import train_model
from lesson4.models.cnn import CNNWithResidual, DepthCNN, KernelCNN, KernelMixCNN
from lesson4.utils.experiment_utils import count_parameters, inference_time, compute_receptive_field
from lesson4.utils.vizualization_utils import plot_training_history, visualize_feature_maps, visualize_first_activation, gradient_flow


def train_and_analyze_architecture(name, model, train_loader, test_loader, device, epochs, lr):
    save_dir = f"lesson4/plots2/{name.replace(' ','_')}"
    os.makedirs(save_dir, exist_ok=True)

    print(name)
    print("Parameters:", count_parameters(model))

    model.to(device)
    start = time.time()
    history = train_model( model, train_loader, test_loader, epochs, lr, device)
    train_time = time.time()-start
    inference = inference_time(model, test_loader, device)

    print(f"Training: {train_time:.2f}s")
    print(f"Inference: {inference:.2f}s")

    history = {"train_losses": history[0], "train_accs": history[1], "test_losses": history[2], "test_accs": history[3]} 
    best_epoch = np.argmax(history["test_accs"])
    best_test_acc = history["test_accs"][best_epoch]
    train_acc_at_best = history["train_accs"][best_epoch]

    print(f"Best epoch: {best_epoch + 1}")
    print(f"Train accuracy: {train_acc_at_best:.4f}")
    print(f"Test accuracy:  {best_test_acc:.4f}")
    print(f"Gap: {train_acc_at_best - best_test_acc:.4f}")
    plot_training_history(history, exp_name="history", save_path=save_dir)

    image, _ = next(iter(test_loader))
    visualize_first_activation(model, image=image[0], device=device, exp_name="first_layer", save_path=save_dir)
    visualize_feature_maps( model, image=image[0], device=device, layer_index=-1, exp_name="feature_map", save_path=save_dir)
    gradient_flow(model, exp_name="grad", save_path=save_dir)

    rf = compute_receptive_field(model)
    print(f"Receptive field: {rf} x {rf}")
    return model, history


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(43)
    np.random.seed(43)
    epochs = 10
    lr = 0.001
    batch_size = 64

    train_loader, test_loader = get_cifar_loaders(batch_size=batch_size)

    # 2.1 Влияние размера ядра свертки

    print("\nKernel Size Experiments")

    kernel_models = [
        ("Kernel_3x3", KernelCNN(kernel_size=3, channels=(40, 64))),
        ("Kernel_5x5", KernelCNN(kernel_size=5, channels=(25, 62))),
        ("Kernel_7x7", KernelCNN(kernel_size=7, channels=(38, 54))),
        ("Kernel_1x1_3x3", KernelMixCNN()),
    ]

    kernel_histories = {}
    
    for name, model in kernel_models:
        trained_model, history = train_and_analyze_architecture(
            name=name,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=epochs,
            lr=lr
        )

        kernel_histories[name] = history

    kernel_histories = {}

    # 2.2 Влияние глубины CNN

    print("\nCNN Depth Experiments")

    depth_models = [("Depth_2", DepthCNN(depth=2)), ("Depth_4", DepthCNN(depth=4)), ("Depth_6", DepthCNN(depth=6)), ("Residual_CNN", CNNWithResidual(input_channels=3))]
    depth_histories = {}

    for name, model in depth_models:
        trained_model, history = train_and_analyze_architecture(
            name=name,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=epochs,
            lr=lr
        )

        depth_histories[name] = history

    print("\nKernel size experiments:")

    for name, history in kernel_histories.items():
        best_acc = max(history["test_accs"])
        print(f"{name:20s} : {best_acc:.4f}")

    print("\nDepth experiments:")

    for name, history in depth_histories.items():
        best_acc = max(history["test_accs"])
        print(f"{name:20s} : {best_acc:.4f}")