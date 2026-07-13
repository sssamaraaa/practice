import os
import matplotlib.pyplot as plt
import numpy as np
import time
import tracemalloc
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from PIL import Image
from utils import plot_training_history, visualize_standard_augmentations, get_one_image_per_class
from custom_augs import AugmentationPipeline, RandomGaussianBlur, RandomBrightnessContrast, RandomPerspectiveCustom
from extra_augs import AutoContrast, Solarize,Posterize
from datasets import CustomImageDataset


def task_one(save_dir, exp_name):
    save_path = os.path.join(save_dir, exp_name)
    os.makedirs(save_path, exist_ok=True)

    augmentation_dict = {
        "HorizontalFlip": transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor()
        ]),

        "RandomCrop": transforms.Compose([
            transforms.RandomCrop(180),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]),

        "ColorJitter": transforms.Compose([
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            ),
            transforms.ToTensor()
        ]),

        "RandomRotation": transforms.Compose([
            transforms.RandomRotation(30),
            transforms.ToTensor()
        ]),

        "RandomGrayscale": transforms.Compose([
            transforms.RandomGrayscale(p=1.0),
            transforms.ToTensor()
        ])
    }

    # Все аугментации вместе
    all_augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(180),
        transforms.Resize((224, 224)),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1
        ),
        transforms.RandomRotation(30),
        transforms.RandomGrayscale(p=0.3),
        transforms.ToTensor()
    ])

    visualize_standard_augmentations("data/train", augmentation_dict, all_augmentations, save_path=save_path)


def task_two(save_dir, exp_name):
    save_path = os.path.join(save_dir, exp_name)
    os.makedirs(save_path, exist_ok=True)

    custom_augs = {
        "GaussianBlur": transforms.Compose([
            RandomGaussianBlur(p=1),
            transforms.ToTensor()
        ]),

        "BrightnessContrast": transforms.Compose([
            RandomBrightnessContrast(p=1),
            transforms.ToTensor()
        ]),

        "Perspective": transforms.Compose([
            RandomPerspectiveCustom(p=1),
            transforms.ToTensor()
        ])
    }

    extra_augs = {
        "AutoContrast": transforms.Compose([
            transforms.ToTensor(),
            AutoContrast(p=1)
        ]),

        "Solarize": transforms.Compose([
            transforms.ToTensor(),
            Solarize()
        ]),

        "Posterize": transforms.Compose([
            transforms.ToTensor(),
            Posterize(bits=4)
        ])
    }

    images = get_one_image_per_class("data/train", max_images=5)

    for idx, (cls_name, img_path) in enumerate(images):
        image = Image.open(img_path).convert("RGB")
        image = image.resize((224, 224))
        fig, axes = plt.subplots(1, 7, figsize=(20,4))
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis("off")

        for i, (name, aug) in enumerate(custom_augs.items()):
            img = aug(image)
            img = img.permute(1,2,0).numpy()
            axes[i+1].imshow(img)
            axes[i+1].set_title(name)
            axes[i+1].axis("off")

        for j, (name, aug) in enumerate(extra_augs.items()):
            img = aug(image)
            img = img.permute(1,2,0).numpy()
            axes[j+4].imshow(img)
            axes[j+4].set_title(name)
            axes[j+4].axis("off")

        plt.suptitle(cls_name)
        plt.savefig(os.path.join(save_path, f"{cls_name}"), dpi=200,bbox_inches="tight")
        plt.close()


def task_three(save_dir, exp_name):
    save_path = os.path.join(save_dir, exp_name)
    os.makedirs(save_path, exist_ok=True)

    dataset = CustomImageDataset("data/train")
    class_counts = {cls: 0 for cls in dataset.classes}
    widths = []
    heights = []

    for img_path, label in zip(dataset.images, dataset.labels):
        class_name = dataset.classes[label]
        class_counts[class_name] += 1
        with Image.open(img_path) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)

    widths = np.array(widths)
    heights = np.array(heights)

    print("Количество изображений по классам")

    for cls, count in class_counts.items():
        print(f"{cls:20s}: {count}")

    print("\nРазмеры изображений")
    print(f"Минимальная ширина : {widths.min()}")
    print(f"Максимальная ширина: {widths.max()}")
    print(f"Средняя ширина     : {widths.mean():.2f}")
    print(f"Минимальная высота : {heights.min()}")
    print(f"Максимальная высота: {heights.max()}")
    print(f"Средняя высота     : {heights.mean():.2f}")


    plt.figure(figsize=(10, 5))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title("Количество изображений по классам")
    plt.xlabel("Класс")
    plt.ylabel("Количество")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "class_distribution"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(widths, heights, alpha=0.6)
    plt.xlabel("Ширина")
    plt.ylabel("Высота")
    plt.title("Распределение размеров изображений")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "image_sizes"), dpi=200)
    plt.close()


def task_four(save_dir, exp_name):
    save_path = os.path.join(save_dir, exp_name)
    os.makedirs(save_path, exist_ok=True)

    # Light
    light = AugmentationPipeline()
    light.add_augmentation("Flip", transforms.RandomHorizontalFlip(p=0.5))
    light.add_augmentation("ToTensor", transforms.ToTensor())

    # Medium
    medium = AugmentationPipeline()
    medium.add_augmentation("Flip", transforms.RandomHorizontalFlip())
    medium.add_augmentation("Rotation", transforms.RandomRotation(20))
    medium.add_augmentation("Color", transforms.ColorJitter(brightness=0.3, contrast=0.3))
    medium.add_augmentation("ToTensor", transforms.ToTensor())

    # Heavy
    heavy = AugmentationPipeline()
    heavy.add_augmentation("Crop", transforms.RandomResizedCrop(224, scale=(0.6, 1.0)))
    heavy.add_augmentation("Flip", transforms.RandomHorizontalFlip())
    heavy.add_augmentation("Rotation", transforms.RandomRotation(35))
    heavy.add_augmentation("Color", transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2))
    heavy.add_augmentation("Gray", transforms.RandomGrayscale(0.3))
    heavy.add_augmentation("ToTensor", transforms.ToTensor())

    configs = {"light": light, "medium": medium, "heavy": heavy}
    images = get_one_image_per_class("data/train", max_images=5)

    for cfg_name, pipeline in configs.items():
        cfg_folder = os.path.join(save_path, cfg_name)
        os.makedirs(cfg_folder, exist_ok=True)
        print(f"\nКонфигурация: {cfg_name}")
        print("Аугментации:", pipeline.get_augmentations())

        for class_name, img_path in images:
            image = Image.open(img_path).convert("RGB")
            image = image.resize((224, 224))
            aug = pipeline.apply(image)
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(image)
            axes[0].set_title("Original")
            axes[0].axis("off")
            axes[1].imshow(aug.permute(1, 2, 0).numpy().clip(0, 1))
            axes[1].set_title(cfg_name)
            axes[1].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join( cfg_folder,f"{class_name}"), dpi=200)
            plt.close()


def task_five(save_dir, exp_name):
    save_path = os.path.join(save_dir, exp_name)
    os.makedirs(save_path, exist_ok=True)

    image_sizes = [(64, 64), (128, 128), (224, 224), (512, 512)]

    augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3
        ),
        transforms.ToTensor()
    ])

    times = []
    memories = []

    for size in image_sizes:
        dataset = CustomImageDataset(
            "data/train",
            transform=augmentation,
            target_size=size
        )

        num_images = min(100, len(dataset))
        tracemalloc.start()
        start = time.perf_counter()

        for i in range(num_images):
            _ = dataset[i]

        elapsed = time.perf_counter() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append(elapsed)
        memories.append(peak / (1024 ** 2))

        print(
            f"{size}: "
            f"time={elapsed:.3f}s, "
            f"memory={peak/(1024**2):.2f} MB"
        )

    # График времени
    labels = [f"{w}x{h}" for w, h in image_sizes]
    plt.figure(figsize=(8, 5))
    plt.plot(labels, times, marker="o")
    plt.title("Время обработки 100 изображений")
    plt.xlabel("Размер изображения")
    plt.ylabel("Время (сек.)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "time"), dpi=200)
    plt.close()

    # График памяти
    plt.figure(figsize=(8, 5))
    plt.plot(labels, memories, marker="o")
    plt.title("Пиковое использование памяти")
    plt.xlabel("Размер изображения")
    plt.ylabel("Память (MB)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "memory"), dpi=200)
    plt.close()


def task_six(save_dir, exp_name):
    save_path = os.path.join(save_dir, exp_name)
    os.makedirs(save_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = CustomImageDataset("data/train", transform=train_transform)
    val_dataset = CustomImageDataset("data/test", transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.get_class_names()))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 10

    history = {"train_losses": [], "val_losses": [], "train_accs": [],"val_accs": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss /= len(train_loader)
        train_acc = correct / total
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total
        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["train_accs"].append(train_acc)
        history["val_accs"].append(val_acc)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    torch.save(model.state_dict(), os.path.join(save_path, "resnet18.pth"))
    plot_training_history(history, exp_name="training_history.png", save_path=save_path)

    with open(os.path.join(save_path, "metrics.txt"), "w") as f:
        f.write(f"Train Accuracy: {history['train_accs'][-1]:.4f}\n")
        f.write(f"Validation Accuracy: {history['val_accs'][-1]:.4f}\n")
        f.write(f"Train Loss: {history['train_losses'][-1]:.4f}\n")
        f.write(f"Validation Loss: {history['val_losses'][-1]:.4f}\n")


if __name__ == "__main__":
    save_dir = "lesson5/results/"
    task_six(save_dir, "task6/")