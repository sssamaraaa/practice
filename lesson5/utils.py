import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from torchvision import transforms
from PIL import Image


def show_images(images, labels=None, nrow=8, title=None, size=128):
    """Визуализирует батч изображений."""
    images = images[:nrow]
    
    # Увеличиваем изображения до 128x128 для лучшей видимости
    resize_transform = transforms.Resize((size, size), antialias=True)
    images_resized = [resize_transform(img) for img in images]
    
    # Создаем сетку изображений
    fig, axes = plt.subplots(1, nrow, figsize=(nrow*2, 2))
    if nrow == 1:
        axes = [axes]
    
    for i, img in enumerate(images_resized):
        img_np = img.numpy().transpose(1, 2, 0)
        # Нормализуем для отображения
        img_np = np.clip(img_np, 0, 1)
        axes[i].imshow(img_np)
        axes[i].axis('off')
        if labels is not None:
            axes[i].set_title(f'Label: {labels[i]}')
    
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def show_single_augmentation(original_img, augmented_img, title="Аугментация"):
    """Визуализирует оригинальное и аугментированное изображение рядом."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    # Проверяем, является ли изображение PIL Image, и преобразуем в тензор
    if isinstance(original_img, Image.Image):
        transform_to_tensor = transforms.ToTensor()
        original_tensor = transform_to_tensor(original_img)
    else:
        original_tensor = original_img
    
    if isinstance(augmented_img, Image.Image):
        transform_to_tensor = transforms.ToTensor()
        augmented_tensor = transform_to_tensor(augmented_img)
    else:
        augmented_tensor = augmented_img
    
    # Увеличиваем изображения
    resize_transform = transforms.Resize((128, 128), antialias=True)
    orig_resized = resize_transform(original_tensor)
    aug_resized = resize_transform(augmented_tensor)
    
    # Оригинальное изображение
    orig_np = orig_resized.numpy().transpose(1, 2, 0)
    orig_np = np.clip(orig_np, 0, 1)
    ax1.imshow(orig_np)
    ax1.set_title("Оригинал")
    ax1.axis('off')
    
    # Аугментированное изображение
    aug_np = aug_resized.numpy().transpose(1, 2, 0)
    aug_np = np.clip(aug_np, 0, 1)
    ax2.imshow(aug_np)
    ax2.set_title(title)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

def show_multiple_augmentations(original_img, augmented_imgs, titles):
    """Визуализирует оригинальное изображение и несколько аугментаций."""
    n_augs = len(augmented_imgs)
    fig, axes = plt.subplots(1, n_augs + 1, figsize=((n_augs + 1) * 2, 2))
    
    # PIL Image -> тензор
    if isinstance(original_img, Image.Image):
        transform_to_tensor = transforms.ToTensor()
        original_tensor = transform_to_tensor(original_img)
    else:
        original_tensor = original_img
    
    # Увеличиваем изображения
    resize_transform = transforms.Resize((128, 128), antialias=True)
    orig_resized = resize_transform(original_tensor)
    
    # Оригинальное изображение
    orig_np = orig_resized.numpy().transpose(1, 2, 0)
    orig_np = np.clip(orig_np, 0, 1)
    axes[0].imshow(orig_np)
    axes[0].set_title("Оригинал")
    axes[0].axis('off')
    
    for i, (aug_img, title) in enumerate(zip(augmented_imgs, titles)):
        if isinstance(aug_img, Image.Image):
            aug_tensor = transform_to_tensor(aug_img)
        else:
            aug_tensor = aug_img
            
        aug_resized = resize_transform(aug_tensor)
        aug_np = aug_resized.numpy().transpose(1, 2, 0)
        aug_np = np.clip(aug_np, 0, 1)
        axes[i + 1].imshow(aug_np)
        axes[i + 1].set_title(title)
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.show()


def get_one_image_per_class(train_dir, max_images=5):
    """
    Берет по одному изображению из разных классов.
    """
    images = []
    classes = sorted([cls for cls in os.listdir(train_dir)
                      if os.path.isdir(os.path.join(train_dir, cls))])

    for cls in classes:
        class_dir = os.path.join(train_dir, cls)
        files = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]

        if len(files) == 0:
            continue

        img_path = os.path.join(class_dir, files[0])
        images.append((cls, img_path))

        if len(images) == max_images:
            break

    return images


def visualize_standard_augmentations(train_dir, augmentation_dict, all_augmentations, save_path=""):
    selected_images = get_one_image_per_class(train_dir, max_images=5)

    for idx, (class_name, img_path) in enumerate(selected_images):
        image = Image.open(img_path).convert("RGB")
        image = image.resize((224, 224))
        transformed_images = []
        titles = []

        # Каждая аугментация отдельно
        for aug_name, aug in augmentation_dict.items():
            transformed_images.append(aug(image))
            titles.append(aug_name)

        # Все вместе
        transformed_images.append(all_augmentations(image))
        titles.append("All")
        show_multiple_augmentations(image, transformed_images, titles)

        fig, axes = plt.subplots(1, len(transformed_images) + 1, figsize=(18, 3))
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis("off")

        for i, (img, title) in enumerate(zip(transformed_images, titles)):
            img_np = img.permute(1, 2, 0).numpy()
            img_np = img_np.clip(0, 1)
            axes[i + 1].imshow(img_np)
            axes[i + 1].set_title(title)
            axes[i + 1].axis("off")

        plt.suptitle(f"Class: {class_name}")
        plt.savefig(f"{save_path}/{class_name}", dpi=200, bbox_inches="tight")
        plt.close()


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