import random
from PIL import Image, ImageEnhance, ImageFilter
from torchvision.transforms import functional as F


class RandomGaussianBlur:
    """Случайное размытие изображения."""
    def __init__(self, p=0.5, radius=(1, 3)):
        self.p = p
        self.radius = radius

    def __call__(self, img):
        if random.random() > self.p:
            return img

        r = random.uniform(*self.radius)
        return img.filter(ImageFilter.GaussianBlur(r))


class RandomBrightnessContrast:
    """Случайное изменение яркости и контраста."""
    def __init__(self,p=0.5, brightness=(0.6, 1.4), contrast=(0.6, 1.4)):
        self.p = p
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        b = random.uniform(*self.brightness)
        c = random.uniform(*self.contrast)
        img = ImageEnhance.Brightness(img).enhance(b)
        img = ImageEnhance.Contrast(img).enhance(c)

        return img


class RandomPerspectiveCustom:
    """Случайное перспективное преобразование."""
    def __init__(self, p=0.5, distortion_scale=0.5):
        self.p = p
        self.distortion_scale = distortion_scale

    def __call__(self, img):
        if random.random() > self.p:
            return img

        return F.perspective(
            img,
            startpoints=[[0, 0], [img.width, 0], [img.width, img.height], [0, img.height]],
            endpoints=[
                [
                    random.randint(0, int(self.distortion_scale * img.width)),
                    random.randint(0, int(self.distortion_scale * img.height))
                ],
                [
                    img.width-random.randint(0, int(self.distortion_scale * img.width)),
                    random.randint(0, int(self.distortion_scale * img.height))
                ],
                [
                    img.width-random.randint(0, int(self.distortion_scale * img.width)),
                    img.height-random.randint(0, int(self.distortion_scale * img.height))
                ],
                [
                    random.randint(0, int(self.distortion_scale * img.width)),
                    img.height-random.randint(0, int(self.distortion_scale * img.height))
                ]
            ]
        )
    

class AugmentationPipeline:
    def __init__(self):
        self.augmentations = {}

    def add_augmentation(self, name, aug):
        self.augmentations[name] = aug

    def remove_augmentation(self, name):
        if name in self.augmentations:
            del self.augmentations[name]

    def apply(self, image):
        for aug in self.augmentations.values():
            image = aug(image)

        return image

    def get_augmentations(self):
        return list(self.augmentations.keys())