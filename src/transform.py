import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


def clahe_transform(pil_img: Image.Image):
    img = np.array(pil_img)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


# Normalization parameters based on the ViT-B-16 model
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


# Deterministic transforms for test set
det_transform = transforms.Compose(
    [
        transforms.Resize(size=(224, 224), antialias=True),
        transforms.Lambda(clahe_transform),  # Enhancing contrast of stains
        transforms.ToTensor(),
        normalize,
    ]
)


# Random transforms for train and validation set
rand_transform = transforms.Compose(
    [
        transforms.Resize(size=(224, 224), antialias=True),
        transforms.Lambda(clahe_transform),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor(),
        normalize,
    ]
)

ssl_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.2, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ]
)
