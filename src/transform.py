import cv2
import numpy as np
import torch
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


# Deterministic transforms for test and validation set
det_transform = transforms.Compose(
    [
        transforms.Resize(size=(256, 256), antialias=True),
        transforms.CenterCrop(224),
        # transforms.Lambda(clahe_transform),
        transforms.ToTensor(),
        normalize,
    ]
)


# Random transforms for train
rand_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        # transforms.Lambda(clahe_transform),
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
        transforms.ToTensor(),
        normalize,
    ]
)
