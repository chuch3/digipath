import os

import pandas as pd
import torch
import torch.nn as nn
import torchvision
from matplotlib.pyplot import imshow
from PIL import Image
from rich.progress import track
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ViT_B_16_Weights

from config import LUNG_LOADED_FILE
from dataset import load_dataset


class LungImageLoaderDataset(Dataset):
    """
    PIL Image loader of the LungImage700 dataset after dataset split.

    NOTE: This dataset loader is not generic and has to be modified for preference
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        loader=lambda p: Image.open(p, "r").convert("RGB"),
    ) -> None:
        self._df = df
        self._loader = loader
        self._transform = transform

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self._df.iloc[idx]
        image = self._loader(row["path"])

        if self._transform:
            image = self._transform(image)

        return (image, row["label"])


# TODO:
# - [ ] Masks by Grad-CAM algorithm on last convolution layer


def train(batch_size=32, lr=1e-1, epochs=200):
    # This guarantees the loaded dataset is built before reading
    train_idx, valid_idx, test_idx, label_map = load_dataset()

    df = pd.read_csv(LUNG_LOADED_FILE, encoding="utf-8")

    train_valid_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=(224, 224), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=(224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_set, valid_set, test_set = (
        LungImageLoaderDataset(df.iloc[train_idx], train_valid_transform),
        LungImageLoaderDataset(df.iloc[valid_idx], train_valid_transform),
        LungImageLoaderDataset(df.iloc[test_idx], test_transform),
    )

    train_loader, valid_loader, test_loader = (
        DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        ),
        DataLoader(
            valid_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
        ),
        DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
        ),
    )

    model = torchvision.models.vit_b_16(
        weights=ViT_B_16_Weights.DEFAULT, image_size=224
    )
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in track(range(epochs)):
        model.train()

        epoch_loss = 0
        correct = 0
        total = 0

        for image, label in train_loader:
            optimizer.zero_grad()
            logits = model(image)
            loss = loss_fn(logits, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)

        acc = correct / total
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {acc:.4f}")


def main():
    train()


if __name__ == "__main__":
    main()
