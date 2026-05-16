import os

import pandas as pd
import torch
import torch.nn as nn
import torchvision
from matplotlib.pyplot import imshow
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ViT_B_16_Weights
from tqdm import tqdm

from constant import LUNG_IMAGES_DIR, LUNG_LOADED_FILE, LUNG_METADATA_FILE
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


def train(
    batch_size=8,
    lr=1e-1,
    epochs=200,
    random_state=42,
    csv_file=LUNG_METADATA_FILE,
    root_dir=LUNG_IMAGES_DIR,
    load_file=LUNG_LOADED_FILE,
):
    print("test")
    # This guarantees the loaded dataset is built before reading
    train_idx, valid_idx, test_idx, label_map = load_dataset(
        csv_file, root_dir, load_file
    )

    df = pd.read_csv(load_file, encoding="utf-8")

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
            batch_size,
            shuffle=True,
            drop_last=True,
        ),
        DataLoader(
            valid_set,
            batch_size,
            shuffle=False,
            drop_last=True,
        ),
        DataLoader(
            test_set,
            batch_size,
            shuffle=False,
            drop_last=True,
        ),
    )

    torch.manual_seed = random_state

    model = torchvision.models.vit_b_16(
        weights=ViT_B_16_Weights.DEFAULT, image_size=224
    )
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    loss_epochs = []
    accuracy_epochs = []

    current_run = 0

    for e in tqdm(range(epochs), desc="> Training~ "):
        model.train()

        accuracy_batch = loss_batch = 0

        for image_batch, label_batch in train_loader:
            logits = model.forward(image_batch)
            loss = loss_fn(logits, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_batch += loss.item()
            accuracy_batch += (torch.argmax(logits, axis=1) == label_batch).sum().item()

        loss_epochs.append(loss_batch / len(train_loader))
        accuracy_epochs.append(accuracy_batch / len(train_loader.dataset))

        print(
            f"Epochs {e + 1:03d} | Train loss : {loss_epochs[-1]:.2f} | "
            f"Train accuracy {accuracy_epochs[-1] * 100:.2f} % | "
        )

        if e % (100 - 1) == 0:
            state = {
                "epoch": e + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss_history": loss_epochs,
                "acc_history": accuracy_epochs,
            }
            torch.save(state, f"LUNG_VIT_B_16_{e + 1}_EPOCHS.pth.tar")

        current_run += 1


def main():
    train()


if __name__ == "__main__":
    main()
