import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torchvision
from matplotlib.pyplot import imshow
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.models import ViT_B_16_Weights
from tqdm import tqdm

from constant import (
    EMBEDDING_CACHE,
    LUNG_IMAGES_DIR,
    LUNG_LOADED_FILE,
    LUNG_METADATA_FILE,
    LUNG_PREPROCESS_DIR,
)
from dataset import load_dataset
from extract_feature import extract_embeddings
from model import ClassifierHead


def train(
    batch_size=8,
    lr=1e-5,
    epochs=200,
    random_state=42,
    csv_file=LUNG_METADATA_FILE,
    root_dir=LUNG_IMAGES_DIR,
    load_file=LUNG_LOADED_FILE,
    embedding_cache=EMBEDDING_CACHE,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=> Using device: {device}")

    torch.manual_seed(random_state)

    # This guarantees the loaded dataset is built before reading
    train_idx, valid_idx, test_idx, label_map = load_dataset(
        csv_file, root_dir, load_file
    )

    df = pd.read_csv(load_file, encoding="utf-8")

    # Normalization parameters based on the ViT
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # Deterministic transforms for test set
    det_transform = transforms.Compose(
        [
            transforms.Resize(size=(224, 224), antialias=True),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Random transforms for train and validation set
    rand_transform = transforms.Compose(
        [
            transforms.Resize(size=(224, 224), antialias=True),
            transforms.RandomCrop(size=224, padding=16),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ]
    )

    if os.path.exists(embedding_cache):
        print(f"=> Loading cached embeddings from {embedding_cache}")
        cache = torch.load(embedding_cache, map_location="cpu")
        train_X, train_y = cache["train"]
        valid_X, valid_y = cache["valid"]
        test_X, test_y = cache["test"]
    else:
        print("=> Building ViT backbone model for embedding extraction")
        backbone = torchvision.models.vit_b_16(
            weights=ViT_B_16_Weights.DEFAULT, image_size=224
        )

        # Removing classfication layer as we don't need the logits
        backbone.heads = nn.Identity()
        backbone.eval().to(device)

        train_X, train_y = extract_embeddings(
            df, train_idx, rand_transform, backbone, device, "train"
        )
        valid_X, valid_y = extract_embeddings(
            df, valid_idx, rand_transform, backbone, device, "valid"
        )
        test_X, test_y = extract_embeddings(
            df, test_idx, det_transform, backbone, device, "test"
        )

        torch.save(
            {
                "train": (train_X, train_y),
                "valid": (valid_X, valid_y),
                "test": (test_X, test_y),
            },
            embedding_cache,
        )
        print(f"=> Embeddings cached to {embedding_cache}")

        # Free backbone memory before training
        del backbone
        if device == "cuda":
            torch.cuda.empty_cache()

    def make_loader(X, y, shuffle, drop_last=True):
        ds = TensorDataset(X.to(device), y.to(device))
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
        )

    train_loader = make_loader(train_X, train_y, True)
    valid_loader = make_loader(valid_X, valid_y, False)
    test_loader = make_loader(test_X, test_y, False)

    embed_dim = train_X.shape[1]  # 768 for ViT-B/16
    model = ClassifierHead(embed_dim, num_classes=len(label_map)).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    loss_epochs, acc_epochs = [], []

    for e in tqdm(range(epochs), desc="> Training~ "):
        model.train()

        accuracy_batch = loss_batch = 0

        for image_batch, label_batch in train_loader:
            logits = model(image_batch)
            loss = loss_fn(logits, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_batch += loss.item()
            accuracy_batch += (logits.argmax(1) == label_batch).sum().item()

        loss_epochs.append(loss_batch / len(train_loader))
        acc_epochs.append(accuracy_batch / len(train_loader.dataset))

        # Model validation
        model.eval()
        val_loss = val_acc = 0
        with torch.inference_mode():
            for image_batch, label_batch in valid_loader:
                logits = model(image_batch)
                val_loss += loss_fn(logits, label_batch).item()
                val_acc += (logits.argmax(1) == label_batch).sum().item()

        val_loss /= len(valid_loader)
        val_acc /= len(valid_loader.dataset)

        print(
            f"Epochs {e + 1:03d} | "
            f"Train loss : {loss_epochs[-1]:.2f} | Train acc {acc_epochs[-1] * 100:.2f} % | "
            f"Val loss {val_loss:.4f} | Val acc {val_acc * 100:.2f}%"
        )

        if e % (100 - 1) == 0:
            state = {
                "epoch": e + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss_history": loss_epochs,
                "acc_history": acc_epochs,
                "label_map": label_map,
            }
            torch.save(
                state,
                Path(
                    *[
                        LUNG_PREPROCESS_DIR,
                        f"LUNG_{model.__class__.__name__}_{e + 1}_EPOCHS.pth.tar",
                    ]
                ),
            )

    """
    # Final model evaluation
    model.eval()
    test_acc = 0
    with torch.inference_mode():
        for feat_batch, label_batch in test_loader:
            logits = model(feat_batch)
            test_acc += (logits.argmax(1) == label_batch).sum().item()
    test_acc /= len(test_loader.dataset)
    print(f"\nTest accuracy: {test_acc * 100:.2f}%")
    """


def main():
    train()


if __name__ == "__main__":
    main()
