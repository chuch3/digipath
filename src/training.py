import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from matplotlib.pyplot import imshow
from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.models import ViT_B_16_Weights
from tqdm import tqdm

from constant import (
    LUNG_IMAGES_DIR,
    LUNG_LOADED_FILE,
    LUNG_METADATA_FILE,
    LUNG_MODEL_DIR,
    LUNG_PREPROCESS_DIR,
    SSL_CHECKPOINT,
)
from dataset import load_dataset
from extract import extract_embeddings
from macenko import build_macenko_normalizer
from model import ClassifierHead
from ssl_train import pretrain_ssl
from transform import det_transform, rand_transform


def train(
    batch_size=8,
    lr=1e-5,
    epochs=200,
    ssl_epochs=10,
    csv_file=LUNG_METADATA_FILE,
    root_dir=LUNG_IMAGES_DIR,
    load_file=LUNG_LOADED_FILE,
    embedding_cache_dir=LUNG_PREPROCESS_DIR,
    device=None,
    use_stain_norm=True,
    use_ssl=True,
    random_state=42,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=> Using device : {device}")

    torch.manual_seed(random_state)

    # This guarantees the loaded dataset is built before reading
    import pickle

    train_idx, valid_idx, test_idx, label_map = load_dataset(
        csv_file, root_dir, load_file
    )

    df = pd.read_csv(load_file, encoding="utf-8")

    stain_suffix = "stain" if use_stain_norm else "raw"
    ssl_suffix = "ssl" if use_ssl else "labeled"
    embedding_cache = Path(
        *[embedding_cache_dir, f"lung_embedded_{stain_suffix}_{ssl_suffix}.pt"]
    )

    if os.path.exists(embedding_cache):
        print(f"=> Loading cached embeddings from {embedding_cache}")
        cache = torch.load(embedding_cache, map_location="cpu")
        train_X, train_y = cache["train"]
        valid_X, valid_y = cache["valid"]
        test_X, test_y = cache["test"]
    else:
        stain_norm = build_macenko_normalizer(df) if use_stain_norm else None

        print("=> Building ViT backbone model for embedding extraction")
        backbone = torchvision.models.vit_b_16(
            weights=ViT_B_16_Weights.DEFAULT, image_size=224
        )

        if use_ssl:
            if not os.path.exists(SSL_CHECKPOINT):
                print("=> SSL Checkpoint not found")
                pretrain_ssl(df, device, epochs=ssl_epochs, stain_normalizer=stain_norm)
            backbone.load_state_dict(
                torch.load(SSL_CHECKPOINT, map_location="cpu"), strict=False
            )
            print("=> Loaded SSL backbone weights")

        # Removing classfication layer as we don't need the logits
        backbone.heads = nn.Identity()
        backbone.eval().to(device)

        train_X, train_y = extract_embeddings(
            df, train_idx, rand_transform, backbone, device, "train", stain_norm
        )
        valid_X, valid_y = extract_embeddings(
            df, valid_idx, det_transform, backbone, device, "valid", stain_norm
        )
        test_X, test_y = extract_embeddings(
            df, test_idx, det_transform, backbone, device, "test", stain_norm
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
            f" Epochs {e + 1:03d} | "
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
                        LUNG_MODEL_DIR,
                        f"LUNG_{model.__class__.__name__}_{e + 1}_EPOCHS.pth.tar",
                    ]
                ),
            )

    # Final model evaluation
    model.eval()
    test_acc = 0
    with torch.inference_mode():
        for feat_batch, label_batch in test_loader:
            logits = model(feat_batch)
            test_acc += (logits.argmax(1) == label_batch).sum().item()
    test_acc /= len(test_loader.dataset)
    print(f"\n=> Final Test accuracy: {test_acc * 100:.2f}% \n")


def main():
    train()


if __name__ == "__main__":
    main()
