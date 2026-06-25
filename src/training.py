import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights
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
from eval import run_full_evaluation
from extract import extract_embeddings
from macenko import build_macenko_normalizer
from model import ClassifierHead
from ssl_train import pretrain_ssl
from transform import det_transform, rand_transform


def train_eval(
    batch_size=32,
    lr=1e-4,
    epochs=200,
    ssl_epochs=10,
    csv_file=LUNG_METADATA_FILE,
    root_dir=LUNG_IMAGES_DIR,
    load_file=LUNG_LOADED_FILE,
    embedding_file=None,  # example value : Path(*[LUNG_PREPROCESS_DIR, "lung_embedded_stain_labeled.pt"])
    embedding_cache_dir=LUNG_PREPROCESS_DIR,
    device=None,
    use_stain_norm=False,
    use_ssl=False,
    use_resnet=False,  # False for ViT-B-16 backbone model
    verbose=True,
    use_eval=True,
    resolution="20x",
    eval_label="",
    early_patience=10,
    random_state=42,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=> Using device : {device}")

    torch.manual_seed(random_state)

    # This guarantees the loaded dataset is built before reading
    train_idx, valid_idx, test_idx, label_map = load_dataset(
        csv_file, root_dir, load_file
    )

    df = pd.read_csv(load_file, encoding="utf-8")

    if resolution in ["20x", "40x"]:
        print(f"=> Dataset used in resolution {resolution}")
        df = df[df["resolution"] == resolution].reset_index(drop=True)

    # Readjusting indecies based on the resolution selected
    train_idx = [i for i in train_idx if i < len(df)]
    valid_idx = [i for i in valid_idx if i < len(df)]
    test_idx = [i for i in test_idx if i < len(df)]

    stain_suffix = "stain" if use_stain_norm else "raw"
    ssl_suffix = "ssl" if use_ssl else "labeled"
    resnet_suffix = "resnet" if use_resnet else "embedded"

    if not embedding_file:
        embedding_cache = Path(
            *[
                embedding_cache_dir,
                f"lung_{resnet_suffix}_{resolution}_{stain_suffix}_{ssl_suffix}.pt",
            ]
        )
    else:
        embedding_cache = embedding_file

    print(f"\n=> Embedding file used : {embedding_cache}\n")

    if os.path.exists(embedding_cache):
        print(f"=> Loading cached embeddings from {embedding_cache}")
        cache = torch.load(embedding_cache, map_location="cpu")
        train_X, train_y = cache["train"]
        valid_X, valid_y = cache["valid"]
        test_X, test_y = cache["test"]
    else:
        stain_norm = build_macenko_normalizer(df) if use_stain_norm else None

        if use_resnet:
            print("=> Building ResNet-50 backbone model for embedding extraction")
            backbone = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            print("=> Building ViT-B-16 backbone model for embedding extraction")
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

    def make_loader(X, y, drop_last=True):
        ds = TensorDataset(X.to(device), y.to(device))
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=drop_last,
        )

    train_loader = make_loader(train_X, train_y, True)
    valid_loader = make_loader(valid_X, valid_y, False)
    test_loader = make_loader(test_X, test_y, False)

    embed_dim = train_X.shape[1]  # 768 for ViT-B/16
    model = ClassifierHead(embed_dim, num_classes=len(label_map)).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)
    loss_fn = nn.CrossEntropyLoss()

    loss_epochs, acc_epochs = [], []
    val_loss_epochs, val_acc_epochs = [], []

    val_acc = 0.0

    best_flag = True
    best_val_loss = float("inf")
    best_epoch = -1
    early_counter = 0

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
        val_loss_epochs.append(val_loss)
        val_acc_epochs.append(val_acc)

        scheduler.step(val_loss)

        if verbose:
            print(
                f" Epochs {e + 1:03d} | "
                f"Train loss : {loss_epochs[-1]:.2f} | Train acc {acc_epochs[-1] * 100:.2f} % | "
                f"Val loss {val_loss:.4f} | Val acc {val_acc * 100:.2f}%",
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
        else:
            early_counter += 1

        # Stop if patience exceeded
        if early_counter >= early_patience and best_flag:
            best_epoch = e + 1
            print(f"\n>>> Early stopping triggered at epoch {best_epoch + 1}")
            print(f">>> Best val_loss: {best_val_loss:.4f}")
            best_flag = False
            # Final model evaluation
            model.eval()
            test_acc = 0
            with torch.inference_mode():
                for feat_batch, label_batch in test_loader:
                    logits = model(feat_batch)
                    test_acc += (logits.argmax(1) == label_batch).sum().item()
            test_acc /= len(test_loader.dataset)
            best_train = acc_epochs[-1] * 100
            best_val = val_acc * 100
            best_test = test_acc * 100

    # Final model evaluation
    model.eval()
    test_acc = 0
    with torch.inference_mode():
        for feat_batch, label_batch in test_loader:
            logits = model(feat_batch)
            test_acc += (logits.argmax(1) == label_batch).sum().item()
    test_acc /= len(test_loader.dataset)

    print(f"\n=> Final Train Accuracy: {acc_epochs[-1] * 100:.2f}% \n")
    print(f"=> Final Valid Accuracy: {val_acc * 100:.2f}% \n")
    print(f"=> Final Test Accuracy: {test_acc * 100:.2f}% \n")

    print(f"\n=> Best Epoch {best_epoch}\n")
    print(f"=> Best Train Accuracy: {best_train:.2f}%\n")
    print(f"=> Best Valid Accuracy: {best_val:.2f}%\n")
    print(f"=> Best Test Accuracy: {best_test:.2f}%\n")

    if use_eval:
        run_full_evaluation(
            model,
            test_loader,
            label_map,
            device,
            best_epoch,
            loss_epochs,
            acc_epochs,
            val_loss_epochs=val_loss_epochs,
            val_acc_epochs=val_acc_epochs,
            output_dir=LUNG_MODEL_DIR,
            label=f"{eval_label}_{resolution}",
        )


if __name__ == "__main__":
    train_eval()
