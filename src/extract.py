import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LungImageLoaderDataset


def extract_embeddings(
    df, split_idx, transform, model, device, split_name, stain_normalizer
):
    dataset = LungImageLoaderDataset(df.iloc[split_idx], transform, stain_normalizer)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_features, all_labels = [], []

    # Removing classfication layer as we don't need the logits
    print(f"=> Extracting {split_name} embeddings ({len(split_idx)} images)")

    # NOTE: In future implementations, should return a X batch containing the MIL bag identifications and coordinates
    with torch.inference_mode():
        for img_batch, label_batch in tqdm(loader, desc=f"=> Embedding {split_name}"):
            img_batch = img_batch.to(device, non_blocking=True)
            features = model(img_batch)
            all_features.append(features.cpu())
            all_labels.append(label_batch)

    return torch.cat(all_features), torch.cat(all_labels)


if __name__ == "__main__":
    extract_embeddings()
