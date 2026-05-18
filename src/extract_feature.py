import numpy as np
import torch
import tqdm


def extract_embedding(output_path, loader, model, device="cpu", verbose=None):
    all_embedding = []
    all_features = []
    all_labels = []

    # Removing classfication layer as we don't need the logits
    model.heads = torch.nn.Identity()

    if verbose:
        print(f"=> Processing a total of {len(loader)} batches")

    # Should return a X batch containing the MIL bag identifications and coordinates
    with torch.inference_mode():
        for img_batch, label_batch in tqdm(loader):
            img_batch = img_batch.to(device, non_blocking=True)
            embeddings = model(img_batch)
            embeddings = embeddings.cpu().numpy().astype(np.float32)

    torch.save({})
