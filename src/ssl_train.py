import copy
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ViT_B_16_Weights
from tqdm import tqdm

from constant import SSL_CHECKPOINT
from transform import ssl_transform


class MoCoDataset(Dataset):
    def __init__(self, df, ssl_transform, stain_normalizer=None):
        self._df = df
        self._aug = ssl_transform
        self._stain = stain_normalizer

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        row = self._df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        if self._stain:
            image = self._stain(image)
        return self._aug(image), self._aug(
            image
        )  # (Query Encoder, Key Encoder) as two augmented images


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


def build_encoder(freeze_until_block=10):
    backbone = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    backbone.heads = nn.Identity()

    for name, p in backbone.named_parameters():
        trainable = (
            f"encoder.layers.encoder_layer_{freeze_until_block}" in name
            or f"encoder.layers.encoder_layer_{freeze_until_block + 1}" in name
            or name.startswith("encoder.ln")
        )
        p.requires_grad = trainable

    proj_head = ProjectionHead(in_dim=768)
    return nn.Sequential(
        backbone, proj_head
    )  # encoder[0]=backbone, encoder[1]=proj_head


class MoCo(nn.Module):
    def __init__(
        self, encoder_fn, dim=128, queue_size=4096, momentum=0.99, temperature=0.07
    ):
        super().__init__()
        self.K = queue_size
        self.m = momentum
        self.T = temperature

        self.encoder_q = encoder_fn()
        self.encoder_k = copy.deepcopy(self.encoder_q)
        for p in self.encoder_k.parameters():
            p.requires_grad = (
                False  # momentum encoder is EMA-updated only, never via backward
            )

        self.register_buffer("queue", F.normalize(torch.randn(dim, queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update(self):
        for pq, pk in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            pk.data.mul_(self.m).add_(pq.data, alpha=1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        end = ptr + batch_size
        if end <= self.K:
            self.queue[:, ptr:end] = keys.T
        else:
            first = self.K - ptr
            self.queue[:, ptr:] = keys[:first].T
            self.queue[:, : end - self.K] = keys[first:].T
        self.queue_ptr[0] = end % self.K

    def forward(self, im_q, im_k):
        q = self.encoder_q(im_q)  # Query vector
        with torch.no_grad():
            self._momentum_update()
            k = self.encoder_k(im_k)  # Key vector

        # Positive pair similarity
        l_pos = (q * k).sum(dim=1, keepdim=True)
        # Negative pair similarity
        l_neg = q @ self.queue.clone().detach()  # (B, K)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)

        self._dequeue_and_enqueue(k)
        return F.cross_entropy(logits, labels)


def pretrain_ssl(
    df,
    device="cpu",
    epochs=1000,
    batch_size=16,
    lr=3e-4,
    freeze_until_block=10,
    queue_size=4096,
    momentum=0.999,
    temperature=0.07,
    max_train_seconds=9 * 3600,
    stain_normalizer=None,
    checkpoint=SSL_CHECKPOINT,
    num_workers=None,
):
    if os.path.exists(checkpoint):
        print(f"=> SSL checkpoint found at {checkpoint}, skipping pre-training")
        return

    torch.set_num_threads(os.cpu_count())
    num_workers = num_workers if num_workers is not None else max(1, os.cpu_count() - 1)

    print(
        "\n=> Starting Self-Supervised Pretraining with MoCo (ViT-B/16, mostly frozen)\n"
    )

    dataset = MoCoDataset(df, ssl_transform, stain_normalizer)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    def encoder_fn():
        return build_encoder(freeze_until_block=freeze_until_block)

    model = MoCo(
        encoder_fn,
        dim=128,
        queue_size=queue_size,
        momentum=momentum,
        temperature=temperature,
    ).to(device)

    trainable_params = [p for p in model.encoder_q.parameters() if p.requires_grad]
    print(
        f"=> Training {sum(p.numel() for p in trainable_params):,} of "
        f"{sum(p.numel() for p in model.encoder_q.parameters()):,} backbone+head params"
    )

    optimizer = AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    start_time = time.time()
    stop_early = False

    loss_epochs = []
    for epoch in tqdm(range(epochs), desc="=> SSL epochs"):
        if stop_early:
            break

        model.train()
        total_loss = 0.0
        n_steps = 0

        for im_q, im_k in tqdm(loader, leave=False):
            elapsed = time.time() - start_time
            if elapsed > max_train_seconds:
                print(
                    f"\n=> Time elapsed of {max_train_seconds / 3600:.1f}h reached, stopping"
                )
                stop_early = True
                break

            im_q, im_k = im_q.to(device), im_k.to(device)
            loss = model(im_q, im_k)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_steps += 1
            loss_epochs.append(total_loss / len(loader))

        if n_steps > 0:
            print(
                f"=> SSL Epoch {epoch + 1:03d} | Loss {total_loss / n_steps:.4f} "
                f"| Elapsed {(time.time() - start_time) / 3600:.2f}h"
            )

    backbone_state = {
        "state_dict": model.encoder_q[0].state_dict(),
        "loss_history": loss_epochs,
    }
    torch.save(backbone_state, checkpoint)
    print(f"=> SSL backbone saved to: {checkpoint}")
