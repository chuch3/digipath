import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset


class EmbeddedClassifer(nn.Module):
    def __init__(self, embedding, input_size, num_classes) -> None:
        super().__init__()
        self.embedding = embedding
        self.encoder = nn.Sequential(*self.conv_blocks)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.Dropout(p=0.5),
            nn.Softmax(dim=1),
        )

    def forward(self, X):
        with torch.no_grad():
            features = self.vit(X)

        logits = self.head(features)
        return logits


class ClassifierHead(nn.Module):
    """Mini MLP that lays on top of the frozen ViT CLS token (dim=768)"""

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)
