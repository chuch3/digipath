import torch.nn as nn


class ClassifierHead(nn.Module):
    # Mini MLP that lays on top of the frozen ViT CLS token (dim=768)

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)
