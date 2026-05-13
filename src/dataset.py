import os
from math import nan
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from config import LUNG_IMAGES_DIR, LUNG_METADATA_FILE


class LungHist700Dataset(Dataset):
    """
    Lung Histopathological Dataset Information

    Each image has a resolution of 1200 x 1600 pixels in `.jpg` format

    Superclasses:
    - “aca” (adenocarcinoma)
    - “scc” (squamous cell carcinoma)
    - “nor” (normal)

    Subclasses:
    - "wd" (well differentiated)
    - "md" (moderately differentiated)
    - "pd" (poorly differentiated)
    """

    def __init__(self, csv_file, root_dir, transform=None) -> None:
        self.metadata: pd.DateFrame = pd.read_csv(csv_file)
        self.paths = list(Path(LUNG_IMAGES_DIR).rglob("*"))
        self.label_map = {
            label: i
            for (i, label) in enumerate(
                (self.metadata["superclass"] + "_" + self.metadata["subclass"])
                .unique()
                .fillna("nor")
            )
        }

        self.labels = self.label_map.keys()

        # Each sample contains a `(path, label)` format for training
        self.sample = [(path, str(path.parent.name)) for path in self.paths]

        self.transform = transform

    def get_label(self, path: Path):
        label = path.name
        return label

    def __len__(self):
        return len(self.images)

    def __getitem__(self):
        pass


def load_dataset():
    # Image file naming convention for LungHist700 : "{label}_{resolution}_{image_id}_{patient_id}.jpg"
    LungHist700Dataset(csv_file=LUNG_METADATA_FILE, root_dir=LUNG_IMAGES_DIR)


def main():
    load_dataset()


if __name__ == "__main__":
    main()
