import os
from math import nan
from pathlib import Path

import pandas as pd
import torch
from matplotlib.pyplot import imshow
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

    * 691 Records
    *
    """

    def __init__(
        self,
        csv_file,
        root_dir,
        transform=None,
        loader=lambda p: Image.open(p, "r").convert("RGB"),
    ) -> None:
        self.metadata: pd.DateFrame = pd.read_csv(csv_file)
        self.paths = list(
            path for path in Path(LUNG_IMAGES_DIR).rglob("*") if path.is_file()
        )
        # Each sample contains a `(path, label, pid)` format for training
        self.sample = [
            (path, str(path.parent.name), path.stem.split("_")[-1])
            for path in self.paths
        ]
        self.label_map = {
            label: i
            for (i, label) in enumerate(
                (self.metadata["superclass"] + "_" + self.metadata["subclass"])
                .unique()
                .fillna("nor")
            )
        }
        self.loader = loader
        self.transform = transform

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        path, label, pid = self.sample[idx]
        image = self.loader(path)
        if self.transform:
            self.transform(image)
        return (image, label, pid)


def load_dataset():
    # Image file naming convention for LungHist700 : "{label}_{resolution}_{image_id}_{patient_id}.jpg"
    data = LungHist700Dataset(csv_file=LUNG_METADATA_FILE, root_dir=LUNG_IMAGES_DIR)


def main():
    load_dataset()


if __name__ == "__main__":
    main()
