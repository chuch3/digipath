import os

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from config import _LUNG_DIR


class LungHist700Dataset(torch.utils.data.Dataset):
    def __init__(self, dir, transform=None):
        self.data_dir = dir
        self.images = os.listdir(dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_dir, self.images[index])
        image = np.array(Image.open(image_path))

        # Applying the transform
        if self.transform:
            image = self.transform(image)

        return image


def load_dataset():
    data = ImageFolder(root=_LUNG_DIR)

    loader = DataLoader(
        data,
        shuffle=True,
    )


if __name__ == "__main__":
    load_dataset()
