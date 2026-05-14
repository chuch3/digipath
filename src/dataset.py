from pathlib import Path

import pandas as pd
import rich
import torch
from matplotlib.pyplot import imshow
from PIL import Image
from rich.panel import Panel
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset

from config import LUNG_IMAGES_DIR, LUNG_METADATA_FILE


class LungHist700Dataset(Dataset):
    """
    # Lung Histopathological Dataset Information

    Each image has a resolution of 1200 x 1600 pixels in `.jpg` format with 691 Records in total.

    - Superclasses:
        - “aca” (adenocarcinoma)
        - “scc” (squamous cell carcinoma)
        - “nor” (normal)

    - Subclasses:
        - "wd" (well differentiated)
        - "md" (moderately differentiated)
        - "pd" (poorly differentiated)

    > For more dataset info at [link](https://www.nature.com/articles/s41597-024-03944-3.pdf)
    """

    def __init__(
        self,
        csv_file,
        root_dir,
    ) -> None:
        print("=> Loading LungHist700 dataset from root")
        self._metadata: pd.DateFrame = pd.read_csv(csv_file)

        self._paths = list(
            path for path in Path(LUNG_IMAGES_DIR).rglob("*") if path.is_file()
        )

        # Each sample contains a `(path, label, pid)` format for training
        self._sample = [
            (path, str(path.parent.name), path.stem.split("_")[-1])
            for path in self._paths
        ]

        assert len(self._sample) == 691  # checking for proper dataset

        self._sample_df = pd.DataFrame(
            self._sample,
            columns=["path", "label", "pid"],
        )

        # Label map doesn't have to be in getter as it's rarely modified in datasets
        self.label_map = {
            label: i
            for (i, label) in enumerate(
                (self._metadata["superclass"] + "_" + self._metadata["subclass"])
                .unique()
                .fillna("nor")
            )
        }
        assert len(self.label_map) == 7  # checking for proper dataset

    # Dataset must be copied before modifying, hence copy wrapper
    @property
    def df(self):
        return self._sample_df.copy()

    def __len__(self):
        return len(self._sample)


# i don't know how to stratify split directly from torch `Dataset`, so df will do for now
#
# splliting requires dataframes, dataloader requires `Dataset`s ... bruh


class LungImageLoaderDataset(Dataset):
    """
    PIL Image loader of the LungImage700 dataset after dataset split.

    NOTE: This dataset loader is not generic and has to be modified for preference
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        loader=lambda p: Image.open(p, "r").convert("RGB"),
    ) -> None:
        self._df = df
        self._loader = loader
        self._transform = transform

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self._df.iloc[idx]
        image = self.loader(row["path"])

        if self.transform:
            self.transform(image)

        return (image, row["label"], row["pid"])


def load_dataset(random_state=42):
    # Image file naming convention for LungHist700 : "{label}_{resolution}_{image_id}_{patient_id}.jpg"
    data = LungHist700Dataset(csv_file=LUNG_METADATA_FILE, root_dir=LUNG_IMAGES_DIR)
    df = data.df
    X, y, groups = df["path"], df["label"], df["pid"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state
    )

    rich.print(
        Panel(
            "Dataset Split Statistics (%) : \n"
            f"Train split : {X_train.size / len(df):.4f}%\n"
            f"Validation splt : {X_test.size / len(df):.4f}%\n"
            f"Test splt : {X_valid.size / len(df):.4f}%"
        )
    )

    # Stratify splitting and shuffle based on groups (patients)
    # train_index, test_index = gss.split(X=df["path"], y=df["label"], groups=df["pid"])


def main():
    load_dataset()


if __name__ == "__main__":
    main()
