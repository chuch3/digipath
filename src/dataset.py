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

        self._meta: pd.DateFrame = pd.read_csv(csv_file)

        self._meta = self._meta.fillna("")  # Filling NaN from subclasses of `nor`

        print(self._meta.info())

        cols = ["superclass", "subclass", "resolution", "image_id"]

        """
        As the provided dataset doesn't include the patient ID within the filepaths, 
        we have to map using the metadata provided for patient-level stratify sampling. 
        Amazing :) ....

        Conveniently (not), the filename matches the composite join of each of series excluding
        patient ID. So, we create the compostie keys as index to locate the patient's ID. daammmmmnnn
        """
        self._meta.index = (
            self._meta[cols]
            .astype(str)
            .apply(lambda row: "_".join(part for part in row if part.strip()), axis=1)
        )  # Creates composite key index by joining each series as strings and stripping empty subclasses

        # Each sample contains a `(path, label, pid)` format for training
        self._paths = list(
            path for path in Path(LUNG_IMAGES_DIR).rglob("*") if path.is_file()
        )

        self._sample = [
            (
                path,
                str(path.parent.name),
                self._meta.loc[path.stem]["patient_id"],
            )
            for path in self._paths
        ]

        assert len(self._sample) == 691  # checking for proper dataset

        self._sample_df = pd.DataFrame(
            self._sample,
            columns=["path", "label", "patient_id"],
        )

        # Label map doesn't have to be in getter as it's rarely modified in datasets
        self.label_map = {
            label: i
            for (i, label) in enumerate(
                (self._meta["superclass"] + "_" + self._meta["subclass"]).unique()
            )
        }
        self.label_map["nor"] = self.label_map.pop(
            "nor_"
        )  # NOTE: this logic can be optimized more but whateves

        assert len(self.label_map) == 7  # checking for proper dataset

    # Dataset must be copied before modifying, hence copy wrapper
    @property
    def df(self) -> pd.DataFrame:
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
    df["label_encode"] = df["label"].map(data.label_map)

    rich.print(
        Panel(
            "Target Label Statistics\n"
            "----------\n"
            f"Label map: {data.label_map}\n\n"
            f"{df['label_encode'].value_counts()}"
        )
    )

    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)

    # Stratify splitting and shuffle based on groups (patients)
    train_index, temp_index = next(
        gss.split(X=df["path"], y=df["label"], groups=df["patient_id"])
    )
    temp_df = df.iloc[temp_index]

    a, b = train_test_split(temp_df, test_size=0.5, stratify=temp_df["patient_id"])
    print(a, b)

    train_df = df.loc[train_index].reset_index(drop=True)

    rich.print(
        Panel(
            "Dataset Split Statistics\n"
            "----------\n"
            f"Train split : {len(train_df) / len(df) * 100:.4f}%"
            # f"Validation splt : {X_test.size / len(df):.4f}%\n"
            # f"Test splt : {X_valid.size / len(df):.4f}%"
        )
    )


def main():
    load_dataset()


if __name__ == "__main__":
    main()
