import os
from pathlib import Path

import numpy as np
import pandas as pd
import rich
import torch
from PIL import Image
from rich.panel import Panel
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

from constant import LUNG_IMAGES_DIR, LUNG_LOADED_FILE, LUNG_METADATA_FILE


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

    Each model will classify based on the supercalss due to the large number of classes needed to compute with each distinct superclass and subclass (7)
    """

    def __init__(
        self,
        csv_file,
        root_dir,
    ) -> None:

        print(f"=> Metadata file : `{csv_file}` ")
        print(f"=> Root directory : `{root_dir}`")

        self._meta: pd.DateFrame = pd.read_csv(csv_file)

        self._meta = self._meta.fillna("")  # Filling NaN from subclasses of `nor`

        cols = ["superclass", "subclass", "resolution", "image_id"]

        """
        As the provided dataset didn't include patient IDs within the filepaths, 
        we have to map using the metadata provided for patient-level group sampling. 

        Amazing...

        Conveniently (not), the filename matches the string join of each of series excluding
        patient ID. Hence, we create the composite keys as index to locate the patient's ID. 
        """
        self._meta.index = (
            self._meta[cols]
            .astype(str)
            .apply(lambda row: "_".join(part for part in row if part.strip()), axis=1)
        )  # Creates composite key index by joining each series as strings and stripping empty subclasses

        self._paths = list(
            path for path in Path(LUNG_IMAGES_DIR).rglob("*") if path.is_file()
        )

        # Each sample contains a `(path, label, patient_id)` format for training
        self._sample = [
            (
                path,
                self._meta.loc[path.stem]["superclass"],
                self._meta.loc[path.stem]["patient_id"],
                self._meta.loc[path.stem]["resolution"],
            )
            for path in self._paths
        ]

        assert len(self._sample) == 691  # Checking for proper dataset

        self._sample_df = pd.DataFrame(
            self._sample,
            columns=["path", "label", "patient_id", "resolution"],
        )

        # Label map doesn't have to be in getter method as it's rarely modified in datasets
        # {'aca_bd': 0, 'aca_md': 1, 'aca_pd': 2, 'scc_bd': 3, 'scc_md': 4, 'scc_pd': 5, 'nor': 6}
        self.label_map = {
            label: i for (i, label) in enumerate((self._meta["superclass"]).unique())
        }
        print(f"=> Label map : {self.label_map}")

    # Dataset must be copied before modifying, hence copy wrapper
    @property
    def df(self) -> pd.DataFrame:
        return self._sample_df.copy()

    def __len__(self):
        return len(self._sample)


# 7 Class version of the LungHist700 with grading differentiation
class LungHist700DatasetGrading(Dataset):
    def __init__(
        self,
        csv_file,
        root_dir,
    ) -> None:

        print(f"=> Metadata file : `{csv_file}` ")
        print(f"=> Root directory : `{root_dir}`")

        self._meta: pd.DateFrame = pd.read_csv(csv_file)

        self._meta = self._meta.fillna("")  # Filling NaN from subclasses of `nor`

        cols = ["superclass", "subclass", "resolution", "image_id"]

        """
        As the provided dataset didn't include patient IDs within the filepaths, 
        we have to map using the metadata provided for patient-level group sampling. 

        Amazing...

        Conveniently (not), the filename matches the string join of each of series excluding
        patient ID. Hence, we create the composite keys as index to locate the patient's ID. 
        """
        self._meta.index = (
            self._meta[cols]
            .astype(str)
            .apply(lambda row: "_".join(part for part in row if part.strip()), axis=1)
        )  # Creates composite key index by joining each series as strings and stripping empty subclasses

        self._paths = list(
            path for path in Path(LUNG_IMAGES_DIR).rglob("*") if path.is_file()
        )

        # Each sample contains a `(path, label, patient_id)` format for training
        self._sample = [
            (
                path,
                str(path.parent.name),
                self._meta.loc[path.stem]["patient_id"],
                self._meta.loc[path.stem]["resolution"],
            )
            for path in self._paths
        ]

        assert len(self._sample) == 691  # Checking for proper dataset

        self._sample_df = pd.DataFrame(
            self._sample,
            columns=["path", "label", "patient_id", "resolution"],
        )

        # Label map doesn't have to be in getter method as it's rarely modified in datasets
        # {'aca_bd': 0, 'aca_md': 1, 'aca_pd': 2, 'scc_bd': 3, 'scc_md': 4, 'scc_pd': 5, 'nor': 6}
        self.label_map = {
            label: i
            for (i, label) in enumerate(
                (self._meta["superclass"] + "_" + self._meta["subclass"]).unique()
            )
        }
        print(f"=> Label map : {self.label_map}")

    # Dataset must be copied before modifying, hence copy wrapper
    @property
    def df(self) -> pd.DataFrame:
        return self._sample_df.copy()

    def __len__(self):
        return len(self._sample)


class LungImageLoaderDataset(Dataset):
    """
    PIL Image loader of the LungImage700 dataset after dataset split.

    NOTE: This dataset loader is not generic and has to be modified for preference
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        stain_normalizer=None,
        loader=lambda p: Image.open(p).convert("RGB"),
    ) -> None:
        self._df = df
        self._loader = loader
        self._transform = transform
        self._stain_norm = stain_normalizer

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self._df.iloc[idx]
        image = self._loader(row["path"])

        if self._stain_norm:
            image = self._stain_norm(image)

        if self._transform:
            image = self._transform(image)

        return (image, row["label"])


def stratify_group_split(df, label_col, group_col, train_size, random_state):
    k = max(
        2, round(1 / (1 - train_size))
    )  # k is either the minimum of 2 or 1 - train_size
    stratify = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=random_state)

    groups = df[group_col].to_numpy()
    labels = df[label_col].to_numpy()

    kept_idx, held_out_idx = next(stratify.split(df, labels, groups))
    return kept_idx, held_out_idx


def load_dataset(
    csv_file=LUNG_METADATA_FILE,
    root_dir=LUNG_IMAGES_DIR,
    load_file=LUNG_LOADED_FILE,
    random_state=42,
):
    # Image file naming convention for LungHist700 : "{label}_{resolution}_{image_id}_{patient_id}.jpg"
    data = LungHist700Dataset(csv_file, root_dir)
    df = data.df
    df["label"] = df["label"].map(data.label_map)  # Multi-class encoding for training

    rich.print(
        Panel(
            "LungHist700 Target Label Statistics\n"
            "----------\n"
            f"Label map: {data.label_map}\n\n"
            f"{df['label'].value_counts()}"
        )
    )

    # Patient level stratification while maintaining target label balance
    train_idx, temp_idx = stratify_group_split(
        df,
        label_col="label",
        group_col="patient_id",
        train_size=0.7,
        random_state=random_state,
    )

    df_temp = df.iloc[temp_idx].reset_index(drop=True)

    valid_pos, test_pos = stratify_group_split(
        df_temp,
        label_col="label",
        group_col="patient_id",
        train_size=0.5,
        random_state=random_state,
    )

    temp_idx = np.asarray(temp_idx)
    valid_idx = temp_idx[valid_pos]
    test_idx = temp_idx[test_pos]

    # The splits aren't exactly 70/15/15 due to patient-level and stratified grouping
    rich.print(
        Panel(
            "LungHist700 Dataset Split Statistics (Patient-Level, Stratified)\n"
            "--------------\n"
            f"Train split : {len(train_idx) / len(df) * 100:.4f}%\n"
            f"Validation split : {len(valid_idx) / len(df) * 100:.4f}%\n"
            f"Test split : {len(test_idx) / len(df) * 100:.4f}%\n\n"
            f"Train label dist:\n{df.iloc[train_idx]['label'].value_counts(normalize=True).sort_index() * 100}\n\n"
            f"Valid label dist:\n{df.iloc[valid_idx]['label'].value_counts(normalize=True).sort_index() * 100}\n\n"
            f"Test label dist:\n{df.iloc[test_idx]['label'].value_counts(normalize=True).sort_index() * 100}"
        )
    )

    # Check if the data loaded file exists or not
    print("=> Checking if loaded dataset exists.")
    if not os.path.isfile(load_file):
        print(f"=> Loaded dataset doesn't exist! Saving to `{load_file}`")
        df.to_csv(load_file, encoding="utf-8", header=True, index=False)
    else:
        print("=> Loaded dataset already exists, continuing process.")

    return train_idx, valid_idx, test_idx, data.label_map


def main():
    load_dataset()


if __name__ == "__main__":
    main()
