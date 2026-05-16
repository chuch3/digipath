import os
from pathlib import Path

import pandas as pd
import rich
from rich.panel import Panel
from sklearn.model_selection import GroupShuffleSplit
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
    """

    def __init__(
        self,
        csv_file,
        root_dir,
    ) -> None:

        print(f"=> Loading metada `{csv_file}` from root `{root_dir}`\n")

        self._meta: pd.DateFrame = pd.read_csv(csv_file)

        self._meta = self._meta.fillna("")  # Filling NaN from subclasses of `nor`

        print(self._meta.info())

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

        # Each sample contains a `(path, label, patient_id)` format for training
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

        assert len(self._sample) == 691  # Checking for proper dataset

        self._sample_df = pd.DataFrame(
            self._sample,
            columns=["path", "label", "patient_id"],
        )

        # Label map doesn't have to be in getter method as it's rarely modified in datasets
        # {'aca_bd': 0, 'aca_md': 1, 'aca_pd': 2, 'scc_bd': 3, 'scc_md': 4, 'scc_pd': 5, 'nor': 6}
        self.label_map = {
            label: i
            for (i, label) in enumerate(
                (self._meta["superclass"] + "_" + self._meta["subclass"]).unique()
            )
        }
        self.label_map["nor"] = self.label_map.pop(
            "nor_"
        )  # NOTE: this logic can be optimized further but whateves

        assert len(self.label_map) == 7  # Checking for proper dataset

    # Dataset must be copied before modifying, hence copy wrapper
    @property
    def df(self) -> pd.DataFrame:
        return self._sample_df.copy()

    def __len__(self):
        return len(self._sample)


def load_dataset(random_state=42):
    # Image file naming convention for LungHist700 : "{label}_{resolution}_{image_id}_{patient_id}.jpg"
    data = LungHist700Dataset(csv_file=LUNG_METADATA_FILE, root_dir=LUNG_IMAGES_DIR)
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

    X, y = df[["path", "patient_id"]], df["label"]

    # Keeps the groups together in patient-level with shuffling, not stratification
    gss_1 = GroupShuffleSplit(
        n_splits=1,
        train_size=0.7,
        random_state=random_state,
    )

    gss_2 = GroupShuffleSplit(
        n_splits=1,
        train_size=0.5,
        random_state=random_state,
    )

    train_idx, temp_idx = next(gss_1.split(X, y, X["patient_id"]))

    X_temp, y_temp = (
        X.iloc[temp_idx].reset_index(drop=True),
        y.iloc[temp_idx].reset_index(drop=True),
    )

    test_idx, valid_idx = next(gss_2.split(X_temp, y_temp, X_temp["patient_id"]))

    """
    X_test, y_test = X.iloc[test_idx ], y.iloc[test_idx ]
    X_valid, y_valid = X.iloc[valid_idx ], y.iloc[valid_idx ]
    X_train, y_train = X.iloc[train_idx ], y.iloc[train_idx ]
    """

    # The splits arent' as accurate as (80/10/10) due to patient-level splits
    rich.print(
        Panel(
            "LungHist700 Dataset Split Statistics (Patient-Level)\n"
            "----------\n"
            f"Train split : {len(train_idx) / len(df) * 100:.4f}%\n"
            f"Validation splt : {len(test_idx) / len(df) * 100:.4f}%\n"
            f"Test splt : {len(valid_idx) / len(df) * 100:.4f}%"
        )
    )

    # Check if the data loaded file exists or not
    print("=> Checking if loaded dataset exists.")
    if not os.path.isfile(LUNG_LOADED_FILE):
        print(f"=> Loaded dataset doesn't exist! Saving to `{LUNG_LOADED_FILE}`")
        df.to_csv(LUNG_LOADED_FILE, encoding="utf-8", header=True, index=False)
    else:
        print("=> Loaded dataset already exists, continuing process.")

    return train_idx, valid_idx, test_idx, data.label_map


def main():
    load_dataset()


if __name__ == "__main__":
    main()
