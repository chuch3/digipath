import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from skimage.color import rgb2gray, rgb2hed, rgb2hsv
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel, threshold_otsu
from skimage.measure import label as cc_label
from skimage.measure import regionprops

warnings.filterwarnings("ignore", category=UserWarning)

sns.set_theme(style="whitegrid", context="notebook")

CLASS_PALETTE = {
    "normal": "#4C72B0",
    "aca": "#DD8452",
    "scc": "#55A868",
}


def _has_col(df: pd.DataFrame, col: str) -> bool:
    """Check a column exists; print a one-line note if it doesn't so you
    know why a given plot was skipped, rather than failing silently."""
    ok = col in df.columns
    if not ok:
        print(f"[skip] column '{col}' not found in df — skipping this analysis.")
    return ok


def plot_class_distribution(df: pd.DataFrame, label_col: str = "label"):
    counts = df[label_col].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color=sns.color_palette("Set2", len(counts)))
    ax.set_xlabel("Class label")
    ax.set_ylabel("Image count")
    ax.set_title("Class distribution")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.5, str(v), ha="center", va="bottom")
    plt.tight_layout()
    plt.show()
    return counts


def plot_patient_distribution(df: pd.DataFrame, patient_col: str = "patient_id"):
    if not _has_col(df, patient_col):
        return None

    per_patient = df[patient_col].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(per_patient.values, bins=20, color="#4C72B0", edgecolor="white")
    axes[0].set_xlabel("Images per patient")
    axes[0].set_ylabel("Number of patients")
    axes[0].set_title(
        f"Images-per-patient distribution\n({per_patient.shape[0]} patients total)"
    )

    if "label" in df.columns:
        patients_per_class = df.groupby("label")[patient_col].nunique()
        patients_per_class.plot(kind="bar", ax=axes[1], color=sns.color_palette("Set2"))
        axes[1].set_xlabel("Class label")
        axes[1].set_ylabel("Unique patients")
        axes[1].set_title("Unique patients per class")

    plt.tight_layout()
    plt.show()
    return per_patient


def _load_rgb(path: str, max_size: int = 512) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_size, max_size))
        return np.asarray(im).astype(np.float64) / 255.0


def plot_rgb_histograms_by_class(
    df: pd.DataFrame,
    n_per_class: int = 15,
    label_col: str = "label",
    path_col: str = "path",
):
    classes = sorted(df[label_col].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    channel_names = ["Red", "Green", "Blue"]

    for cls in classes:
        sub = df[df[label_col] == cls].sample(
            min(n_per_class, (df[label_col] == cls).sum()), random_state=0
        )
        imgs = [_load_rgb(p) for p in sub[path_col]]

        for ch in range(3):
            pixels = np.concatenate([im[..., ch].ravel() for im in imgs])
            axes[ch].hist(
                pixels,
                bins=50,
                range=(0, 1),
                alpha=0.5,
                label=f"class {cls}",
                density=True,
            )

    for ch, name in enumerate(channel_names):
        axes[ch].set_title(f"{name} channel")
        axes[ch].set_xlabel("Normalized intensity")
        axes[ch].legend(fontsize=8)
    axes[0].set_ylabel("Density")
    plt.suptitle("RGB channel histograms by class")
    plt.tight_layout()
    plt.show()


def plot_hed_separation_by_class(
    df: pd.DataFrame,
    n_per_class: int = 10,
    label_col: str = "label",
    path_col: str = "path",
):
    classes = sorted(df[label_col].unique())
    h_means, e_means, labels_used = [], [], []

    fig, axes = plt.subplots(len(classes), 3, figsize=(12, 3.2 * len(classes)))
    if len(classes) == 1:
        axes = axes[None, :]

    for row, cls in enumerate(classes):
        sub = df[df[label_col] == cls].sample(1, random_state=1)
        path = sub[path_col].iloc[0]
        rgb = _load_rgb(path)
        hed = rgb2hed(rgb)

        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"class {cls}: RGB")
        axes[row, 1].imshow(hed[..., 0], cmap="Purples")
        axes[row, 1].set_title("Hematoxylin channel")
        axes[row, 2].imshow(hed[..., 1], cmap="RdPu")
        axes[row, 2].set_title("Eosin channel")
        for ax in axes[row]:
            ax.axis("off")

    plt.suptitle("Example HED color deconvolution per class")
    plt.tight_layout()
    plt.show()

    # Now aggregate mean H/E intensity across a larger sample per class
    for cls in classes:
        sub = df[df[label_col] == cls].sample(
            min(n_per_class, (df[label_col] == cls).sum()), random_state=0
        )
        for p in sub[path_col]:
            hed = rgb2hed(_load_rgb(p))
            h_means.append(hed[..., 0].mean())
            e_means.append(hed[..., 1].mean())
            labels_used.append(cls)

    he_df = pd.DataFrame(
        {"label": labels_used, "hematoxylin_mean": h_means, "eosin_mean": e_means}
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.boxplot(data=he_df, x="label", y="hematoxylin_mean", ax=axes[0])
    axes[0].set_title(
        "Mean Hematoxylin intensity by class\n(proxy for nuclear density)"
    )
    sns.boxplot(data=he_df, x="label", y="eosin_mean", ax=axes[1])
    axes[1].set_title(
        "Mean Eosin intensity by class\n(proxy for cytoplasm/stroma density)"
    )
    plt.tight_layout()
    plt.show()

    return he_df


def plot_stain_variability_by_patient(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    n_per_patient: int = 3,
    path_col: str = "path",
):
    if not _has_col(df, patient_col):
        return None

    rows = []
    for patient, sub in df.groupby(patient_col):
        sample = sub.sample(min(n_per_patient, len(sub)), random_state=0)
        for p in sample[path_col]:
            rgb = _load_rgb(p, max_size=256)
            rows.append(
                {
                    "patient_id": patient,
                    "mean_R": rgb[..., 0].mean(),
                    "mean_G": rgb[..., 1].mean(),
                    "mean_B": rgb[..., 2].mean(),
                }
            )
    stain_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 4))
    melted = stain_df.melt(
        id_vars="patient_id",
        value_vars=["mean_R", "mean_G", "mean_B"],
        var_name="channel",
        value_name="mean_intensity",
    )
    sns.boxplot(data=melted, x="channel", y="mean_intensity", ax=ax)
    sns.stripplot(
        data=melted,
        x="channel",
        y="mean_intensity",
        color="black",
        alpha=0.25,
        size=3,
        ax=ax,
    )
    ax.set_title(
        "Per-patient mean channel intensity spread\n(wide spread = stain normalization likely needed)"
    )
    plt.tight_layout()
    plt.show()
    return stain_df


def plot_hsv_saturation_by_class(
    df: pd.DataFrame,
    n_per_class: int = 15,
    label_col: str = "label",
    path_col: str = "path",
):
    classes = sorted(df[label_col].unique())
    fig, ax = plt.subplots(figsize=(8, 4))

    for cls in classes:
        sub = df[df[label_col] == cls].sample(
            min(n_per_class, (df[label_col] == cls).sum()), random_state=0
        )
        sats = []
        for p in sub[path_col]:
            hsv = rgb2hsv(_load_rgb(p))
            sats.append(hsv[..., 1].ravel())
        sats = np.concatenate(sats)
        ax.hist(
            sats, bins=50, range=(0, 1), alpha=0.5, label=f"class {cls}", density=True
        )

    ax.set_xlabel("Saturation")
    ax.set_ylabel("Density")
    ax.set_title(
        "HSV saturation distribution by class\n(low saturation ~ background/whitespace)"
    )
    ax.legend()
    plt.tight_layout()
    plt.show()


def estimate_nuclei_stats(
    df: pd.DataFrame,
    n_per_class: int = 10,
    label_col: str = "label",
    path_col: str = "path",
):
    classes = sorted(df[label_col].unique())
    rows = []

    for cls in classes:
        sub = df[df[label_col] == cls].sample(
            min(n_per_class, (df[label_col] == cls).sum()), random_state=0
        )
        for p in sub[path_col]:
            rgb = _load_rgb(p, max_size=512)
            hed = rgb2hed(rgb)
            hema = hed[..., 0]
            # Hematoxylin channel: higher value = more nuclear stain.
            # Normalize before thresholding since HED values aren't in [0,1].
            hema_norm = (hema - hema.min()) / (hema.max() - hema.min() + 1e-8)
            try:
                thresh = threshold_otsu(hema_norm)
            except ValueError:
                continue
            mask = hema_norm > thresh

            labeled = cc_label(mask)
            props = regionprops(labeled)
            # Filter out tiny specks (noise) and implausibly huge blobs
            # (merged nuclei clumps) to get a cleaner per-nucleus size estimate.
            areas = [p_.area for p_ in props if 10 < p_.area < 2000]

            rows.append(
                {
                    "label": cls,
                    "path": p,
                    "n_nuclei_est": len(areas),
                    "mean_nucleus_area": np.mean(areas) if areas else np.nan,
                    "std_nucleus_area": np.std(areas) if areas else np.nan,
                }
            )

    nuclei_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.boxplot(data=nuclei_df, x="label", y="n_nuclei_est", ax=axes[0])
    axes[0].set_title("Estimated nuclei count per image")
    sns.boxplot(data=nuclei_df, x="label", y="mean_nucleus_area", ax=axes[1])
    axes[1].set_title("Mean nucleus area (pixels)")
    sns.boxplot(data=nuclei_df, x="label", y="std_nucleus_area", ax=axes[2])
    axes[2].set_title("Nucleus area std dev\n(higher = more pleomorphism)")
    plt.tight_layout()
    plt.show()

    return nuclei_df


def compute_glcm_texture_features(
    df: pd.DataFrame,
    n_per_class: int = 10,
    label_col: str = "label",
    path_col: str = "path",
):
    classes = sorted(df[label_col].unique())
    rows = []

    for cls in classes:
        sub = df[df[label_col] == cls].sample(
            min(n_per_class, (df[label_col] == cls).sum()), random_state=0
        )
        for p in sub[path_col]:
            gray = rgb2gray(_load_rgb(p, max_size=256))
            gray_uint8 = (gray * 255).astype(np.uint8)

            glcm = graycomatrix(
                gray_uint8,
                distances=[1],
                angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                levels=256,
                symmetric=True,
                normed=True,
            )
            rows.append(
                {
                    "label": cls,
                    "path": p,
                    "contrast": graycoprops(glcm, "contrast").mean(),
                    "homogeneity": graycoprops(glcm, "homogeneity").mean(),
                    "energy": graycoprops(glcm, "energy").mean(),
                    "correlation": graycoprops(glcm, "correlation").mean(),
                }
            )

    texture_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, feat in zip(axes, ["contrast", "homogeneity", "energy", "correlation"]):
        sns.boxplot(data=texture_df, x="label", y=feat, ax=ax)
        ax.set_title(feat.capitalize())
    plt.suptitle("GLCM texture features by class")
    plt.tight_layout()
    plt.show()

    return texture_df


def plot_edge_density_by_class(
    df: pd.DataFrame,
    n_per_class: int = 10,
    label_col: str = "label",
    path_col: str = "path",
):
    classes = sorted(df[label_col].unique())
    rows = []

    for cls in classes:
        sub = df[df[label_col] == cls].sample(
            min(n_per_class, (df[label_col] == cls).sum()), random_state=0
        )
        for p in sub[path_col]:
            gray = rgb2gray(_load_rgb(p, max_size=256))
            edges = sobel(gray)
            rows.append({"label": cls, "edge_density": edges.mean()})

    edge_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=edge_df, x="label", y="edge_density", ax=ax)
    ax.set_title("Mean Sobel edge magnitude by class")
    plt.tight_layout()
    plt.show()
    return edge_df
