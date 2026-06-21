"""
eda.py — Exploratory Data Analysis for the LungHist700 H&E dataset.

Designed to be run cell-by-cell in a Jupyter notebook. Each function is
self-contained: load the dataframe once, then call whichever functions you
want, in any order, and each one will plot inline and/or return a small
results object you can inspect further.

Typical notebook usage:

    from eda import *

    df = load_eda_dataframe()          # cell 1
    plot_class_distribution(df)        # cell 2
    plot_patient_distribution(df)      # cell 3
    plot_magnification_by_class(df)    # cell 4
    sample_image_grid(df)              # cell 5
    ...

Assumes the same project conventions as gradcam.py:
    from constant import LUNG_IMAGES_DIR, LUNG_LOADED_FILE, LUNG_METADATA_FILE
    from dataset import load_dataset

The dataframe is expected to have at least:
    'path'  : str, file path to each image
    'label' : int, encoded class label

Optional columns, used opportunistically if present (functions degrade
gracefully — printing a note and skipping — if a column is missing):
    'patient_id'      : patient / subject identifier
    'magnification'   : e.g. '20x' / '40x'
    'differentiation' : e.g. 'well' / 'moderate' / 'poor'
    'class_name'       : human-readable class name (e.g. 'aca_md', 'scc_pd', 'normal')

If your df uses different column names, just rename them after loading, e.g.:
    df = df.rename(columns={'mag': 'magnification', 'patient': 'patient_id'})
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from skimage.color import rgb2hed, rgb2gray, rgb2hsv
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import threshold_otsu, sobel
from skimage.measure import label as cc_label, regionprops

from constant import LUNG_IMAGES_DIR, LUNG_LOADED_FILE, LUNG_METADATA_FILE
from dataset import load_dataset

warnings.filterwarnings("ignore", category=UserWarning)

sns.set_theme(style="whitegrid", context="notebook")

# A consistent color per class makes every plot in the notebook comparable
# at a glance. Extend this if you have more than 4 classes (e.g. 7-way grade split).
CLASS_PALETTE = {
    "normal": "#4C72B0",
    "aca": "#DD8452",
    "scc": "#55A868",
}


# ---------------------------------------------------------------------------
# 0. Data loading
# ---------------------------------------------------------------------------

def load_eda_dataframe() -> pd.DataFrame:
    """
    Load the LungHist700 metadata/image dataframe using the same
    `load_dataset` pipeline as the rest of the project, so the EDA reflects
    exactly the data your model will train/evaluate on (not a separate,
    possibly-inconsistent reading of the raw files).

    Returns
    -------
    df : DataFrame with (at minimum) 'path' and 'label' columns, read from
         LUNG_LOADED_FILE after load_dataset() has (re)generated it.
    """
    # load_dataset() writes/refreshes LUNG_LOADED_FILE as a side effect and
    # also returns train/valid/test split indices + the label map — we only
    # need the dataframe itself here, so the splits are discarded.
    load_dataset(LUNG_METADATA_FILE, LUNG_IMAGES_DIR, LUNG_LOADED_FILE)
    df = pd.read_csv(LUNG_LOADED_FILE, encoding="utf-8")
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
    return df


def _has_col(df: pd.DataFrame, col: str) -> bool:
    """Check a column exists; print a one-line note if it doesn't so you
    know why a given plot was skipped, rather than failing silently."""
    ok = col in df.columns
    if not ok:
        print(f"[skip] column '{col}' not found in df — skipping this analysis.")
    return ok


# ---------------------------------------------------------------------------
# 1. Class / metadata distribution
# ---------------------------------------------------------------------------

def plot_class_distribution(df: pd.DataFrame, label_col: str = "label"):
    """
    Bar chart of image counts per class.

    Why: confirms class balance before you trust any accuracy metric later.
    A skewed split (e.g. far more normal than SCC) means accuracy alone is
    misleading and you'll want stratified sampling / class weighting /
    macro-F1 instead of raw accuracy when you evaluate the classifier.
    """
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
    """
    Histogram of how many images each patient contributes, plus a per-class
    breakdown of unique patient counts.

    Why this matters: LungHist700 has 45 patients but 691 images — multiple
    images per patient. If you split train/test by IMAGE rather than by
    PATIENT, the same patient's tissue can appear in both train and test,
    which leaks patient-specific staining/morphology and inflates your test
    accuracy. This plot tells you whether that risk is severe (a few
    patients contributing many images) or mild (fairly even contribution).
    """
    if not _has_col(df, patient_col):
        return None

    per_patient = df[patient_col].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(per_patient.values, bins=20, color="#4C72B0", edgecolor="white")
    axes[0].set_xlabel("Images per patient")
    axes[0].set_ylabel("Number of patients")
    axes[0].set_title(f"Images-per-patient distribution\n({per_patient.shape[0]} patients total)")

    if "label" in df.columns:
        patients_per_class = df.groupby("label")[patient_col].nunique()
        patients_per_class.plot(kind="bar", ax=axes[1], color=sns.color_palette("Set2"))
        axes[1].set_xlabel("Class label")
        axes[1].set_ylabel("Unique patients")
        axes[1].set_title("Unique patients per class")

    plt.tight_layout()
    plt.show()
    return per_patient


def plot_magnification_by_class(df: pd.DataFrame, mag_col: str = "magnification", label_col: str = "label"):
    """
    Cross-tabulated count of magnification (20x / 40x) by class, shown as
    a grouped bar chart and a normalized heatmap.

    Why: if one class is mostly 20x and another mostly 40x, a model can
    learn to classify based on magnification-correlated cues (e.g. overall
    cell size in pixels) rather than genuine histological pattern — a subtle
    form of shortcut learning that won't generalize. This plot is a direct
    check for that confound before you train anything.
    """
    if not _has_col(df, mag_col):
        return None

    ct = pd.crosstab(df[label_col], df[mag_col])
    ct_norm = pd.crosstab(df[label_col], df[mag_col], normalize="index")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ct.plot(kind="bar", ax=axes[0], color=sns.color_palette("Set2"))
    axes[0].set_title("Magnification count by class")
    axes[0].set_ylabel("Image count")

    sns.heatmap(ct_norm, annot=True, fmt=".2f", cmap="Blues", ax=axes[1])
    axes[1].set_title("Magnification proportion by class\n(rows sum to 1.0)")

    plt.tight_layout()
    plt.show()
    return ct


def plot_differentiation_distribution(df: pd.DataFrame, diff_col: str = "differentiation", label_col: str = "label"):
    """
    Count of well/moderate/poor differentiation grades, split by cancer
    type (ACA vs SCC).

    Why: differentiation grade is itself a 7-way classification target in
    the original LungHist700 design. Even if you're only doing 3-way
    (normal/ACA/SCC) classification, checking this distribution tells you
    whether your "ACA" class is dominated by one differentiation level —
    which matters because poorly-differentiated tumors of either type can
    look more similar to each other than to their own well-differentiated
    counterparts, a known source of inter-class confusion in this dataset.
    """
    if not _has_col(df, diff_col):
        return None

    ct = pd.crosstab(df[label_col], df[diff_col])
    fig, ax = plt.subplots(figsize=(7, 4))
    ct.plot(kind="bar", stacked=True, ax=ax, color=sns.color_palette("Set2", ct.shape[1]))
    ax.set_title("Differentiation grade by class")
    ax.set_ylabel("Image count")
    plt.tight_layout()
    plt.show()
    return ct


def check_image_dimensions(df: pd.DataFrame, n_check: int = 50, path_col: str = "path"):
    """
    Open a random sample of images and record their (width, height) to
    confirm dataset uniformity.

    Why: LungHist700 images are documented as 1200x1600. If a subset
    deviates (cropped, resized, or corrupted on disk), your transform
    pipeline (which assumes a consistent aspect ratio before the final
    Resize(224,224)) may stretch or distort some images differently from
    others, introducing a hidden preprocessing inconsistency. This is a
    sanity check, not a visualization.
    """
    sample = df.sample(min(n_check, len(df)), random_state=0)
    sizes = []
    for p in sample[path_col]:
        try:
            with Image.open(p) as im:
                sizes.append(im.size)  # (width, height)
        except Exception as e:
            sizes.append((None, None))
            print(f"[warn] could not open {p}: {e}")

    sizes_df = pd.DataFrame(sizes, columns=["width", "height"])
    print(sizes_df.value_counts())
    return sizes_df


# ---------------------------------------------------------------------------
# 2. Color & stain-space EDA (H&E specific)
# ---------------------------------------------------------------------------

def _load_rgb(path: str, max_size: int = 512) -> np.ndarray:
    """
    Load an image as a float RGB array in [0, 1], downsized for speed.

    Why downsize: these are 1200x1600 images — running per-pixel color-space
    conversions (HED, HSV) and GLCM texture features on the full resolution
    across hundreds of images in a notebook is slow and usually unnecessary
    for distributional EDA. 512px preserves overall color/texture character
    while keeping this fast enough to iterate on interactively.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_size, max_size))
        return np.asarray(im).astype(np.float64) / 255.0


def plot_rgb_histograms_by_class(df: pd.DataFrame, n_per_class: int = 15,
                                  label_col: str = "label", path_col: str = "path"):
    """
    Average R/G/B channel histograms per class, overlaid.

    Why: H&E staining colors nuclei blue-purple (hematoxylin) and
    cytoplasm/stroma pink (eosin). Differences in tissue density and
    composition between ACA, SCC, and normal tissue often produce visibly
    different raw color distributions even before any morphological
    analysis — e.g. densely packed nuclei (more hematoxylin) shifts the
    blue channel histogram, while keratinized squamous tissue tends to
    show stronger pink/eosin saturation. This is the simplest possible
    'does color alone separate these classes' check.
    """
    classes = sorted(df[label_col].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    channel_names = ["Red", "Green", "Blue"]

    for cls in classes:
        sub = df[df[label_col] == cls].sample(
            min(n_per_class, (df[label_col] == cls).sum()), random_state=0
        )
        imgs = [_load_rgb(p) for p in sub[path_col]]
        stacked = np.stack(imgs)  # (N, H, W, 3) -- thumbnails may differ slightly in size

        for ch in range(3):
            pixels = np.concatenate([im[..., ch].ravel() for im in imgs])
            axes[ch].hist(pixels, bins=50, range=(0, 1), alpha=0.5,
                           label=f"class {cls}", density=True)

    for ch, name in enumerate(channel_names):
        axes[ch].set_title(f"{name} channel")
        axes[ch].set_xlabel("Normalized intensity")
        axes[ch].legend(fontsize=8)
    axes[0].set_ylabel("Density")
    plt.suptitle("RGB channel histograms by class")
    plt.tight_layout()
    plt.show()


def plot_hed_separation_by_class(df: pd.DataFrame, n_per_class: int = 10,
                                  label_col: str = "label", path_col: str = "path"):
    """
    Convert images to HED (Hematoxylin-Eosin-DAB) color space and plot the
    mean Hematoxylin-channel and Eosin-channel intensity distributions per
    class, plus example separated-channel images.

    Why HED instead of raw RGB: RGB channels each mix hematoxylin and eosin
    contributions together (e.g. blue isn't "pure nuclear stain" — it's
    affected by eosin too). The HED color deconvolution model
    (Ruifrok & Johnston) estimates the contribution of each stain
    independently, so the Hematoxylin channel approximates "how much
    nuclear material is here" and the Eosin channel approximates "how much
    cytoplasm/stroma is here" — a much more biologically direct signal than
    raw color for comparing nuclear density across ACA/SCC/normal.
    """
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

    he_df = pd.DataFrame({"label": labels_used, "hematoxylin_mean": h_means, "eosin_mean": e_means})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.boxplot(data=he_df, x="label", y="hematoxylin_mean", ax=axes[0])
    axes[0].set_title("Mean Hematoxylin intensity by class\n(proxy for nuclear density)")
    sns.boxplot(data=he_df, x="label", y="eosin_mean", ax=axes[1])
    axes[1].set_title("Mean Eosin intensity by class\n(proxy for cytoplasm/stroma density)")
    plt.tight_layout()
    plt.show()

    return he_df


def plot_stain_variability_by_patient(df: pd.DataFrame, patient_col: str = "patient_id",
                                       n_per_patient: int = 3, path_col: str = "path"):
    """
    Mean RGB intensity per patient, shown as a scatter/strip plot.

    Why: H&E staining is notoriously inconsistent batch-to-batch and
    scanner-to-scanner — the same tissue type can look noticeably different
    in overall color depending on which slide/lab/scanner produced it. If
    color statistics vary a lot *within* a class across patients, that's a
    strong signal you should apply stain normalization (e.g. Macenko or
    Vahadane) before training, otherwise your model may partly learn to
    recognize patients/batches rather than histology.
    """
    if not _has_col(df, patient_col):
        return None

    rows = []
    for patient, sub in df.groupby(patient_col):
        sample = sub.sample(min(n_per_patient, len(sub)), random_state=0)
        for p in sample[path_col]:
            rgb = _load_rgb(p, max_size=256)
            rows.append({
                "patient_id": patient,
                "mean_R": rgb[..., 0].mean(),
                "mean_G": rgb[..., 1].mean(),
                "mean_B": rgb[..., 2].mean(),
            })
    stain_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 4))
    melted = stain_df.melt(id_vars="patient_id", value_vars=["mean_R", "mean_G", "mean_B"],
                            var_name="channel", value_name="mean_intensity")
    sns.boxplot(data=melted, x="channel", y="mean_intensity", ax=ax)
    sns.stripplot(data=melted, x="channel", y="mean_intensity", color="black",
                  alpha=0.25, size=3, ax=ax)
    ax.set_title("Per-patient mean channel intensity spread\n(wide spread = stain normalization likely needed)")
    plt.tight_layout()
    plt.show()
    return stain_df


def plot_hsv_saturation_by_class(df: pd.DataFrame, n_per_class: int = 15,
                                  label_col: str = "label", path_col: str = "path"):
    """
    Saturation channel (from HSV) distribution per class.

    Why: saturation is a quick proxy for "how much actual tissue vs. white
    background/whitespace" is in an image, and also tracks staining
    intensity/quality. Tissue-sparse images (e.g. mostly glandular lumen or
    background) will show a saturation distribution skewed toward zero.
    Comparing this across classes can reveal if one class systematically
    has more empty/background area (e.g. ACA's glandular lumens vs SCC's
    denser sheets), which is itself a real histological signal but also
    worth knowing about as a potential modeling shortcut.
    """
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
        ax.hist(sats, bins=50, range=(0, 1), alpha=0.5, label=f"class {cls}", density=True)

    ax.set_xlabel("Saturation")
    ax.set_ylabel("Density")
    ax.set_title("HSV saturation distribution by class\n(low saturation ~ background/whitespace)")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 3. Texture & morphology EDA
# ---------------------------------------------------------------------------

def estimate_nuclei_stats(df: pd.DataFrame, n_per_class: int = 10,
                           label_col: str = "label", path_col: str = "path"):
    """
    Rough nuclei count + size estimate per image via Otsu thresholding on
    the Hematoxylin channel, then connected-component analysis.

    Why: this is a crude but fast proxy for nuclear density and
    pleomorphism (variation in nucleus size/shape) without needing a
    trained nucleus-segmentation model. Squamous cell carcinoma and
    poorly-differentiated tumors of either type often show more irregular,
    variably-sized nuclei than well-differentiated or normal tissue — this
    function gives you a quick numeric handle on that pattern.

    Caveat: Otsu thresholding on a single channel will under/over-segment
    touching nuclei. Treat these as relative, comparative statistics across
    classes — not as ground-truth nucleus counts.
    """
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

            rows.append({
                "label": cls,
                "path": p,
                "n_nuclei_est": len(areas),
                "mean_nucleus_area": np.mean(areas) if areas else np.nan,
                "std_nucleus_area": np.std(areas) if areas else np.nan,
            })

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


def compute_glcm_texture_features(df: pd.DataFrame, n_per_class: int = 10,
                                   label_col: str = "label", path_col: str = "path"):
    """
    Gray-Level Co-occurrence Matrix (GLCM) texture features per class:
    contrast, homogeneity, energy, and correlation.

    Why: GLCM features quantify how pixel intensities relate to their
    neighbors — a numeric description of "texture" independent of color.
    Glandular ACA tissue tends to look smoother/more homogeneous in patches
    (open lumens, organized epithelium) while squamous tissue tends to be
    more textured/heterogeneous (keratinization, intercellular bridges,
    denser cell sheets). These four features are the classic Haralick
    texture descriptors and are a standard first pass in histopathology
    image analysis before reaching for learned features.
    """
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
                gray_uint8, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                levels=256, symmetric=True, normed=True,
            )
            rows.append({
                "label": cls,
                "path": p,
                "contrast": graycoprops(glcm, "contrast").mean(),
                "homogeneity": graycoprops(glcm, "homogeneity").mean(),
                "energy": graycoprops(glcm, "energy").mean(),
                "correlation": graycoprops(glcm, "correlation").mean(),
            })

    texture_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, feat in zip(axes, ["contrast", "homogeneity", "energy", "correlation"]):
        sns.boxplot(data=texture_df, x="label", y=feat, ax=ax)
        ax.set_title(feat.capitalize())
    plt.suptitle("GLCM texture features by class")
    plt.tight_layout()
    plt.show()

    return texture_df


def plot_edge_density_by_class(df: pd.DataFrame, n_per_class: int = 10,
                                label_col: str = "label", path_col: str = "path"):
    """
    Sobel edge-magnitude density per class.

    Why: glandular lumens in adenocarcinoma create distinct closed circular
    edges, while squamous architecture tends to produce denser, more
    irregular edge patterns from cell-to-cell borders and keratin pearls.
    Overall edge density (mean Sobel gradient magnitude) is a cheap global
    summary of "how much structural boundary" is visible in an image,
    complementing the GLCM texture features above.
    """
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


# ---------------------------------------------------------------------------
# 4. Dimensionality reduction / embedding visualization
# ---------------------------------------------------------------------------

def extract_vit_embeddings(df: pd.DataFrame, backbone=None, device="cpu",
                            n_per_class: int = 40, label_col: str = "label",
                            path_col: str = "path"):
    """
    Run a (frozen) ViT-B/16 backbone over a sample of images and collect
    the [CLS] token embedding for each.

    Why: this reuses the same frozen-embedding approach as your
    classifier/Grad-CAM pipeline, so the embedding space you visualize here
    is the SAME space your ClassifierHead operates on — directly relevant
    to understanding what the model "sees", rather than an unrelated
    feature space.

    Parameters
    ----------
    backbone : an already-loaded torchvision ViT model (e.g. from
        torchvision.models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)).
        If None, loads the default pretrained weights fresh — convenient
        for EDA, but note this is NOT necessarily the same checkpoint your
        classifier head was trained against if you've since fine-tuned it.

    Returns
    -------
    emb_df : DataFrame with one row per image: the embedding (as a numpy
        array in a single column) plus label/path/any extra metadata
        columns present in df.
    """
    import torch
    from torchvision import transforms
    from torchvision.models import ViT_B_16_Weights
    import torchvision

    if backbone is None:
        backbone = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT, image_size=224)
    backbone.eval().to(device)
    for param in backbone.parameters():
        param.requires_grad_(False)

    tfm = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    classes = sorted(df[label_col].unique())
    extra_cols = [c for c in ["patient_id", "magnification", "differentiation", "class_name"]
                  if c in df.columns]

    rows = []
    with torch.no_grad():
        for cls in classes:
            sub = df[df[label_col] == cls].sample(
                min(n_per_class, (df[label_col] == cls).sum()), random_state=0
            )
            for _, row in sub.iterrows():
                img = Image.open(row[path_col]).convert("RGB")
                x = tfm(img).unsqueeze(0).to(device)

                # Replicate ViT forward up to encoder output, take [CLS] token.
                # Mirrors ViTWithHead.forward() in gradcam.py for consistency.
                feats = backbone._process_input(x)
                cls_tok = backbone.class_token.expand(feats.shape[0], -1, -1)
                feats = torch.cat([cls_tok, feats], dim=1)
                feats = backbone.encoder(feats)
                cls_embedding = feats[:, 0].squeeze(0).cpu().numpy()  # (768,)

                record = {"label": cls, "path": row[path_col], "embedding": cls_embedding}
                for c in extra_cols:
                    record[c] = row[c]
                rows.append(record)

    emb_df = pd.DataFrame(rows)
    print(f"Extracted {len(emb_df)} embeddings of dim {emb_df['embedding'].iloc[0].shape[0]}")
    return emb_df


def plot_embedding_projection(emb_df: pd.DataFrame, method: str = "umap",
                               color_by: str = "label"):
    """
    2D projection (UMAP or t-SNE) of ViT [CLS] embeddings, colored by class
    (or any other metadata column you choose).

    Why: this is the single most informative embedding-space check. If
    classes form visually separated clusters, your classifier has an easy
    job and high accuracy is believable. If clusters instead separate by
    'magnification' or 'patient_id' when you set color_by to those columns,
    that's a strong warning sign the model may be keying off a confound
    rather than genuine histological pattern — re-run this plot with
    color_by='magnification' and color_by='patient_id' as a deliberate
    confound check, not just color_by='label'.

    method: 'umap' (recommended — better preserves both local and some
    global structure, faster) or 'tsne' (classic, purely local structure).
    """
    X = np.stack(emb_df["embedding"].values)

    if method == "umap":
        import umap
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0)
        proj = reducer.fit_transform(X)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        proj = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(X)
    else:
        raise ValueError("method must be 'umap' or 'tsne'")

    plot_df = emb_df.copy()
    plot_df["x"] = proj[:, 0]
    plot_df["y"] = proj[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x="x", y="y", hue=color_by, palette="Set2", s=40, ax=ax)
    ax.set_title(f"ViT [CLS] embedding — {method.upper()} projection, colored by '{color_by}'")
    ax.set_xlabel(f"{method}-1")
    ax.set_ylabel(f"{method}-2")
    plt.tight_layout()
    plt.show()
    return plot_df


def compare_color_vs_embedding_separation(df: pd.DataFrame, emb_df: pd.DataFrame,
                                           label_col: str = "label"):
    """
    Side-by-side UMAP: one on simple RGB-histogram features, one on ViT
    embeddings — both colored by class.

    Why: if a UMAP of crude color histograms ALREADY separates classes
    almost as cleanly as the deep embedding, that's a signal the ViT model
    might largely be exploiting stain-color differences rather than
    learning genuine morphological/structural patterns — worth knowing
    before you over-trust Grad-CAM maps or accuracy numbers. If the deep
    embedding separates classes much more cleanly than raw color, that's
    reassuring evidence the model is using more than just color.
    """
    import umap

    # Build simple per-image RGB histogram feature vectors (32 bins x 3 channels = 96-d)
    color_feats, labels = [], []
    for _, row in df.iterrows():
        if row["label"] not in emb_df["label"].unique():
            continue
        rgb = _load_rgb(row["path"], max_size=128)
        hist = np.concatenate([
            np.histogram(rgb[..., c], bins=32, range=(0, 1), density=True)[0] for c in range(3)
        ])
        color_feats.append(hist)
        labels.append(row["label"])
    color_feats = np.stack(color_feats)

    reducer1 = umap.UMAP(random_state=0)
    color_proj = reducer1.fit_transform(color_feats)

    X = np.stack(emb_df["embedding"].values)
    reducer2 = umap.UMAP(random_state=0)
    emb_proj = reducer2.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(x=color_proj[:, 0], y=color_proj[:, 1], hue=labels, palette="Set2", ax=axes[0], s=30)
    axes[0].set_title("UMAP of raw RGB histogram features\n(color-only signal)")

    sns.scatterplot(x=emb_proj[:, 0], y=emb_proj[:, 1], hue=emb_df[label_col], palette="Set2", ax=axes[1], s=30)
    axes[1].set_title("UMAP of ViT [CLS] embeddings\n(learned deep features)")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 5. Sample mosaics / qualitative grids
# ---------------------------------------------------------------------------

def sample_image_grid(df: pd.DataFrame, n_per_class: int = 6,
                       label_col: str = "label", path_col: str = "path"):
    """
    Grid of randomly sampled images, one row per class.

    Why: always look at your data before computing anything about it. This
    is the fastest way to catch obviously wrong labels, corrupted images,
    or a class that's secretly dominated by one visual pattern (e.g. mostly
    whitespace, or all from one weirdly-stained slide).
    """
    classes = sorted(df[label_col].unique())
    fig, axes = plt.subplots(len(classes), n_per_class, figsize=(2.2 * n_per_class, 2.2 * len(classes)))
    if len(classes) == 1:
        axes = axes[None, :]

    for row, cls in enumerate(classes):
        sub = df[df[label_col] == cls].sample(
            min(n_per_class, (df[label_col] == cls).sum()), random_state=0
        )
        for col, (_, r) in enumerate(sub.iterrows()):
            img = Image.open(r[path_col]).convert("RGB")
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(f"class {cls}", fontsize=10)
        # Fill any remaining empty cells if a class had fewer than n_per_class images
        for col in range(len(sub), n_per_class):
            axes[row, col].axis("off")

    plt.suptitle("Random sample grid by class")
    plt.tight_layout()
    plt.show()


def magnification_comparison_grid(df: pd.DataFrame, label_col: str = "label",
                                   mag_col: str = "magnification", path_col: str = "path"):
    """
    For each class, show one example at 20x and one at 40x side by side
    (where both magnifications exist for that class).

    Why: directly visualizes how much magnification changes the apparent
    cell size/density for the same tissue type — useful context for
    interpreting the magnification-vs-class confound check from
    plot_magnification_by_class(), and for deciding whether your model
    needs magnification-aware handling (e.g. separate models, or
    magnification as an auxiliary input) versus being fine to ignore.
    """
    if not _has_col(df, mag_col):
        return None

    classes = sorted(df[label_col].unique())
    mags = sorted(df[mag_col].unique())

    fig, axes = plt.subplots(len(classes), len(mags), figsize=(4 * len(mags), 4 * len(classes)))
    if len(classes) == 1:
        axes = axes[None, :]
    if len(mags) == 1:
        axes = axes[:, None]

    for row, cls in enumerate(classes):
        for col, mag in enumerate(mags):
            sub = df[(df[label_col] == cls) & (df[mag_col] == mag)]
            ax = axes[row, col]
            if len(sub) == 0:
                ax.axis("off")
                ax.set_title(f"class {cls}, {mag}\n(no images)")
                continue
            r = sub.sample(1, random_state=0).iloc[0]
            img = Image.open(r[path_col]).convert("RGB")
            ax.imshow(img)
            ax.set_title(f"class {cls}, {mag}")
            ax.axis("off")

    plt.suptitle("Magnification comparison by class")
    plt.tight_layout()
    plt.show()


def find_color_outliers(df: pd.DataFrame, label_col: str = "label", path_col: str = "path",
                         n_show: int = 8):
    """
    Within each class, find the images whose mean RGB is furthest (in
    Euclidean distance) from that class's mean RGB, and display them.

    Why: outlier images by raw color are a fast way to flag potential
    mislabels, corrupted scans, or unusually-stained slides worth a manual
    second look before training — much cheaper than waiting to discover
    them as confidently-wrong predictions after a model is already trained.
    """
    classes = sorted(df[label_col].unique())
    outlier_paths = []

    for cls in classes:
        sub = df[df[label_col] == cls].copy()
        means = np.stack([_load_rgb(p, max_size=64).reshape(-1, 3).mean(axis=0) for p in sub[path_col]])
        class_mean = means.mean(axis=0)
        dists = np.linalg.norm(means - class_mean, axis=1)
        sub = sub.assign(color_dist=dists)
        top = sub.nlargest(min(n_show, len(sub)), "color_dist")
        outlier_paths.append(top)

    outliers_df = pd.concat(outlier_paths)

    fig, axes = plt.subplots(len(classes), n_show, figsize=(2.2 * n_show, 2.2 * len(classes)))
    if len(classes) == 1:
        axes = axes[None, :]

    for row, cls in enumerate(classes):
        sub = outliers_df[outliers_df[label_col] == cls]
        for col, (_, r) in enumerate(sub.iterrows()):
            img = Image.open(r[path_col]).convert("RGB")
            axes[row, col].imshow(img)
            axes[row, col].set_title(f"d={r['color_dist']:.3f}", fontsize=8)
            axes[row, col].axis("off")
        for col in range(len(sub), n_show):
            axes[row, col].axis("off")

    plt.suptitle("Color outliers per class\n(images furthest from their class's mean color)")
    plt.tight_layout()
    plt.show()
    return outliers_df
