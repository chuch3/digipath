"""
Macenko Stain Normalization via SVD

Alternatively, color vector extraction can be done with PCA.

References for Macenko Stain Normalization are used below, many thanks :)

- Reference for overall Macenko algorithm : [link](https://www.geeksforgeeks.org/machine-learning/macenko-method-for-normalizing-histology-slides-for-quantitative-analysis/)

- Reference for Robust Max/Min SVD implementation : [link](https://github.com/m4ln/stain-normalization-reinhard-macenko-vahadane/blob/main/models/stainNorm_Macenko.py)
"""

import os

import numpy as np
from PIL import Image


class MacenkoNormalizer:
    def __init__(self, beta=0.15, alpha=1) -> None:
        self.beta = beta  # OD threshold to in keeping background
        self.alpha = alpha  # percentile for robust max/min
        self._HE_ref = None  # Reference Hematoxylin (H) & Eosin (E) stain vectors
        self._max_C_ref = None  # Reference maximum stain concentration

    @staticmethod
    def _rgb_to_od(img: np.ndarray):
        img = img.astype(np.float32)
        img = np.clip(img, 1, 255)  # Avoid log(0) error
        return -np.log(img / 255)  # Bear-Lambert law

    @staticmethod
    def _od_to_rgb(od: np.ndarray):
        od = np.clip(od, -5, 5)  # Prevent exp overflow
        rgb = np.exp(-od) * 255.0  # Reverse log transform and division by 255
        return np.clip(rgb, 0, 255).astype(np.uint8)  # Convert RGB interval

    def _get_stain_matrix(self, img_od):
        """
        try:
        except (ValueError, np.linalg.LinAlgError):
            return pil_img
        """
        mask = (
            np.sum(img_od, axis=1) > self.beta
        )  # Removing background pixels, as nearly white background has no stain information
        od_hat = img_od[mask]

        # Error fif not enough tissue pixels
        if len(od_hat) < 10:
            raise ValueError("=> ERROR: Not enough pixels for stain matrix estimation")

        # SVD on OD values
        _, _, Vt = np.linalg.svd(od_hat, full_matrices=False)
        plane = Vt[:2, :].T  # Getting the 2 PCs from right singular value in (3, 2)

        # Maps pixel to plane as angles
        phi = np.arctan2(plane[:, 1], plane[:, 0])

        # Setting robust maximum and minimum angles
        min_phi = np.percentile(phi, self.alpha)
        max_phi = np.percentile(phi, 100 - self.alpha)

        # Converting angles to directioons
        v1 = np.array([np.cos(min_phi), np.sin(min_phi)])
        v2 = np.array([np.cos(max_phi), np.sin(max_phi)])

        HE = np.array([Vt[:2].T @ v1, Vt[:2].T @ v2]).T

        # Ensure deterministic ordering of Hematoxylin (H) and Eosin (E) stain
        if HE[0, 0] < HE[1, 0]:
            HE = HE[:, [1, 0]]

        # Normalize stain vectors to unit length
        HE = HE / np.linalg.norm(HE, axis=0)

        # Compute max concentration with least squares from refernence image
        C = np.linalg.lstsq(HE, od_hat.T, rcond=None)[0]
        max_C = np.percentile(C, 99, axis=1)

        return HE, max_C

    def fit(self, img: Image.Image):
        # Image must be array of RGB channels for OD transform
        od = self._rgb_to_od(np.array(img).reshape(-1, 3))
        self._HE_ref, self._max_C_ref = self._get_stain_matrix(od)

    def transform(self, pil_img: Image.Image):
        if self._HE_ref is None:
            raise RuntimeError("Call .fit() with a reference image first")

        img = np.array(pil_img)
        h, w = img.shape[:2]
        od = self._rgb_to_od(img.reshape(-1, 3))

        # Obtaining max concentratoin with least squares from input image
        C = np.linalg.lstsq(self._HE_ref, od.T, rcond=None)[0]
        # Normalize concentrations to reference scale
        C *= (self._max_C_ref / np.percentile(C, 99, axis=1))[:, None]

        # Reconstruct in reference stain space
        od_norm = self._HE_ref @ C

        rgb = self._od_to_rgb(od_norm.T.reshape(h, w, 3))  # Back to RGB shape from H&E
        return Image.fromarray(rgb)


def build_macenko_normalizer(df, n_refs=10):
    samples = df.sample(n=min(n_refs, len(df)), random_state=42)

    best_ref, best_score = None, float("inf")
    target_mean = 128.0  # average color brightness / neutral

    for _, row in samples.iterrows():
        if not os.path.exists(row["path"]):
            continue
        img = np.array(Image.open(row["path"]).convert("RGB"))
        score = abs(img.mean() - target_mean)  # Absolute error from the target mean
        if score < best_score:
            best_score = score
            best_ref = row["path"]

    if not os.path.exists(best_ref):
        print("=> WARNING: reference image not found, skipping stain normalization!!!")
        return None
    print(f"=> Best stain reference image used : {best_ref} (score: {best_score:.2f})")

    normalizer = MacenkoNormalizer()
    normalizer.fit(Image.open(best_ref).convert("RGB"))

    class _StainWrapper:
        def __init__(self, n):
            self._n = n

        def __call__(self, img: Image.Image):
            return self._n.transform(img)

    return _StainWrapper(normalizer)
