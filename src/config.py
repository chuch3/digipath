from pathlib import Path

_PROJECT_DIR = Path.cwd().parent

_LUNG_DIR = Path(*[_PROJECT_DIR, "data", "LungHist700"])
_LUNG_IMAGES_DIR = Path(*[_LUNG_DIR, "images"])
_LUNG_METADATA_FILE = Path(*[_LUNG_DIR, "images"])

_SAMPLE_TCGA_DCM_FILE = Path(
    *[
        _LUNG_DIR,
    ]
)
