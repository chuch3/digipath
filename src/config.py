from pathlib import Path

_PROJECT_DIR = Path.cwd().parent

_DATA_LUNG_DIR = Path(*[_PROJECT_DIR, "data", "LungHist700"])

_METADATA_FILE = Path(*[_PROJECT_DIR, ""])

_SAMPLE_TCGA_DCM_FILE = Path(
    *[
        _DATA_LUNG_DIR,
    ]
)
