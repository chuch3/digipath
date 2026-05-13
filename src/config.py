from pathlib import Path

PROJECT_DIR = Path.cwd().parent

LUNG_DIR = Path(*[PROJECT_DIR, "data", "LungHist700"])
LUNG_IMAGES_DIR = Path(*[LUNG_DIR, "images"])
LUNG_METADATA_FILE = Path(*[LUNG_DIR, "data.csv"])
