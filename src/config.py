from pathlib import Path

PROJECT_DIR = Path.cwd().parent

LUNG_DIR = Path(*[PROJECT_DIR, "data", "LungHist700"])
LUNG_IMAGES_DIR = Path(*[LUNG_DIR, "images"])
LUNG_METADATA_FILE = Path(*[LUNG_DIR, "data.csv"])

TRAIN_SPLIT = 0.8
TEST_SPLIT = 0.1
VALID_SPLIT = 0.1
