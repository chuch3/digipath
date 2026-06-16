from pathlib import Path

"""
File path constats are ordered via a tree structure with names ending with its corresponding filetype. 
Each tree are seperate by a linebreak in code. 

For example,

> LUNG_DIR ("data/")
    - LUNG_IMAGES_DIR ("data/images")
    - LUNG_METADATA_FILE ("data/data.csv")

> LUNG_DIR ("preprocess/")
    - LUNG_LOADED_DIR ("data/loaded.csv")

> ...
    - ...

"""

PROJECT_DIR = Path.cwd().parent

LUNG_DIR = Path(*[PROJECT_DIR, "data", "LungHist700"])
LUNG_IMAGES_DIR = Path(*[LUNG_DIR, "images"])
LUNG_METADATA_FILE = Path(*[LUNG_DIR, "data.csv"])

LUNG_PREPROCESS_DIR = Path(*[PROJECT_DIR, "preprocess"])
LUNG_LOADED_FILE = Path(*[LUNG_PREPROCESS_DIR, "loaded.csv"])
EMBEDDING_CACHE = Path(*[LUNG_PREPROCESS_DIR, "lung_embeddings.pt"])

LUNG_MODEL_DIR = Path(*[PROJECT_DIR, "model"])
