# from openslide import open_slide

from pydicom import dcmread

from config import _SAMPLE_TCGA_DCM_FILE


def main():
    pass
    slide = dcmread(_SAMPLE_TCGA_DCM_FILE.as_posix())
    print()


if __name__ == "__main__":
    main()
