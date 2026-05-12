# from openslide import open_slide

import matplotlib.pyplot as plt
from pydicom import dcmread

from config import _SAMPLE_TCGA_DCM_FILE


def tissue_segment():
    pass


def main():
    ds = dcmread(_SAMPLE_TCGA_DCM_FILE)
    plt.imshow(ds.pixel_array)

    plt.show()


if __name__ == "__main__":
    main()
