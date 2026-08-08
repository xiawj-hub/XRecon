"""Mayo/AAPM direct helical FDK-style reconstruction."""

from __future__ import annotations

import os
from pathlib import Path

from xrecon.utils import mayo_fdk


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("MAYO_DATA_DIR", "data/mayo/full_DICOM-CT-PD"))
FILE_PATTERN = "*"
OUTPUT_DIR = ROOT / "outputs" / "mayo_direct_fdk"

FIRST_VIEW = 1000
NUM_VIEWS = 6912
VIEW_STRIDE = 1

IMAGE_SIZE = 512
PIXEL_SIZE = 0.68
NUM_SLICES = 64
SLICE_SPACING = 1.0

HELICAL_VIEW_MODE = "one_turn"
HELICAL_WEIGHT_MODE = "2d"
FILTER_TYPE = "cosine"


def main() -> None:
    mayo_fdk.DATA_DIR = DATA_DIR
    mayo_fdk.FILE_PATTERN = FILE_PATTERN
    mayo_fdk.OUTPUT_DIR = OUTPUT_DIR
    mayo_fdk.ALGORITHM = "direct_fdk"
    mayo_fdk.FIRST_VIEW = FIRST_VIEW
    mayo_fdk.NUM_VIEWS = NUM_VIEWS
    mayo_fdk.VIEW_STRIDE = VIEW_STRIDE
    mayo_fdk.IMAGE_SIZE = IMAGE_SIZE
    mayo_fdk.PIXEL_SIZE = PIXEL_SIZE
    mayo_fdk.NUM_SLICES = NUM_SLICES
    mayo_fdk.SLICE_SPACING = SLICE_SPACING
    mayo_fdk.HELICAL_VIEW_MODE = HELICAL_VIEW_MODE
    mayo_fdk.HELICAL_WEIGHT_MODE = HELICAL_WEIGHT_MODE
    mayo_fdk.FILTER_TYPE = FILTER_TYPE
    mayo_fdk.main()


if __name__ == "__main__":
    main()
