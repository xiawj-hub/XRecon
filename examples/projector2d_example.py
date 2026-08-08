"""2D projection and FBP example.

Run from the repository root:
    python examples/projector2d_example.py
"""

from pathlib import Path
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import xrecon


# A compact fan-beam CT geometry suitable for quick experiments.
IMAGE_SIZE = 256
NUM_DET = 256
PIX_SIZE = 1.0
DET_SIZE = 2.0
ISO_SOURCE = 500.0
SOURCE_DETECTOR = 1000.0
NUM_VIEWS = 720
GEOMETRIES = ("parallel", "flat", "arc")
FILTER_TYPE = "ramp"
OUTPUT_DIR = ROOT / "outputs" / "examples"


def make_image(device: torch.device) -> torch.Tensor:
    image = xrecon.phantom(IMAGE_SIZE, dim=2).astype(np.float32)
    return torch.from_numpy(image)[None, None].to(device)


def make_projector(scan_type: str, device: torch.device) -> xrecon.Projector2d:
    volume = xrecon.Volume2d(IMAGE_SIZE, PIX_SIZE)
    detector = xrecon.Detector2d(NUM_DET, DET_SIZE)
    if scan_type == "parallel":
        geometry = xrecon.ParallelBeam2d(detector)
    else:
        geometry = xrecon.FanBeam2d(detector, ISO_SOURCE, SOURCE_DETECTOR, detector_type=scan_type)
    return xrecon.Projector2d(volume, geometry, filter_type=FILTER_TYPE).to(device)


def save_panel(path: Path, image: torch.Tensor, sino: torch.Tensor, recon: torch.Tensor, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=160, constrained_layout=True)
    axes[0].imshow(image[0, 0].detach().cpu(), cmap="gray", vmin=-0.1, vmax=1.0)
    axes[0].set_title("phantom")
    axes[1].imshow(sino[0, 0].detach().cpu(), cmap="gray", aspect="auto")
    axes[1].set_title("sinogram")
    axes[2].imshow(recon[0, 0].detach().cpu(), cmap="gray", vmin=-0.1, vmax=1.0)
    axes[2].set_title("FBP")
    fig.suptitle(title)
    for ax in axes:
        ax.axis("off")
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image = make_image(device)
    angles = torch.arange(NUM_VIEWS, device=device, dtype=torch.float32) * (2.0 * torch.pi / NUM_VIEWS)

    print(f"device: {device}")
    print(f"image: {tuple(image.shape)}")
    for scan_type in GEOMETRIES:
        projector = make_projector(scan_type, device)
        sino = projector.projection(image, angles)
        recon = projector.filtered_backprojection(sino, angles)
        out = OUTPUT_DIR / f"projector2d_{scan_type}.png"
        save_panel(out, image, sino, recon, f"2D {scan_type}")
        print(f"{scan_type:8s} sino={tuple(sino.shape)} recon={tuple(recon.shape)} -> {out}")


if __name__ == "__main__":
    main()
