"""3D projection and FDK example.

Run from the repository root:
    python examples/projector3d_example.py
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


# A compact cone-beam CT geometry suitable for quick experiments.
IMAGE_SIZE = (256, 256, 10)  # (Nx, Ny, Nz)
NUM_DET = (256, 10)          # (Nu, Nv)
PIX_SIZE = (1.0, 1.0, 1.0)
DET_SIZE = (2.0, 2.0)
ISO_SOURCE = 500.0
SOURCE_DETECTOR = 1000.0
NUM_VIEWS = 360
GEOMETRIES = ("parallel", "flat", "arc")
FILTER_TYPE = "ramp"
OUTPUT_DIR = ROOT / "outputs" / "examples"


def make_image(device: torch.device) -> torch.Tensor:
    image = xrecon.phantom(IMAGE_SIZE[::-1], dim=3).astype(np.float32)
    return torch.from_numpy(image.copy())[None, None].to(device)


def make_projector(scan_type: str, device: torch.device) -> xrecon.Projector3d:
    volume = xrecon.Volume3d(IMAGE_SIZE, PIX_SIZE)
    detector = xrecon.Detector3d(NUM_DET, DET_SIZE)
    if scan_type == "parallel":
        geometry = xrecon.ParallelBeam3d(detector)
    else:
        geometry = xrecon.ConeBeam3d(detector, ISO_SOURCE, SOURCE_DETECTOR, detector_type=scan_type)
    return xrecon.Projector3d(volume, geometry, filter_type=FILTER_TYPE).to(device)


def save_slices(path: Path, volume: torch.Tensor, title: str, vmin: float = -0.15, vmax: float = 0.35) -> None:
    data = volume[0, 0].detach().cpu().numpy()
    z = data.shape[0] // 2
    y = data.shape[1] // 2
    x = data.shape[2] // 2
    panels = [("axial", data[z]), ("coronal", data[:, y, :]), ("sagittal", data[:, :, x])]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), dpi=160, constrained_layout=True)
    fig.suptitle(title)
    for ax, (name, img) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(name)
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
    save_slices(OUTPUT_DIR / "projector3d_input.png", image, "3D phantom")

    for scan_type in GEOMETRIES:
        projector = make_projector(scan_type, device)
        sino = projector.projection(image, angles)
        recon = projector.filtered_backprojection(sino, angles)
        out = OUTPUT_DIR / f"projector3d_{scan_type}.png"
        save_slices(out, recon, f"3D {scan_type} FDK")
        print(f"{scan_type:8s} sino={tuple(sino.shape)} recon={tuple(recon.shape)} -> {out}")


if __name__ == "__main__":
    main()
