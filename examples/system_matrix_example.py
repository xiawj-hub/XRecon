"""Small system-matrix usage example.

The system matrix is intended for inspection, debugging, or iterative methods
on small problems. Forward/backprojection should use the native projectors.

Run from the repository root:
    python examples/system_matrix_example.py
"""

from pathlib import Path
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import xrecon


def example_2d() -> None:
    # Same geometry scale as the projector examples, with a smaller image to
    # keep the sparse matrix easy to inspect.
    image_size = 32
    num_det = 32
    angles = torch.linspace(0.0, torch.pi, 30).contiguous()

    projector = xrecon.Projector2d(
        xrecon.Volume2d(image_size, 1.0),
        xrecon.FanBeam2d(
            xrecon.Detector2d(num_det, 2.0),
            iso_source=500.0,
            source_detector=1000.0,
            detector_type="flat",
        ),
    )
    matrix = projector.system_matrix(angles).coalesce()
    image = torch.zeros(1, 1, image_size, image_size)
    image[:, :, 12:20, 12:20] = 1.0

    sino_native = projector.projection(image, angles)
    sino_matrix = torch.sparse.mm(matrix, image.reshape(1, -1).t()).t()
    sino_matrix = sino_matrix.reshape_as(sino_native)
    err = (sino_native - sino_matrix).abs().mean().item()

    print("2D flat system matrix")
    print(f"  shape: {tuple(matrix.shape)}")
    print(f"  nnz  : {matrix._nnz()}")
    print(f"  mean |native - matrix|: {err:.3e}")


def example_3d() -> None:
    # Keep this intentionally tiny: 3D system matrices grow very quickly.
    image_size = 8
    num_det = (8, 4)
    angles = torch.linspace(0.0, torch.pi, 12).contiguous()

    projector = xrecon.Projector3d(
        xrecon.Volume3d(image_size, 1.0),
        xrecon.ConeBeam3d(
            xrecon.Detector3d(num_det, (2.0, 2.0)),
            iso_source=500.0,
            source_detector=1000.0,
            detector_type="flat",
        ),
    )
    matrix = projector.system_matrix(angles).coalesce()
    image = torch.zeros(1, 1, image_size, image_size, image_size)
    image[:, :, 3:5, 3:5, 3:5] = 1.0

    sino_native = projector.projection(image, angles)
    sino_matrix = torch.sparse.mm(matrix, image.reshape(1, -1).t()).t()
    sino_matrix = sino_matrix.reshape_as(sino_native)
    err = (sino_native - sino_matrix).abs().mean().item()

    print("3D flat system matrix")
    print(f"  shape: {tuple(matrix.shape)}")
    print(f"  nnz  : {matrix._nnz()}")
    print(f"  mean |native - matrix|: {err:.3e}")


def main() -> None:
    example_2d()
    example_3d()


if __name__ == "__main__":
    main()
