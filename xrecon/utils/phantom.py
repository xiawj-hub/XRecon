from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

SizeLike = Union[int, Tuple[int, ...], List[int]]


def phantom(
    n: SizeLike = 256,
    p_type: str = "Modified Shepp-Logan",
    ellipses: Union[List, np.ndarray, None] = None,
    dim: int = 2,
    zlims: Union[Tuple[float, float], List[float]] = (-1.0, 1.0),
) -> np.ndarray:
    if np.isscalar(n):
        shape = tuple([int(n)] * dim)
    else:
        shape = tuple(int(v) for v in n)

    if ellipses is None:
        ellipses = _select_phantom(p_type, dim)

    if dim == 2:
        return create_phantom2d(shape, ellipses)
    if dim == 3:
        return create_phantom3d(shape, ellipses, zlims)
    raise ValueError(f"Unexpected dim: {dim}")


def create_phantom2d(shape, ellipses) -> np.ndarray:
    if len(shape) != 2:
        raise ValueError("2D phantom shape should have 2 dimensions")
    ellipses = np.asarray(ellipses, dtype=np.float32)
    if ellipses.ndim != 2 or ellipses.shape[1] != 6:
        raise ValueError("2D phantom ellipses should have shape [N, 6]")

    image = np.zeros(shape, dtype=np.float32)
    rows, cols = shape
    ygrid, xgrid = np.mgrid[1:-1:(1j * rows), -1:1:(1j * cols)]

    for intensity, a, b, x0, y0, phi_deg in ellipses:
        phi = phi_deg * np.pi / 180.0
        x = xgrid - x0
        y = ygrid - y0
        cos_p = np.cos(phi)
        sin_p = np.sin(phi)
        inside = (((x * cos_p + y * sin_p) ** 2) / (a * a) + ((y * cos_p - x * sin_p) ** 2) / (b * b)) <= 1.0
        image[inside] += intensity

    return image


def create_phantom3d(shape, ellipses, zlims=(-1.0, 1.0)) -> np.ndarray:
    if len(shape) != 3:
        raise ValueError("3D phantom shape should have 3 dimensions")
    ellipses = np.asarray(ellipses, dtype=np.float32)
    if ellipses.ndim != 2 or ellipses.shape[1] != 8:
        raise ValueError("3D phantom ellipses should have shape [N, 8]")
    if len(zlims) != 2 or zlims[0] > zlims[1]:
        raise ValueError("zlims should be (lower, upper)")

    volume = np.zeros(shape, dtype=np.float32)
    rows, cols, slices = shape
    ygrid, xgrid, zgrid = np.mgrid[1:-1:(1j * rows), -1:1:(1j * cols), zlims[0]:zlims[1]:(1j * slices)]

    for intensity, a, b, c, x0, y0, z0, phi_deg in ellipses:
        phi = phi_deg * np.pi / 180.0
        x = xgrid - x0
        y = ygrid - y0
        z = zgrid - z0
        cos_p = np.cos(phi)
        sin_p = np.sin(phi)
        inside = (
            ((x * cos_p + y * sin_p) ** 2) / (a * a)
            + ((y * cos_p - x * sin_p) ** 2) / (b * b)
            + (z * z) / (c * c)
        ) <= 1.0
        volume[inside] += intensity

    return volume


def _select_phantom(name: str, dim: int):
    key = name.lower().strip()
    if dim == 2:
        if key == "shepp-logan":
            return _shepp_logan()
        if key == "modified shepp-logan":
            return _mod_shepp_logan()
    elif dim == 3:
        if key == "shepp-logan":
            return _shepp_logan_3d()
        if key == "modified shepp-logan":
            return _mod_shepp_logan_3d()
    raise ValueError(f"Unknown phantom type '{name}' for dim={dim}")


def _shepp_logan():
    return [
        [2, 0.69, 0.92, 0, 0, 0],
        [-0.98, 0.6624, 0.8740, 0, -0.0184, 0],
        [-0.02, 0.1100, 0.3100, 0.22, 0, -18],
        [-0.02, 0.1600, 0.4100, -0.22, 0, 18],
        [0.01, 0.2100, 0.2500, 0, 0.35, 0],
        [0.01, 0.0460, 0.0460, 0, 0.1, 0],
        [0.02, 0.0460, 0.0460, 0, -0.1, 0],
        [0.01, 0.0460, 0.0230, -0.08, -0.605, 0],
        [0.01, 0.0230, 0.0230, 0, -0.606, 0],
        [0.01, 0.0230, 0.0460, 0.06, -0.605, 0],
    ]


def _mod_shepp_logan():
    return [
        [1, 0.69, 0.92, 0, 0, 0],
        [-0.80, 0.6624, 0.8740, 0, -0.0184, 0],
        [-0.20, 0.1100, 0.3100, 0.22, 0, -18],
        [-0.20, 0.1600, 0.4100, -0.22, 0, 18],
        [0.10, 0.2100, 0.2500, 0, 0.35, 0],
        [0.10, 0.0460, 0.0460, 0, 0.1, 0],
        [0.10, 0.0460, 0.0460, 0, -0.1, 0],
        [0.10, 0.0460, 0.0230, -0.08, -0.605, 0],
        [0.10, 0.0230, 0.0230, 0, -0.606, 0],
        [0.10, 0.0230, 0.0460, 0.06, -0.605, 0],
    ]


def _shepp_logan_3d():
    return [
        [2, 0.69, 0.92, 0.9, 0, 0, 0, 0],
        [-0.80, 0.6624, 0.8740, 0.8800, 0, 0, 0, 0],
        [-0.20, 0.4100, 0.1600, 0.2100, -0.22, 0, -0.25, 108],
        [-0.20, 0.3100, 0.1100, 0.2200, 0.22, 0, -0.25, 72],
        [0.20, 0.2100, 0.2500, 0.5000, 0, 0.35, -0.25, 0],
        [0.20, 0.0460, 0.0460, 0.0460, 0, 0.10, -0.25, 0],
        [0.10, 0.0460, 0.0230, 0.0200, -0.08, -0.65, -0.25, 0],
        [0.10, 0.0460, 0.0230, 0.0200, 0.06, -0.65, -0.25, 90],
        [0.20, 0.0560, 0.0400, 0.1000, 0.06, -0.105, 0.625, 90],
        [-0.20, 0.0560, 0.0560, 0.1000, 0, 0.1, 0.625, 0],
    ]


def _mod_shepp_logan_3d():
    return [
        [1, 0.69, 0.92, 0.9, 0, 0, 0, 0],
        [-0.80, 0.6624, 0.8740, 0.8800, 0, 0, 0, 0],
        [-0.20, 0.4100, 0.1600, 0.2100, -0.22, 0, -0.25, 108],
        [-0.20, 0.3100, 0.1100, 0.2200, 0.22, 0, -0.25, 72],
        [0.10, 0.2100, 0.2500, 0.5000, 0, 0.35, -0.25, 0],
        [0.10, 0.0460, 0.0460, 0.0460, 0, 0.10, -0.25, 0],
        [0.10, 0.0460, 0.0230, 0.0200, -0.08, -0.65, -0.25, 0],
        [0.10, 0.0460, 0.0230, 0.0200, 0.06, -0.65, -0.25, 90],
        [0.10, 0.0560, 0.0400, 0.1000, 0.06, -0.105, 0.625, 90],
        [-0.10, 0.0560, 0.0560, 0.1000, 0, 0.1, 0.625, 0],
    ]
