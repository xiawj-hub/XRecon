from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

try:
    import pydicom
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pydicom is required: python -m pip install pydicom") from exc

import xrecon


TAG_DET_DU = (0x7029, 0x1002)
TAG_DET_DV = (0x7029, 0x1006)
TAG_DET_TYPE = (0x7029, 0x100B)
TAG_NUM_ROWS = (0x7029, 0x1010)
TAG_NUM_COLS = (0x7029, 0x1011)
TAG_ANGLE = (0x7031, 0x1001)
TAG_SOURCE_Z = (0x7031, 0x1002)
TAG_SOD = (0x7031, 0x1003)
TAG_SDD = (0x7031, 0x1031)
TAG_DET_CENTER = (0x7031, 0x1033)
TAG_SOURCE_SHIFT_Z = (0x7033, 0x100B)
TAG_SOURCE_SHIFT_ANGLE = (0x7033, 0x100C)
TAG_SOURCE_SHIFT_RADIUS = (0x7033, 0x100D)
TAG_VIEWS_PER_ROT = (0x7033, 0x1013)
TAG_SCAN_TRAJECTORY = (0x7037, 0x1009)
TAG_BEAM_TYPE = (0x7037, 0x100A)
TAG_WATER_MU = (0x0018, 0x0061)


@dataclass(frozen=True)
class MayoGeometry:
    num_det_col: int
    num_det_row: int
    det_col_size: float
    det_row_size: float
    iso_source: float
    source_detector: float
    det_center_col: float
    det_center_row: float
    views_per_rotation: int
    detector_type: str
    trajectory: str
    beam_type: str
    water_mu: Optional[float]


def _dicom_files(root: Path, pattern: str) -> list[Path]:
    files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No DICOM files matched {root / pattern}")
    return files


def _raw(ds, tag: Tuple[int, int]) -> bytes:
    value = ds[tag].value
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    return str(value).encode("ascii", errors="ignore")


def _float32(ds, tag: Tuple[int, int], default: Optional[float] = None) -> float:
    if tag not in ds:
        if default is None:
            raise KeyError(tag)
        return float(default)
    raw = _raw(ds, tag)
    if len(raw) == 4:
        return float(np.frombuffer(raw, dtype="<f4", count=1)[0])
    return float(ds[tag].value)


def _float32_pair(ds, tag: Tuple[int, int]) -> Tuple[float, float]:
    raw = _raw(ds, tag)
    values = np.frombuffer(raw, dtype="<f4", count=2)
    if values.size != 2:
        raise ValueError(f"Expected two float32 values in tag {tag}")
    return float(values[0]), float(values[1])


def _int16(ds, tag: Tuple[int, int]) -> int:
    raw = _raw(ds, tag)
    if len(raw) == 2:
        return int(np.frombuffer(raw, dtype="<i2", count=1)[0])
    return int(ds[tag].value)


def _text(ds, tag: Tuple[int, int], default: str = "") -> str:
    if tag not in ds:
        return default
    return _raw(ds, tag).rstrip(b"\x00 ").decode("ascii", errors="ignore").strip().lower()


def read_mayo_geometry(first_file: Path) -> MayoGeometry:
    ds = pydicom.dcmread(str(first_file), stop_before_pixels=True)
    det_center_col, det_center_row = _float32_pair(ds, TAG_DET_CENTER)
    water_mu = _float32(ds, TAG_WATER_MU, default=np.nan)
    if not np.isfinite(water_mu) or water_mu <= 0:
        water_mu = None
    return MayoGeometry(
        num_det_col=_int16(ds, TAG_NUM_COLS),
        num_det_row=_int16(ds, TAG_NUM_ROWS),
        det_col_size=_float32(ds, TAG_DET_DU),
        det_row_size=_float32(ds, TAG_DET_DV),
        iso_source=_float32(ds, TAG_SOD),
        source_detector=_float32(ds, TAG_SDD),
        det_center_col=det_center_col,
        det_center_row=det_center_row,
        views_per_rotation=_int16(ds, TAG_VIEWS_PER_ROT),
        detector_type=_text(ds, TAG_DET_TYPE, "cylindrical"),
        trajectory=_text(ds, TAG_SCAN_TRAJECTORY, "helical"),
        beam_type=_text(ds, TAG_BEAM_TYPE, "fanbeam"),
        water_mu=water_mu,
    )


def choose_files(files: Sequence[Path], first_view: int, num_views: int, view_stride: int) -> list[Path]:
    if first_view < 1:
        raise ValueError("first-view is one-based and should be >= 1")
    start = first_view - 1
    stop = len(files) if num_views <= 0 else min(len(files), start + num_views * view_stride)
    chosen = list(files[start:stop:view_stride])
    if not chosen:
        raise ValueError("No views selected")
    return chosen


def detector_shift(center: float, count: int, stride: int, origin: float, start: int = 0) -> float:
    new_center = (center - origin - start) / stride
    default_center = count / 2.0 - 0.5
    return -(new_center - default_center)


def load_projection_stack(
    files: Sequence[Path],
    det_col_stride: int,
    det_row_stride: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sinos = []
    angles = np.empty(len(files), dtype=np.float32)
    source_z = np.empty(len(files), dtype=np.float32)
    source_shift_z = np.empty(len(files), dtype=np.float32)
    source_shift_angle = np.empty(len(files), dtype=np.float32)
    source_shift_radius = np.empty(len(files), dtype=np.float32)
    for i, path in enumerate(files):
        ds = pydicom.dcmread(str(path), stop_before_pixels=False)
        raw = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        projection = raw * slope + intercept
        projection = projection[::det_col_stride, ::det_row_stride].T
        sinos.append(np.ascontiguousarray(projection))
        angles[i] = _float32(ds, TAG_ANGLE)
        source_z[i] = _float32(ds, TAG_SOURCE_Z)
        source_shift_z[i] = _float32(ds, TAG_SOURCE_SHIFT_Z, default=0.0)
        source_shift_angle[i] = _float32(ds, TAG_SOURCE_SHIFT_ANGLE, default=0.0)
        source_shift_radius[i] = _float32(ds, TAG_SOURCE_SHIFT_RADIUS, default=0.0)
    sino = np.stack(sinos, axis=0)
    return sino, angles, source_z, source_shift_z, source_shift_angle, source_shift_radius


def normalize_angles(angles: np.ndarray) -> np.ndarray:
    unwrapped = np.unwrap(angles.astype(np.float64)).astype(np.float32)
    return np.ascontiguousarray(unwrapped)


def order_views(
    sino: np.ndarray,
    angles: np.ndarray,
    source_z: np.ndarray,
    source_shift_z: np.ndarray,
    source_shift_angle: np.ndarray,
    source_shift_radius: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mode = mode.lower()
    if mode == "dicom":
        return sino, angles, source_z, source_shift_z, source_shift_angle, source_shift_radius
    if angles.size < 2:
        return sino, angles, source_z, source_shift_z, source_shift_angle, source_shift_radius
    decreasing = angles[-1] < angles[0]
    if (mode == "increasing" and decreasing) or (mode == "decreasing" and not decreasing):
        return (
            np.ascontiguousarray(sino[::-1]),
            np.ascontiguousarray(angles[::-1]),
            np.ascontiguousarray(source_z[::-1]),
            np.ascontiguousarray(source_shift_z[::-1]),
            np.ascontiguousarray(source_shift_angle[::-1]),
            np.ascontiguousarray(source_shift_radius[::-1]),
        )
    return sino, angles, source_z, source_shift_z, source_shift_angle, source_shift_radius


def volume_z_center(source_z: np.ndarray, mode: str, value: Optional[float]) -> float:
    if value is not None:
        return float(value)
    mode = mode.lower()
    if mode == "mean":
        return float(source_z.mean())
    if mode == "middle":
        return float(source_z[len(source_z) // 2])
    if mode == "first":
        return float(source_z[0])
    raise ValueError(f"Unsupported z-center mode: {mode}")


def to_hu(recon: np.ndarray, water_mu: Optional[float], force_raw: bool) -> np.ndarray:
    if force_raw or water_mu is None or water_mu <= 0:
        return recon
    return 1000.0 * (recon / water_mu - 1.0)


def save_preview(volume: np.ndarray, out_png: Path, vmin: float, vmax: float, title: str) -> None:
    import matplotlib.pyplot as plt

    d, h, w = volume.shape
    slices = [d // 4, d // 2, min(d - 1, 3 * d // 4)]
    coronal = volume[d // 2]
    sagittal = volume[:, :, w // 2]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for ax, idx in zip(axes[0], slices):
        ax.imshow(volume[idx], cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"z slice {idx}")
        ax.axis("off")
    axes[1, 0].imshow(coronal, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("axial center")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(sagittal, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
    axes[1, 1].set_title("sagittal center")
    axes[1, 1].axis("off")
    axes[1, 2].hist(volume[np.isfinite(volume)].ravel(), bins=256, range=(vmin, vmax))
    axes[1, 2].set_title(title)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def save_auto_preview(volume: np.ndarray, out_png: Path) -> None:
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        finite = np.array([0.0], dtype=np.float32)
    vmin, vmax = np.percentile(finite, [1.0, 99.8])
    if vmin == vmax:
        vmax = vmin + 1.0
    save_preview(volume, out_png, float(vmin), float(vmax), f"auto [{vmin:.4g}, {vmax:.4g}]")


def save_projection_preview(sino: np.ndarray, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    views = [0, len(sino) // 2, len(sino) - 1]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, idx in zip(axes, views):
        ax.imshow(sino[idx], cmap="gray", vmin=0.0, vmax=6.0, aspect="auto")
        ax.set_title(f"view {idx}")
        ax.axis("off")
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def save_metadata_csv(path: Path, rows: Iterable[Tuple[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        writer.writerows(rows)


def make_projector(args, geometry: MayoGeometry, nu: int, nv: int, binshift: Tuple[float, float]):
    image_size = (args.image_size, args.image_size, args.num_slices)
    pix_size = (args.pixel_size, args.pixel_size, args.slice_spacing)
    det_size = (
        geometry.det_col_size * args.det_col_stride,
        geometry.det_row_size * args.det_row_stride,
    )
    return xrecon.HelicalProjector3d(
        image_size,
        (nu, nv),
        pix_size,
        det_size,
        geometry.iso_source,
        geometry.source_detector,
        pixshift=(0.0, 0.0, args.volume_z_shift / args.slice_spacing),
        binshift=binshift,
        scan_type="arc",
        pitch=0.0,
        z0=0.0,
        filter_type=args.filter,
    )

