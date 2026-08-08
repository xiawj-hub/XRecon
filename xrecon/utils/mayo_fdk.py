from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]

import xrecon
from xrecon.utils import mayo


# =========================
# Edit parameters here.
# =========================

DATA_DIR = Path(os.environ.get("MAYO_DATA_DIR", "data/mayo/full_DICOM-CT-PD"))
FILE_PATTERN = "*"
OUTPUT_DIR = REPO_ROOT / "outputs" / "mayo_fdk"

ALGORITHM = "slice_fan_arc_fbp"
# Options:
#   "direct_fdk"          : direct helical weighted FDK-like backprojection.
#   "slice_parallel_fbp"  : single-slice rebin to parallel + 2D FBP.
#   "slice_fan_arc_fbp"   : single-slice rebin to fan-beam arc + 2D FBP.
#   "slice_fan_flat_fbp"  : single-slice rebin to fan-beam flat + 2D FBP.
#   "slice_all"           : run the three single-slice rebin routes.
#   "cone_rebin_fdk"      : rebin helical data to circular cone-beam + 3D FDK.
#   "all"                 : run all paths.

FIRST_VIEW = 1000
NUM_VIEWS = 6912         # 2304 = one rotation, 4608 = two, 6912 = three.
VIEW_STRIDE = 1
ANGLE_ORDER = "increasing"

IMAGE_SIZE = 512
PIXEL_SIZE = 0.68        # mm
NUM_SLICES = 64
SLICE_SPACING = 1.0      # mm
VOLUME_Z_SHIFT = 0.0     # mm
Z_CENTER_MODE = "mean"
Z_CENTER_VALUE = None

DET_COL_STRIDE = 1
DET_ROW_STRIDE = 1
DETECTOR_INDEX_ORIGIN = 1.0
BINSHIFT_U = -1.625
BINSHIFT_V = -0.5
FLIP_DET_COLS = False
FLIP_DET_ROWS = True

USE_SOURCE_SHIFTS = False
SMOOTH_SOURCE_Z_FOR_FDK = True
DEVICE = "cuda"
FILTER_TYPE = "cosine"
DIRECT_FDK_SCALE = 1.0
HELICAL_VIEW_MODE = "one_turn"  # "auto", "half_scan", "one_turn", or "fixed".
HELICAL_VIEW_TURNS = 1.0        # Used only when HELICAL_VIEW_MODE = "fixed".
HELICAL_HALFSCAN_COEF = 1.0
HELICAL_WEIGHT_MODE = "2d"      # "tang3d" or "2d".
HELICAL_TANG_K = 5.0
HELICAL_TANG_BLEND = 0.0

# Slice rebin uses one local rotation per slice by default.
SLICE_REBIN_VIEWS = 2304
REBIN_DET_COUNT = None       # None keeps Nu.
REBIN_DET_SPACING = None     # None maps fan coverage to a uniform t grid.
REBIN_ROW_SCALE = "sdd/sod"  # "sdd/sod" or "1".

# Circular cone-beam rebin for actual 3D FDK.
CONE_REBIN_CENTER_Z = 0.0
CONE_REBIN_NUM_VIEWS = 2304
CONE_REBIN_START = "nearest"  # "nearest" or an integer start view.

WATER_MU = 0.0192
DISPLAY_RAW = False
WINDOW_MIN = -160.0
WINDOW_MAX = 240.0


def make_source_pos(
    source_z: np.ndarray,
    source_shift_z: np.ndarray,
    source_shift_angle: np.ndarray,
    source_shift_radius: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    source_pos = torch.zeros((4, source_z.size), device=device, dtype=torch.float32)
    source_pos[0] = torch.from_numpy(source_z).to(device=device, dtype=torch.float32)
    source_pos[1] = torch.from_numpy(source_shift_radius).to(device=device, dtype=torch.float32)
    source_pos[2] = torch.from_numpy(source_shift_angle).to(device=device, dtype=torch.float32)
    source_pos[3] = torch.from_numpy(source_shift_z).to(device=device, dtype=torch.float32)
    return source_pos.contiguous()


def load_mayo_problem():
    files = mayo._dicom_files(DATA_DIR, FILE_PATTERN)
    geometry = mayo.read_mayo_geometry(files[0])
    chosen = mayo.choose_files(files, FIRST_VIEW, NUM_VIEWS, VIEW_STRIDE)

    sino, angles, source_z, shift_z, shift_angle, shift_radius = mayo.load_projection_stack(
        chosen, DET_COL_STRIDE, DET_ROW_STRIDE
    )
    angles = mayo.normalize_angles(angles)
    reverse_files = False
    if ANGLE_ORDER != "dicom" and angles.size > 1:
        decreasing = angles[-1] < angles[0]
        reverse_files = (ANGLE_ORDER == "increasing" and decreasing) or (
            ANGLE_ORDER == "decreasing" and not decreasing
        )
    sino, angles, source_z, shift_z, shift_angle, shift_radius = mayo.order_views(
        sino, angles, source_z, shift_z, shift_angle, shift_radius, ANGLE_ORDER
    )
    if reverse_files:
        chosen = list(reversed(chosen))

    if FLIP_DET_COLS:
        sino = np.ascontiguousarray(sino[:, :, ::-1])
    if FLIP_DET_ROWS:
        sino = np.ascontiguousarray(sino[:, ::-1, :])
    if not USE_SOURCE_SHIFTS:
        shift_z.fill(0.0)
        shift_angle.fill(0.0)
        shift_radius.fill(0.0)

    z_center = mayo.volume_z_center(source_z, Z_CENTER_MODE, Z_CENTER_VALUE)
    source_z = np.ascontiguousarray(source_z - z_center)
    if SMOOTH_SOURCE_Z_FOR_FDK and source_z.size > 1:
        slope, intercept = np.polyfit(angles.astype(np.float64), source_z.astype(np.float64), 1)
        source_z = np.ascontiguousarray((slope * angles + intercept).astype(np.float32))

    nv, nu = sino.shape[1:]
    binshift_u = mayo.detector_shift(geometry.det_center_col, nu, DET_COL_STRIDE, DETECTOR_INDEX_ORIGIN)
    binshift_v = mayo.detector_shift(geometry.det_center_row, nv, DET_ROW_STRIDE, DETECTOR_INDEX_ORIGIN)
    if BINSHIFT_U is not None:
        binshift_u = float(BINSHIFT_U)
    if BINSHIFT_V is not None:
        binshift_v = float(BINSHIFT_V)

    return geometry, chosen, sino, angles, source_z, shift_z, shift_angle, shift_radius, (binshift_u, binshift_v)


def make_helical_projector(geometry, sino_shape, binshift, device: torch.device) -> xrecon.HelicalProjector3d:
    _, nv, nu = sino_shape
    return xrecon.HelicalProjector3d(
        (IMAGE_SIZE, IMAGE_SIZE, NUM_SLICES),
        (nu, nv),
        (PIXEL_SIZE, PIXEL_SIZE, SLICE_SPACING),
        (geometry.det_col_size * DET_COL_STRIDE, geometry.det_row_size * DET_ROW_STRIDE),
        geometry.iso_source,
        geometry.source_detector,
        pixshift=(0.0, 0.0, VOLUME_Z_SHIFT / SLICE_SPACING),
        binshift=binshift,
        scan_type="arc",
        filter_type=FILTER_TYPE,
        helical_view_mode=HELICAL_VIEW_MODE,
        helical_view_turns=HELICAL_VIEW_TURNS,
        helical_halfscan_coef=HELICAL_HALFSCAN_COEF,
        helical_weight_mode=HELICAL_WEIGHT_MODE,
        helical_tang_k=HELICAL_TANG_K,
        helical_tang_blend=HELICAL_TANG_BLEND,
    ).to(device)


def direct_fdk(projector, sino: torch.Tensor, angles: torch.Tensor, source_pos: torch.Tensor) -> torch.Tensor:
    recon = projector.filtered_backprojection(sino, angles, source_pos, filter_type=FILTER_TYPE)
    return recon * DIRECT_FDK_SCALE


def interp_source_z_for_parallel_bins(
    angles: torch.Tensor,
    source_z: torch.Tensor,
    theta: torch.Tensor,
    t: torch.Tensor,
    iso_source: float,
) -> torch.Tensor:
    gamma = torch.asin(torch.clamp(t / float(iso_source), -1.0 + 1.0e-6, 1.0 - 1.0e-6))
    beta = theta[:, None] - gamma[None, :]
    if angles.numel() < 2:
        return source_z[0].expand_as(beta)

    step = torch.diff(angles).mean()
    idx = (beta - angles[0]) / step
    idx0 = torch.floor(idx).to(torch.long)
    idx1 = idx0 + 1
    frac = (idx - idx0.to(idx.dtype)).clamp(0.0, 1.0)
    valid = (idx0 >= 0) & (idx1 < source_z.numel())
    idx0 = idx0.clamp(0, source_z.numel() - 1)
    idx1 = idx1.clamp(0, source_z.numel() - 1)
    z = (1.0 - frac) * source_z[idx0] + frac * source_z[idx1]
    return torch.where(valid, z, torch.zeros_like(z))


def sample_rows_for_slice(
    sino: torch.Tensor,
    slice_z: float,
    source_z_grid: torch.Tensor,
    geometry,
    binshift_v: float,
) -> torch.Tensor:
    b, c, views, nv, nu = sino.shape
    dv = float(geometry.det_row_size * DET_ROW_STRIDE)
    shift_v = float(binshift_v) * dv
    if REBIN_ROW_SCALE == "sdd/sod":
        z_to_detector = float(geometry.source_detector / geometry.iso_source)
    else:
        z_to_detector = 1.0

    if source_z_grid.ndim == 1:
        source_z_grid = source_z_grid[:, None].expand(views, nu)
    row_index = ((float(slice_z) - source_z_grid) * z_to_detector - shift_v) / dv + nv / 2.0 - 0.5
    col_index = torch.arange(nu, device=sino.device, dtype=sino.dtype).view(1, nu).expand(views, nu)
    grid_x = 2.0 * col_index / max(nu - 1, 1) - 1.0
    grid_y = 2.0 * row_index / max(nv - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).view(1, views, nu, 2)

    src = sino.reshape(b * c * views, 1, nv, nu)
    grid = grid.expand(b * c, views, nu, 2).reshape(b * c * views, 1, nu, 2)
    out = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return out.view(b, c, views, nu).contiguous()


def arc_fan_to_flat_fan(
    sino_arc: torch.Tensor,
    geometry,
    binshift_u: float,
) -> tuple[torch.Tensor, float, float]:
    b, c, views, nu = sino_arc.shape
    du_arc = float(geometry.det_col_size * DET_COL_STRIDE / geometry.source_detector)
    shift_arc = float(binshift_u) * du_arc

    u_arc = (torch.arange(nu, device=sino_arc.device, dtype=sino_arc.dtype) - nu / 2.0 + 0.5) * du_arc + shift_arc
    max_s = float(geometry.source_detector) * torch.tan(u_arc.abs().max())
    ds_flat = 2.0 * float(max_s) / nu
    s_flat = (torch.arange(nu, device=sino_arc.device, dtype=sino_arc.dtype) - nu / 2.0 + 0.5) * ds_flat
    u_query = torch.atan2(s_flat, s_flat.new_tensor(float(geometry.source_detector)))
    col_index = (u_query - shift_arc) / du_arc + nu / 2.0 - 0.5

    grid_x = 2.0 * col_index / max(nu - 1, 1) - 1.0
    grid_y = torch.zeros_like(grid_x)
    grid = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, nu, 2).expand(b * c * views, 1, nu, 2)
    src = sino_arc.reshape(b * c * views, 1, 1, nu)
    out = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return out.view(b, c, views, nu).contiguous(), float(ds_flat), 0.0


def local_view_indices(source_z: torch.Tensor, slice_z: float, views_per_slice: int) -> torch.Tensor:
    count = min(int(views_per_slice), source_z.numel())
    center = int(torch.argmin((source_z - float(slice_z)).abs()).item())
    start = max(0, min(source_z.numel() - count, center - count // 2))
    return torch.arange(start, start + count, device=source_z.device)


def slice_rebin_recon(
    sino: torch.Tensor,
    angles: torch.Tensor,
    source_z: torch.Tensor,
    geometry,
    binshift: tuple[float, float],
    device: torch.device,
    target: str,
) -> torch.Tensor:
    recon_slices = []
    z0 = -(NUM_SLICES / 2.0 - 0.5) * SLICE_SPACING + VOLUME_Z_SHIFT
    slice_zs = [z0 + iz * SLICE_SPACING for iz in range(NUM_SLICES)]

    det_col_size = float(geometry.det_col_size * DET_COL_STRIDE)
    du_arc = det_col_size / float(geometry.source_detector)
    shift_u_arc = float(binshift[0]) * du_arc

    for iz, slice_z in enumerate(slice_zs):
        idx = local_view_indices(source_z, slice_z, SLICE_REBIN_VIEWS)
        sino_i = sino[:, :, idx, :, :].contiguous()
        angles_i = angles[idx].contiguous()
        source_z_i = source_z[idx].contiguous()

        if target == "parallel":
            rebinned, theta, t, _ = xrecon.helical_cone_to_parallel_rebin(
                sino_i,
                angles_i,
                source_z_i,
                geometry.iso_source,
                du_arc,
                det_shift=shift_u_arc,
                detector_type="arc",
                output_det_count=REBIN_DET_COUNT,
                output_det_spacing=REBIN_DET_SPACING,
            )
            source_z_grid = interp_source_z_for_parallel_bins(angles_i, source_z_i, theta, t, geometry.iso_source)
            slice_sino = sample_rows_for_slice(rebinned, slice_z, source_z_grid, geometry, binshift[1])
            projector2d = xrecon.Projector2d(
                IMAGE_SIZE,
                slice_sino.shape[-1],
                (PIXEL_SIZE, PIXEL_SIZE),
                float(t[1] - t[0]) if t.numel() > 1 else PIXEL_SIZE,
                scan_type="parallel",
                filter_type=FILTER_TYPE,
            ).to(device)
            recon_angles = theta
        else:
            slice_sino = sample_rows_for_slice(sino_i, slice_z, source_z_i, geometry, binshift[1])
            if target == "fan_arc":
                det_size = det_col_size
                scan_type = "arc"
                det_binshift = binshift[0]
            elif target == "fan_flat":
                slice_sino, det_size, det_binshift = arc_fan_to_flat_fan(slice_sino, geometry, binshift[0])
                scan_type = "flat"
            else:
                raise ValueError(f"Unsupported slice rebin target: {target}")
            projector2d = xrecon.Projector2d(
                IMAGE_SIZE,
                slice_sino.shape[-1],
                (PIXEL_SIZE, PIXEL_SIZE),
                det_size,
                geometry.iso_source,
                geometry.source_detector,
                binshift=det_binshift,
                scan_type=scan_type,
                filter_type=FILTER_TYPE,
            ).to(device)
            recon_angles = angles_i
        image2d = projector2d.filtered_backprojection(slice_sino, recon_angles, filter_type=FILTER_TYPE)
        recon_slices.append(image2d)
        print(f"slice {target} FBP {iz + 1:03d}/{NUM_SLICES}: z={slice_z:.3f} mm, views={idx.numel()}")

    return torch.stack(recon_slices, dim=2).contiguous()


def choose_cone_rebin_indices(source_z: torch.Tensor) -> torch.Tensor:
    count = min(int(CONE_REBIN_NUM_VIEWS), source_z.numel())
    if CONE_REBIN_START == "nearest":
        center = int(torch.argmin((source_z - float(CONE_REBIN_CENTER_Z)).abs()).item())
        start = max(0, min(source_z.numel() - count, center - count // 2))
    else:
        start = int(CONE_REBIN_START)
        start = max(0, min(source_z.numel() - count, start))
    return torch.arange(start, start + count, device=source_z.device)


def cone_rebin_fdk(
    sino: torch.Tensor,
    angles: torch.Tensor,
    source_z: torch.Tensor,
    geometry,
    binshift: tuple[float, float],
    device: torch.device,
) -> torch.Tensor:
    idx = choose_cone_rebin_indices(source_z)
    sino_i = sino[:, :, idx, :, :].contiguous()
    angles_i = angles[idx].contiguous()
    source_z_i = source_z[idx].contiguous()

    b, c, views, nv, nu = sino_i.shape
    dv = float(geometry.det_row_size * DET_ROW_STRIDE)
    shift_v = float(binshift[1]) * dv
    if REBIN_ROW_SCALE == "sdd/sod":
        z_to_detector = float(geometry.source_detector / geometry.iso_source)
    else:
        z_to_detector = 1.0

    det_rows = (torch.arange(nv, device=device, dtype=sino.dtype) - nv / 2.0 + 0.5) * dv + shift_v
    row_query = ((det_rows[None, :] + float(CONE_REBIN_CENTER_Z) - source_z_i[:, None]) * z_to_detector - shift_v) / dv
    row_query = row_query + nv / 2.0 - 0.5
    col_index = torch.arange(nu, device=device, dtype=sino.dtype).view(1, 1, nu).expand(views, nv, nu)
    row_index = row_query[:, :, None].expand(views, nv, nu)
    grid_x = 2.0 * col_index / max(nu - 1, 1) - 1.0
    grid_y = 2.0 * row_index / max(nv - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).reshape(1, views * nv, nu, 2)

    src = sino_i.reshape(b * c * views, 1, nv, nu)
    # One grid per view. This samples each input view into a circular cone-beam
    # detector whose source z is fixed at CONE_REBIN_CENTER_Z.
    grid = grid.reshape(views, nv, nu, 2).repeat(b * c, 1, 1, 1)
    rebinned = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    rebinned = rebinned.view(b, c, views, nv, nu).contiguous()

    projector3d = xrecon.Projector3d(
        (IMAGE_SIZE, IMAGE_SIZE, NUM_SLICES),
        (nu, nv),
        (PIXEL_SIZE, PIXEL_SIZE, SLICE_SPACING),
        (geometry.det_col_size * DET_COL_STRIDE, geometry.det_row_size * DET_ROW_STRIDE),
        geometry.iso_source,
        geometry.source_detector,
        pixshift=(0.0, 0.0, (VOLUME_Z_SHIFT - CONE_REBIN_CENTER_Z) / SLICE_SPACING),
        binshift=binshift,
        scan_type="arc",
        filter_type=FILTER_TYPE,
    ).to(device)
    print(f"cone rebin FDK: views={idx.numel()}, source_z={source_z_i[0].item():.3f}..{source_z_i[-1].item():.3f} mm")
    return projector3d.filtered_backprojection(rebinned, angles_i, filter_type=FILTER_TYPE)


def save_result(name: str, recon: torch.Tensor, geometry) -> None:
    out_dir = OUTPUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    recon_np = recon.detach().cpu().numpy()[0, 0]
    display = mayo.to_hu(recon_np, geometry.water_mu or WATER_MU, DISPLAY_RAW)
    np.save(out_dir / "recon_raw.npy", recon_np.astype(np.float32))
    np.save(out_dir / "recon_display.npy", display.astype(np.float32))
    mayo.save_auto_preview(recon_np, out_dir / "recon_raw_auto.png")
    mayo.save_preview(display, out_dir / "recon_hu_window.png", WINDOW_MIN, WINDOW_MAX, "HU window")
    save_center_crop(display, out_dir / "recon_hu_center_crop.png", WINDOW_MIN, WINDOW_MAX)
    print(out_dir / "recon_raw_auto.png")
    print(out_dir / "recon_hu_window.png")
    print(out_dir / "recon_hu_center_crop.png")


def save_center_crop(volume: np.ndarray, out_png: Path, vmin: float, vmax: float) -> None:
    import matplotlib.pyplot as plt

    z = volume.shape[0] // 2
    image = volume[z]
    h, w = image.shape
    half = min(180, h // 2, w // 2)
    crop = image[h // 2 - half : h // 2 + half, w // 2 - half : w // 2 + half]

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.imshow(crop, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(f"{out_png.parent.name}, z={z}, window [{vmin:g}, {vmax:g}]")
    ax.axis("off")
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def save_slice_route_comparison() -> None:
    import matplotlib.pyplot as plt

    route_names = ["slice_parallel_fbp", "slice_fan_arc_fbp", "slice_fan_flat_fbp"]
    paths = [OUTPUT_DIR / name / "recon_display.npy" for name in route_names]
    if not all(path.exists() for path in paths):
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, name, path in zip(axes, route_names, paths):
        volume = np.load(path)
        z = volume.shape[0] // 2
        image = volume[z]
        h, w = image.shape
        half = min(180, h // 2, w // 2)
        crop = image[h // 2 - half : h // 2 + half, w // 2 - half : w // 2 + half]
        ax.imshow(crop, cmap="gray", vmin=WINDOW_MIN, vmax=WINDOW_MAX)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(f"slice rebin comparison, {FILTER_TYPE} filter, HU [{WINDOW_MIN:g}, {WINDOW_MAX:g}]")
    out_png = OUTPUT_DIR / "slice_rebin_comparison.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(out_png)


def print_geometry(geometry, chosen, sino, angles, source_z, binshift) -> None:
    print("==== Mayo FDK parameters ====")
    print(f"algorithm             : {ALGORITHM}")
    print(f"data_dir              : {DATA_DIR}")
    print(f"files                 : {chosen[0].name} ... {chosen[-1].name} ({len(chosen)} views)")
    print(f"sino shape [V,Nv,Nu]  : {sino.shape}")
    print(f"det center dicom      : col={geometry.det_center_col}, row={geometry.det_center_row}")
    print(f"binshift [u,v]        : {binshift}")
    print(f"SOD, SDD mm           : {geometry.iso_source}, {geometry.source_detector}")
    print(f"views per rotation    : {geometry.views_per_rotation}")
    print(f"angle range rad       : {angles[0]:.6f} ... {angles[-1]:.6f}")
    print(f"source z rel mm       : {source_z.min():.4f} ... {source_z.max():.4f}")
    print(f"image [W,H,D]         : {IMAGE_SIZE}, {IMAGE_SIZE}, {NUM_SLICES}")
    print(f"pixel [dx,dy,dz] mm   : {PIXEL_SIZE}, {PIXEL_SIZE}, {SLICE_SPACING}")
    print(f"flip cols/rows        : {FLIP_DET_COLS}, {FLIP_DET_ROWS}")
    print(f"use source shifts     : {USE_SOURCE_SHIFTS}")
    print(f"smooth source z       : {SMOOTH_SOURCE_Z_FOR_FDK}")
    print(f"filter                : {FILTER_TYPE}")
    print(f"direct FDK scale      : {DIRECT_FDK_SCALE}")
    if source_z.size > 1 and angles.size > 1:
        pitch_per_turn = abs(float(np.polyfit(angles.astype(np.float64), source_z.astype(np.float64), 1)[0])) * 2.0 * np.pi
        iso_det_height = geometry.det_row_size * DET_ROW_STRIDE * sino.shape[1] * geometry.iso_source / geometry.source_detector
        norm_pitch = pitch_per_turn / max(iso_det_height, 1.0e-6)
        print(f"pitch per turn mm     : {pitch_per_turn:.4f}")
        print(f"norm pitch by det     : {norm_pitch:.4f}")
    print(f"helical view mode     : {HELICAL_VIEW_MODE}")
    print(f"helical view turns    : {HELICAL_VIEW_TURNS}")
    print(f"helical half coef     : {HELICAL_HALFSCAN_COEF}")
    print(f"helical weight mode   : {HELICAL_WEIGHT_MODE}")
    print(f"helical Tang k        : {HELICAL_TANG_K}")
    print(f"helical Tang blend    : {HELICAL_TANG_BLEND}")
    print(f"slice rebin views     : {SLICE_REBIN_VIEWS}")
    print(f"cone rebin views      : {CONE_REBIN_NUM_VIEWS}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(DEVICE if DEVICE == "cpu" or torch.cuda.is_available() else "cpu")

    geometry, chosen, sino_np, angles_np, source_z_np, shift_z, shift_angle, shift_radius, binshift = load_mayo_problem()
    print_geometry(geometry, chosen, sino_np, angles_np, source_z_np, binshift)

    sino = torch.from_numpy(sino_np).view(1, 1, *sino_np.shape).to(device=device, dtype=torch.float32)
    angles = torch.from_numpy(angles_np).to(device=device, dtype=torch.float32)
    source_z = torch.from_numpy(source_z_np).to(device=device, dtype=torch.float32)
    source_pos = make_source_pos(source_z_np, shift_z, shift_angle, shift_radius, device)

    with torch.no_grad():
        if ALGORITHM in {"direct_fdk", "direct", "all"}:
            projector = make_helical_projector(geometry, sino_np.shape, binshift, device)
            recon = direct_fdk(projector, sino, angles, source_pos)
            save_result("direct_fdk", recon, geometry)

        if ALGORITHM in {"slice_parallel_fbp", "slice_parallel", "rebin", "slice_all", "all"}:
            recon = slice_rebin_recon(sino, angles, source_z, geometry, binshift, device, "parallel")
            save_result("slice_parallel_fbp", recon, geometry)

        if ALGORITHM in {"slice_fan_arc_fbp", "slice_fan_arc", "slice_all", "all"}:
            recon = slice_rebin_recon(sino, angles, source_z, geometry, binshift, device, "fan_arc")
            save_result("slice_fan_arc_fbp", recon, geometry)

        if ALGORITHM in {"slice_fan_flat_fbp", "slice_fan_flat", "slice_all", "all"}:
            recon = slice_rebin_recon(sino, angles, source_z, geometry, binshift, device, "fan_flat")
            save_result("slice_fan_flat_fbp", recon, geometry)

        if ALGORITHM in {"cone_rebin_fdk", "cone_fdk", "all"}:
            recon = cone_rebin_fdk(sino, angles, source_z, geometry, binshift, device)
            save_result("cone_rebin_fdk", recon, geometry)

        if ALGORITHM in {"slice_all", "all"}:
            save_slice_route_comparison()


if __name__ == "__main__":
    main()
