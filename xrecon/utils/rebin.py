from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def fan_to_parallel_rebin(
    sino: Tensor,
    angles: Tensor,
    iso_source: float,
    det_spacing: float,
    det_shift: float = 0.0,
    detector_type: str = "arc",
    output_angles: Optional[Tensor] = None,
    output_det_count: Optional[int] = None,
    output_det_spacing: Optional[float] = None,
    output_det_shift: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Rebin fan/cone data to parallel geometry along the detector-column axis.

    Args:
        sino: [B, C, V, Nu] or [B, C, V, Nv, Nu].
        angles: source rotation angles [V].
        iso_source: source-to-isocenter distance.
        det_spacing: detector-column spacing. Use angular spacing for arc
            detectors and physical spacing at isocenter for flat detectors.
        det_shift: detector-column physical/angular shift in the same unit as
            det_spacing.
        detector_type: "arc" or "flat".
        output_angles: parallel angles. Defaults to input angles.
        output_det_count: number of parallel detector bins. Defaults to Nu.
        output_det_spacing: parallel detector spacing in length units. Defaults
            to the fan coverage mapped to a uniform t grid.
        output_det_shift: parallel detector shift in length units.

    Returns:
        parallel_sino, output_angles, output_t
    """
    if sino.ndim not in (4, 5):
        raise ValueError(f"Expected sino shape [B,C,V,Nu] or [B,C,V,Nv,Nu], got {tuple(sino.shape)}")
    if angles.ndim != 1:
        raise ValueError(f"Expected angles shape [V], got {tuple(angles.shape)}")

    detector_type = detector_type.lower().strip()
    if detector_type not in {"arc", "flat"}:
        raise ValueError("detector_type should be 'arc' or 'flat'")

    squeeze_row = sino.ndim == 4
    if squeeze_row:
        sino5 = sino[:, :, :, None, :]
    else:
        sino5 = sino

    B, C, V, Nv, Nu = sino5.shape
    if V != angles.numel():
        raise ValueError(f"Expected {V} angles, got {angles.numel()}")

    output_angles = angles if output_angles is None else output_angles
    output_angles = output_angles.to(device=sino.device, dtype=sino.dtype).contiguous()
    angles = angles.to(device=sino.device, dtype=sino.dtype).contiguous()
    Nt = Nu if output_det_count is None else int(output_det_count)

    u_index = torch.arange(-Nu / 2 + 0.5, Nu / 2, device=sino.device, dtype=sino.dtype)
    u = u_index * float(det_spacing) + float(det_shift)
    if detector_type == "arc":
        gamma_extent = u.abs().max()
    else:
        gamma_extent = torch.atan2(u.abs().max(), u.new_tensor(float(iso_source)))
    max_t = float(iso_source) * torch.sin(gamma_extent)

    if output_det_spacing is None:
        output_det_spacing = (2.0 * max_t / Nt).item() if Nt > 0 else 1.0
    t_index = torch.arange(-Nt / 2 + 0.5, Nt / 2, device=sino.device, dtype=sino.dtype)
    output_t = t_index * float(output_det_spacing) + float(output_det_shift)

    ratio = torch.clamp(output_t / float(iso_source), min=-1.0 + 1e-6, max=1.0 - 1e-6)
    gamma = torch.asin(ratio)
    beta = output_angles[:, None] - gamma[None, :]

    if detector_type == "arc":
        u_query = gamma
    else:
        u_query = torch.tan(gamma) * float(iso_source)

    angle0 = angles[0]
    if angles.numel() < 2:
        angle_step = angles.new_tensor(1.0)
    else:
        angle_step = torch.diff(angles).mean()
    beta_index = (beta - angle0) / angle_step
    u_coord = (u_query - float(det_shift)) / float(det_spacing) + Nu / 2 - 0.5

    grid_y = 2.0 * beta_index / max(V - 1, 1) - 1.0
    grid_x = 2.0 * u_coord[None, :] / max(Nu - 1, 1) - 1.0
    grid = torch.stack((grid_x.expand_as(grid_y), grid_y), dim=-1)

    src = sino5.permute(0, 1, 3, 2, 4).reshape(B * C * Nv, 1, V, Nu)
    grid = grid[None, ...].expand(B * C * Nv, -1, -1, -1)
    out = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    out = out.reshape(B, C, Nv, output_angles.numel(), Nt).permute(0, 1, 3, 2, 4).contiguous()
    if squeeze_row:
        out = out[:, :, :, 0, :]
    return out, output_angles, output_t

def helical_fan_to_parallel_rebin(
    sino: Tensor,
    angles: Tensor,
    source_z: Tensor,
    iso_source: float,
    det_spacing: float,
    det_shift: float = 0.0,
    detector_type: str = "arc",
    output_angles: Optional[Tensor] = None,
    output_det_count: Optional[int] = None,
    output_det_spacing: Optional[float] = None,
    output_det_shift: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Rebin helical fan/cone data to a parallel detector-column parameterization.

    This is the column-direction fan-to-parallel step:
        theta = beta + gamma, t = SOD * sin(gamma)

    The detector-row axis is kept in the original sampled coordinates and the
    source z location is interpolated onto the rebinned parallel angle grid.
    This helper is useful for algorithms that explicitly choose a rebinning
    route, but direct helical backprojection should use HelicalProjector3d
    instead of treating this as a complete helical reconstruction.
    """
    if source_z.ndim != 1:
        raise ValueError(f"Expected source_z shape [V], got {tuple(source_z.shape)}")
    if source_z.numel() != angles.numel():
        raise ValueError(f"Expected source_z length {angles.numel()}, got {source_z.numel()}")

    rebinned, out_angles, out_t = fan_to_parallel_rebin(
        sino,
        angles,
        iso_source,
        det_spacing,
        det_shift=det_shift,
        detector_type=detector_type,
        output_angles=output_angles,
        output_det_count=output_det_count,
        output_det_spacing=output_det_spacing,
        output_det_shift=output_det_shift,
    )

    source_z = source_z.to(device=rebinned.device, dtype=rebinned.dtype).contiguous()
    out_angles = out_angles.to(device=rebinned.device, dtype=rebinned.dtype).contiguous()
    in_angles = angles.to(device=rebinned.device, dtype=rebinned.dtype).contiguous()
    if in_angles.numel() < 2:
        out_source_z = source_z.expand_as(out_angles).contiguous()
    else:
        step = torch.diff(in_angles).mean()
        idx = (out_angles - in_angles[0]) / step
        idx0 = torch.floor(idx).to(torch.long)
        idx1 = idx0 + 1
        w = (idx - idx0.to(idx.dtype)).clamp(0.0, 1.0)
        valid = (idx0 >= 0) & (idx1 < source_z.numel())
        idx0 = idx0.clamp(0, source_z.numel() - 1)
        idx1 = idx1.clamp(0, source_z.numel() - 1)
        out_source_z = (1.0 - w) * source_z[idx0] + w * source_z[idx1]
        out_source_z = torch.where(valid, out_source_z, torch.zeros_like(out_source_z))
    return rebinned, out_angles, out_t, out_source_z.contiguous()


helical_cone_to_parallel_rebin = helical_fan_to_parallel_rebin
