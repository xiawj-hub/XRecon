from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from .filters import filter_generate
from .functions import (
    BackProjection3d,
    BackProjectionHelical3d,
    BackProjectionTransposeHelical3d,
    BackProjectionTranspose3d,
    Projection3d,
    ProjectionHelical3d,
    ProjectionTransposeHelical3d,
    ProjectionTranspose3d,
    WeightedBackProjection3d,
    WeightedBackProjectionHelical3d,
    WeightedBackProjectionTransposeHelical3d,
    WeightedBackProjectionTranspose3d,
    system_matrix3d,
)
from .geometry import ConeBeam3d, Detector3d, Geometry3d, HelicalConeBeam3d, ParallelBeam3d, Volume3d, make_geometry3d


class Projector3d(nn.Module):
    """
    Distance-driven circular cone-beam 3D projector.

    Shapes:
        image:  [B, C, D, H, W]
        angles: [V]
        sino:   [B, C, V, Nv, Nu]
    """

    def __init__(
        self,
        image_size: Union[int, Tuple[int, int, int], List[int], Volume3d],
        num_det: Optional[Union[int, Tuple[int, int], List[int]]] = None,
        pix_size: Union[float, Tuple[float, float, float], List[float], Geometry3d, None] = None,
        det_size: Optional[Union[float, Tuple[float, float], List[float]]] = None,
        iso_source: float = 0.0,
        source_detector: float = 0.0,
        pixshift: Union[float, Tuple[float, float, float], List[float]] = 0.0,
        binshift: Union[float, Tuple[float, float], List[float]] = 0.0,
        scan_type: str = "flat",
        filter_type: str = "ramp",
        trainable: bool = False,
        dtype=None,
    ) -> None:
        super().__init__()

        if isinstance(image_size, Volume3d):
            volume = image_size
            if isinstance(num_det, (ConeBeam3d, ParallelBeam3d)) and pix_size is None:
                pix_size = num_det
            if not isinstance(pix_size, (ConeBeam3d, ParallelBeam3d)):
                raise TypeError("When image_size is Volume3d, pix_size should be a Geometry3d object")
            geometry = pix_size
        else:
            if num_det is None or pix_size is None or det_size is None:
                raise TypeError("num_det, pix_size and det_size are required with the legacy constructor")
            volume = Volume3d(image_size, pix_size, pixshift)
            geometry = make_geometry3d(scan_type, num_det, det_size, iso_source, source_detector, binshift)

        scan_type = geometry.scan_type.lower().strip()
        if scan_type not in {"parallel", "flat", "arc"}:
            raise ValueError(f"Unsupported 3D scan_type: {scan_type}")

        W, H, D = volume.size
        dx, dy, dz = volume.spacing
        shx, shy, shz = volume.shift
        Nu, Nv = geometry.detector.count
        du, dv = geometry.effective_det_spacing
        shu, shv = geometry.effective_det_shift
        raw_du, raw_dv = geometry.detector.spacing
        raw_shu, raw_shv = geometry.detector.shift
        kernel_du = raw_du if scan_type == "flat" else du
        kernel_shu = raw_shu * raw_du if scan_type == "flat" else shu

        self.volume = volume
        self.geometry = geometry
        self.image_size = (W, H, D)
        self.pix_size = (dx, dy, dz)
        self.num_det = (Nu, Nv)
        self.det_size = (du, dv)
        self.binshift = (shu, shv)
        self.pixshift = (shx * dx, shy * dy, shz * dz)
        self.scan_type = scan_type
        self.filter_type = filter_type.lower().strip()
        self.iso_source = float(getattr(geometry, "iso_source", 0.0))
        self.source_detector = float(getattr(geometry, "source_detector", 0.0))
        self._default_dtype = dtype or torch.float32

        options = torch.tensor(
            [
                W, H, D, Nu, Nv,
                dx, dy, dz, kernel_du, raw_dv,
                self.iso_source, self.source_detector,
                self.pixshift[0], self.pixshift[1], self.pixshift[2],
                kernel_shu, raw_shv * raw_dv, geometry.scan_id,
            ],
            dtype=self._default_dtype,
        )
        self.register_buffer("options", options, persistent=False)

        weight = self._make_cosine_weight(dtype=self._default_dtype)
        if trainable:
            self.weight = nn.parameter.Parameter(weight.clone())
        else:
            self.register_buffer("weight", weight, persistent=False)

        filt = self._filter_generate(filter_type=self.filter_type, dtype=self._default_dtype)
        if trainable:
            self.filter = nn.parameter.Parameter(filt.clone())
        else:
            self.register_buffer("filter", filt, persistent=False)

    def forward(self, image: Tensor, angles: Tensor) -> Tensor:
        return self.projection(image, angles)

    def projection(self, image: Tensor, angles: Tensor) -> Tensor:
        self._check_image(image)
        angles = self._prepare_angles(angles, image)
        return Projection3d.apply(image, self.options, angles)

    def projection_t(self, sino: Tensor, angles: Tensor) -> Tensor:
        self._check_sino_angles(sino, angles)
        angles = self._prepare_angles(angles, sino)
        return ProjectionTranspose3d.apply(sino, self.options, angles)

    def backprojection(self, sino: Tensor, angles: Tensor) -> Tensor:
        self._check_sino_angles(sino, angles)
        angles = self._prepare_angles(angles, sino)
        return BackProjection3d.apply(sino, self.options, angles)

    def backprojection_weighted(self, sino: Tensor, angles: Tensor) -> Tensor:
        self._check_sino_angles(sino, angles)
        angles = self._prepare_angles(angles, sino)
        return WeightedBackProjection3d.apply(sino, self.options, angles)

    def backprojection_t(self, image: Tensor, angles: Tensor) -> Tensor:
        self._check_image(image)
        angles = self._prepare_angles(angles, image)
        return BackProjectionTranspose3d.apply(image, self.options, angles)

    def backprojection_weighted_t(self, image: Tensor, angles: Tensor) -> Tensor:
        self._check_image(image)
        angles = self._prepare_angles(angles, image)
        return WeightedBackProjectionTranspose3d.apply(image, self.options, angles)

    def system_matrix(self, angles: Tensor) -> Tensor:
        angles = torch.as_tensor(angles, dtype=self.options.dtype, device="cpu").contiguous()
        return system_matrix3d(self.options.cpu().contiguous(), angles)

    get_sys_matrix = system_matrix

    projection_transpose = projection_t
    backprojection_transpose = backprojection_t
    weighted_backprojection_transpose = backprojection_weighted_t

    def filtered_backprojection(
        self,
        sino: Tensor,
        angles: Tensor,
        filter: Optional[Tensor] = None,
        filter_type: Optional[str] = None,
        redundant: Optional[bool] = None,
        parker: bool = False,
    ) -> Tensor:
        self._check_sino_angles(sino, angles)
        if filter is not None:
            filt = filter
        elif filter_type is None or filter_type.lower() == self.filter_type:
            filt = self.filter
        else:
            filt = self._filter_generate(filter_type.lower(), dtype=sino.dtype).to(sino.device)

        if parker:
            sino = sino * self.parker_weight(angles).to(device=sino.device, dtype=sino.dtype)
            if redundant is None:
                redundant = False
        elif redundant is None:
            redundant = True

        sino = sino * self.weight.to(device=sino.device, dtype=sino.dtype)
        filtered = nn.functional.conv3d(sino, filt.to(device=sino.device, dtype=sino.dtype), padding=(0, 0, self.num_det[0] - 1))
        recon = self.backprojection_weighted(filtered, angles)
        recon = recon * self._mean_angle_step(angles).to(device=recon.device, dtype=recon.dtype)
        if redundant:
            recon = recon / 2
        return recon

    def parker_weight(self, angles: Tensor) -> Tensor:
        if angles.numel() < 2:
            return torch.ones((1, 1, angles.numel(), 1, self.num_det[0]), device=angles.device, dtype=angles.dtype)
        if self.scan_type == "parallel":
            return torch.ones((1, 1, angles.numel(), 1, self.num_det[0]), device=angles.device, dtype=angles.dtype)

        beta = angles - angles[0]
        scan_range = beta[-1].abs()
        gamma = self._fan_angles(device=angles.device, dtype=angles.dtype)
        gamma_max = gamma.abs().max()
        if scan_range >= (2 * math.pi - 1e-4):
            return torch.ones((1, 1, angles.numel(), 1, self.num_det[0]), device=angles.device, dtype=angles.dtype)

        min_scan_range = math.pi + 2.0 * gamma_max
        if scan_range < min_scan_range - self._mean_angle_step(angles):
            raise ValueError(
                "Parker weighting requires a short-scan range of at least "
                f"pi + 2 * gamma_max, got {scan_range.item():.6f} rad"
            )

        overscan = torch.clamp((scan_range - math.pi) / 2.0, min=gamma_max)
        eps = torch.finfo(angles.dtype).eps
        left = 2.0 * (overscan - gamma)
        right = scan_range - 2.0 * (overscan + gamma)
        beta2d = beta[:, None]
        w = torch.ones((angles.numel(), self.num_det[0]), device=angles.device, dtype=angles.dtype)
        w_left = torch.sin(0.5 * math.pi * beta2d / torch.clamp(left, min=eps)[None, :]).pow(2)
        w_right = torch.sin(0.5 * math.pi * (scan_range - beta2d) / torch.clamp(scan_range - right, min=eps)[None, :]).pow(2)
        w = torch.where(beta2d < left[None, :], w_left, w)
        w = torch.where(beta2d > right[None, :], w_right, w)
        return w.clamp_(0.0, 1.0).view(1, 1, angles.numel(), 1, self.num_det[0])

    def _filter_generate(self, filter_type: str, dtype=None) -> Tensor:
        filt = filter_generate(self.num_det[0], self.det_size[0], filter_type, dtype)
        return filt[None, None, None, None, :]

    def _make_cosine_weight(self, dtype=None) -> Tensor:
        dtype = dtype or torch.float32
        Nu, _ = self.num_det
        du, _ = self.det_size
        shu, _ = self.binshift
        u = torch.arange(-Nu / 2 + 0.5, Nu / 2, 1.0, dtype=dtype)
        s = u * du + shu
        if self.scan_type == "parallel":
            return torch.ones((1, 1, 1, 1, Nu), dtype=dtype)
        if self.scan_type == "flat":
            w = self.iso_source / torch.sqrt(s * s + self.iso_source * self.iso_source)
        else:
            w = torch.cos(s)
        return w.view(1, 1, 1, 1, Nu)

    def _fan_angles(self, device=None, dtype=None) -> Tensor:
        dtype = dtype or torch.float32
        Nu, _ = self.num_det
        du, _ = self.det_size
        shu, _ = self.binshift
        u = torch.arange(-Nu / 2 + 0.5, Nu / 2, 1.0, device=device, dtype=dtype)
        s = u * du + shu
        if self.scan_type == "parallel":
            return s
        if self.scan_type == "arc":
            return s
        return torch.atan2(s, torch.as_tensor(self.iso_source, device=device, dtype=dtype))

    def _check_image(self, image: Tensor) -> None:
        if image.ndim != 5:
            raise ValueError(f"Expected image shape [B, C, D, H, W], got {tuple(image.shape)}")
        W, H, D = self.image_size
        if tuple(image.shape[2:]) != (D, H, W):
            raise ValueError(f"Expected image shape [B, C, {D}, {H}, {W}], got {tuple(image.shape)}")

    def _check_sino_angles(self, sino: Tensor, angles: Tensor) -> None:
        if sino.ndim != 5:
            raise ValueError(f"Expected sino shape [B, C, V, Nv, Nu], got {tuple(sino.shape)}")
        if angles.ndim != 1:
            raise ValueError(f"Expected angles shape [V], got {tuple(angles.shape)}")
        Nu, Nv = self.num_det
        if sino.shape[2] != angles.numel() or sino.shape[3] != Nv or sino.shape[4] != Nu:
            raise ValueError(f"Expected sino shape [B, C, {angles.numel()}, {Nv}, {Nu}], got {tuple(sino.shape)}")

    @staticmethod
    def _prepare_angles(angles: Tensor, ref: Tensor) -> Tensor:
        return angles.to(device=ref.device, dtype=ref.dtype).contiguous()

    @staticmethod
    def _mean_angle_step(angles: Tensor) -> Tensor:
        if angles.numel() < 2:
            return angles.new_tensor(0.0)
        diffs = torch.diff(angles)
        wrap = (diffs + math.pi) % (2 * math.pi) - math.pi
        return wrap.abs().mean()


class HelicalProjector3d(nn.Module):
    """
    Distance-driven helical cone-beam 3D projector.

    The native direct helical CUDA kernel currently supports arc-detector
    geometry. Flat-detector helical data can be routed through an explicit
    rebin-to-parallel workflow, or handled by a future direct flat helical
    kernel.
    """

    def __init__(
        self,
        image_size: Union[int, Tuple[int, int, int], List[int], Volume3d],
        num_det: Optional[Union[int, Tuple[int, int], List[int]]] = None,
        pix_size: Union[float, Tuple[float, float, float], List[float], HelicalConeBeam3d, None] = None,
        det_size: Optional[Union[float, Tuple[float, float], List[float]]] = None,
        iso_source: float = 0.0,
        source_detector: float = 0.0,
        pixshift: Union[float, Tuple[float, float, float], List[float]] = 0.0,
        binshift: Union[float, Tuple[float, float], List[float]] = 0.0,
        scan_type: str = "arc",
        pitch: float = 0.0,
        z0: float = 0.0,
        helical_view_mode: str = "one_turn",
        helical_view_turns: float = 0.0,
        helical_halfscan_coef: float = 1.0,
        helical_weight_mode: str = "2d",
        helical_tang_k: float = 5.0,
        helical_tang_blend: float = 0.0,
        filter_type: str = "ramp",
        trainable: bool = False,
        dtype=None,
    ) -> None:
        super().__init__()

        if isinstance(image_size, Volume3d):
            volume = image_size
            if isinstance(num_det, HelicalConeBeam3d) and pix_size is None:
                pix_size = num_det
            if not isinstance(pix_size, HelicalConeBeam3d):
                raise TypeError("When image_size is Volume3d, pix_size should be a HelicalConeBeam3d object")
            geometry = pix_size
        else:
            if num_det is None or pix_size is None or det_size is None:
                raise TypeError("num_det, pix_size and det_size are required with the legacy constructor")
            volume = Volume3d(image_size, pix_size, pixshift)
            geometry = HelicalConeBeam3d(
                Detector3d(num_det, det_size, binshift),
                iso_source,
                source_detector,
                scan_type,
                pitch=pitch,
                z0=z0,
            )

        scan_type = geometry.scan_type.lower().strip()
        if scan_type != "arc":
            raise NotImplementedError("Direct helical projection/backprojection currently supports arc detector geometry only")

        W, H, D = volume.size
        dx, dy, dz = volume.spacing
        shx, shy, shz = volume.shift
        Nu, Nv = geometry.detector.count
        du, dv = geometry.effective_det_spacing
        shu, shv = geometry.effective_det_shift

        self.volume = volume
        self.geometry = geometry
        self.image_size = (W, H, D)
        self.pix_size = (dx, dy, dz)
        self.num_det = (Nu, Nv)
        self.det_size = (du, dv)
        self.binshift = (shu, shv)
        self.pixshift = (shx * dx, shy * dy, shz * dz)
        self.scan_type = scan_type
        self.filter_type = filter_type.lower().strip()
        self.iso_source = float(geometry.iso_source)
        self.source_detector = float(geometry.source_detector)
        self.pitch = float(geometry.pitch)
        self.z0 = float(geometry.z0)
        self.helical_view_mode = self._view_mode_id(helical_view_mode)
        self.helical_weight_mode = self._weight_mode_id(helical_weight_mode)
        self.helical_view_turns = float(helical_view_turns)
        self.helical_halfscan_coef = float(helical_halfscan_coef)
        self.helical_tang_k = float(helical_tang_k)
        self.helical_tang_blend = float(helical_tang_blend)
        self._default_dtype = dtype or torch.float32

        options = torch.tensor(
            [
                W, H, D, Nu, Nv,
                dx, dy, dz, du, dv,
                self.iso_source, self.source_detector,
                self.pixshift[0], self.pixshift[1], self.pixshift[2],
                shu, shv, geometry.scan_id,
                self.helical_view_mode,
                self.helical_weight_mode,
                self.helical_view_turns,
                self.helical_halfscan_coef,
                self.helical_tang_k,
                self.helical_tang_blend,
            ],
            dtype=self._default_dtype,
        )
        self.register_buffer("options", options, persistent=False)

        weight = self._make_weight(dtype=self._default_dtype)
        if trainable:
            self.weight = nn.parameter.Parameter(weight.clone())
        else:
            self.register_buffer("weight", weight, persistent=False)

        filt = self._filter_generate(filter_type=self.filter_type, dtype=self._default_dtype)
        if trainable:
            self.filter = nn.parameter.Parameter(filt.clone())
        else:
            self.register_buffer("filter", filt, persistent=False)

    def forward(self, image: Tensor, angles: Tensor, source_pos: Optional[Tensor] = None) -> Tensor:
        return self.projection(image, angles, source_pos)

    def source_position(self, angles: Tensor, ref: Optional[Tensor] = None) -> Tensor:
        ref = angles if ref is None else ref
        angles = self._prepare_angles(angles, ref)
        source_z = angles.new_empty(angles.shape)
        if angles.numel() == 0:
            return angles.new_zeros((4, 0))
        source_z.copy_(self.z0 + self.pitch * (angles - angles[0]) / (2.0 * math.pi))
        shifts = angles.new_zeros((3, angles.numel()))
        return torch.cat((source_z.view(1, -1), shifts), dim=0).contiguous()

    def projection(self, image: Tensor, angles: Tensor, source_pos: Optional[Tensor] = None) -> Tensor:
        self._check_image(image)
        angles = self._prepare_angles(angles, image)
        source_pos = self._prepare_source_pos(source_pos, angles, image)
        return ProjectionHelical3d.apply(image, self.options, angles, source_pos)

    def projection_t(self, sino: Tensor, angles: Tensor, source_pos: Optional[Tensor] = None) -> Tensor:
        self._check_sino_angles(sino, angles)
        angles = self._prepare_angles(angles, sino)
        source_pos = self._prepare_source_pos(source_pos, angles, sino)
        return ProjectionTransposeHelical3d.apply(sino, self.options, angles, source_pos)

    def backprojection(self, sino: Tensor, angles: Tensor, source_pos: Optional[Tensor] = None) -> Tensor:
        self._check_sino_angles(sino, angles)
        angles = self._prepare_angles(angles, sino)
        source_pos = self._prepare_source_pos(source_pos, angles, sino)
        return BackProjectionHelical3d.apply(sino, self.options, angles, source_pos)

    def backprojection_weighted(self, sino: Tensor, angles: Tensor, source_pos: Optional[Tensor] = None) -> Tensor:
        self._check_sino_angles(sino, angles)
        angles = self._prepare_angles(angles, sino)
        source_pos = self._prepare_source_pos(source_pos, angles, sino)
        return WeightedBackProjectionHelical3d.apply(sino, self.options, angles, source_pos)

    def backprojection_t(self, image: Tensor, angles: Tensor, source_pos: Optional[Tensor] = None) -> Tensor:
        self._check_image(image)
        angles = self._prepare_angles(angles, image)
        source_pos = self._prepare_source_pos(source_pos, angles, image)
        return BackProjectionTransposeHelical3d.apply(image, self.options, angles, source_pos)

    def backprojection_weighted_t(self, image: Tensor, angles: Tensor, source_pos: Optional[Tensor] = None) -> Tensor:
        self._check_image(image)
        angles = self._prepare_angles(angles, image)
        source_pos = self._prepare_source_pos(source_pos, angles, image)
        return WeightedBackProjectionTransposeHelical3d.apply(image, self.options, angles, source_pos)

    projection_transpose = projection_t
    backprojection_transpose = backprojection_t
    weighted_backprojection_transpose = backprojection_weighted_t
    backprojection_w = backprojection_weighted

    def filtered_backprojection(
        self,
        sino: Tensor,
        angles: Tensor,
        source_pos: Optional[Tensor] = None,
        filter: Optional[Tensor] = None,
        filter_type: Optional[str] = None,
    ) -> Tensor:
        self._check_sino_angles(sino, angles)
        angles = self._prepare_angles(angles, sino)
        source_pos = self._prepare_source_pos(source_pos, angles, sino)
        if filter is not None:
            filt = filter
        elif filter_type is None or filter_type.lower() == self.filter_type:
            filt = self.filter
        else:
            filt = self._filter_generate(filter_type.lower(), dtype=sino.dtype).to(sino.device)
        sino = sino * self.weight.to(device=sino.device, dtype=sino.dtype)
        filtered = nn.functional.conv3d(sino, filt.to(device=sino.device, dtype=sino.dtype), padding=(0, 0, self.num_det[0] - 1))
        recon = self.backprojection_weighted(filtered, angles, source_pos)
        return recon * self._mean_angle_step(angles).to(device=recon.device, dtype=recon.dtype)

    helical_fdk = filtered_backprojection

    @staticmethod
    def _view_mode_id(mode: str) -> int:
        aliases = {
            "auto": 0,
            "half": 1,
            "halfscan": 1,
            "half_scan": 1,
            "short": 1,
            "short_scan": 1,
            "one": 2,
            "one_turn": 2,
            "1turn": 2,
            "full": 2,
            "full_scan": 2,
            "fixed": 3,
            "turns": 3,
            "multi": 3,
            "multi_turn": 3,
        }
        key = str(mode).lower().strip()
        if key not in aliases:
            raise ValueError(f"Unsupported helical_view_mode: {mode}")
        return aliases[key]

    @staticmethod
    def _weight_mode_id(mode: str) -> int:
        aliases = {
            "legacy": 0,
            "2d": 0,
            "redundancy": 0,
            "tang": 1,
            "tang3d": 1,
            "3d": 1,
            "cone": 1,
        }
        key = str(mode).lower().strip()
        if key not in aliases:
            raise ValueError(f"Unsupported helical_weight_mode: {mode}")
        return aliases[key]

    def _filter_generate(self, filter_type: str, dtype=None) -> Tensor:
        filt = filter_generate(self.num_det[0], self.det_size[0], filter_type, dtype)
        return filt[None, None, None, None, :]

    def _make_weight(self, dtype=None) -> Tensor:
        dtype = dtype or torch.float32
        Nu, Nv = self.num_det
        du, dv = self.det_size
        shu, shv = self.binshift
        u = torch.arange(-Nu / 2 + 0.5, Nu / 2, 1.0, dtype=dtype) * du + shu
        v = torch.arange(-Nv / 2 + 0.5, Nv / 2, 1.0, dtype=dtype) * dv + shv
        col = torch.tan(u) * self.iso_source
        row = v * self.iso_source / self.source_detector
        w = self.iso_source / torch.sqrt(row[:, None].pow(2) + col[None, :].pow(2) + self.iso_source * self.iso_source)
        return w.view(1, 1, 1, Nv, Nu)

    def _check_image(self, image: Tensor) -> None:
        if image.ndim != 5:
            raise ValueError(f"Expected image shape [B, C, D, H, W], got {tuple(image.shape)}")
        W, H, D = self.image_size
        if tuple(image.shape[2:]) != (D, H, W):
            raise ValueError(f"Expected image shape [B, C, {D}, {H}, {W}], got {tuple(image.shape)}")

    def _check_sino_angles(self, sino: Tensor, angles: Tensor) -> None:
        if sino.ndim != 5:
            raise ValueError(f"Expected sino shape [B, C, V, Nv, Nu], got {tuple(sino.shape)}")
        if angles.ndim != 1:
            raise ValueError(f"Expected angles shape [V], got {tuple(angles.shape)}")
        Nu, Nv = self.num_det
        if sino.shape[2] != angles.numel() or sino.shape[3] != Nv or sino.shape[4] != Nu:
            raise ValueError(f"Expected sino shape [B, C, {angles.numel()}, {Nv}, {Nu}], got {tuple(sino.shape)}")

    def _prepare_source_pos(self, source_pos: Optional[Tensor], angles: Tensor, ref: Tensor) -> Tensor:
        if source_pos is None:
            return self.source_position(angles, ref)
        source_pos = source_pos.to(device=ref.device, dtype=ref.dtype).contiguous()
        if source_pos.shape != (4, angles.numel()):
            raise ValueError(f"Expected source_pos shape [4, {angles.numel()}], got {tuple(source_pos.shape)}")
        return source_pos

    @staticmethod
    def _prepare_angles(angles: Tensor, ref: Tensor) -> Tensor:
        return angles.to(device=ref.device, dtype=ref.dtype).contiguous()

    @staticmethod
    def _mean_angle_step(angles: Tensor) -> Tensor:
        if angles.numel() < 2:
            return angles.new_tensor(0.0)
        diffs = torch.diff(angles)
        wrap = (diffs + math.pi) % (2 * math.pi) - math.pi
        return wrap.abs().mean()

projector3d = Projector3d
