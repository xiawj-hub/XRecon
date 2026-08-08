import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional, Union, Tuple
from .functions import (
    BackProjection2d,
    BackProjectionTranspose2d,
    Projection2d,
    ProjectionTranspose2d,
    WeightedBackProjection2d,
    WeightedBackProjectionTranspose2d,
    system_matrix2d,
)
from .filters import filter_generate
from .geometry import FanBeam2d, Geometry2d, ParallelBeam2d, Volume2d, make_geometry2d

_SCAN_ALIASES = {"para": "parallel", "parallel": "parallel", "flat": "flat", "arc": "arc"}


class Projector2d(nn.Module):
    """
    Distance-driven 2D CT projector with optional (trainable) filter and cosine weight.

    Args:
        image_size: int or (W, H).
        num_det: number of detector bins (Nu).
        pix_size: float or (sx, sy) in [mm] (FOV size along x/y of the reconstruction plane).
        det_size: detector bin size ([mm]).
        iso_source: source-to-isocenter distance (SOD, [mm]).
        source_detector: source-to-detector distance (SDD, [mm]).
        pixshift: float or (shift_x, shift_y) in pixels (object center offset, [pixel]).
        binshift: detector offset in bins (positive to the right, [bin]).
        scan_type: 'parallel' | 'flat' | 'arc'  .
        filter_type: 'ramp' | 'shepplogan' | 'cosine' | 'hamming' | 'hann'.
        trainable: if True, filter and cosine weight are trainable nn.Parameters.
        dtype: options/initial tensors dtype. Defaults to torch.float32.

    Shapes:
        image:  [B, C, H, W]
        angles: [V]
        sino:   [B, C, V, Nu]
    """

    def __init__(
        self,
        image_size: Union[int, Tuple[int, int], List[int], Volume2d],
        num_det: Optional[int] = None,
        pix_size: Union[float, Tuple[float, float], List[float], Geometry2d, None] = None,
        det_size: Optional[float] = None,
        iso_source: float = 0.0,
        source_detector: float = 0.0,
        pixshift: Union[float, Tuple[float, float], List[float]] = 0.0,
        binshift: float = 0.0,
        scan_type: str = "flat",
        filter_type: str = "ramp",
        trainable: bool = False,
        dtype=None,
    ) -> None:
        super().__init__()

        if isinstance(image_size, Volume2d):
            volume = image_size
            if isinstance(num_det, (FanBeam2d, ParallelBeam2d)) and pix_size is None:
                pix_size = num_det
            if not isinstance(pix_size, (FanBeam2d, ParallelBeam2d)):
                raise TypeError("When image_size is Volume2d, pix_size should be a 2D geometry object")
            geometry = pix_size
        else:
            if num_det is None or pix_size is None or det_size is None:
                raise TypeError("num_det, pix_size and det_size are required with the legacy constructor")
            volume = Volume2d(image_size, pix_size, pixshift)
            geometry = make_geometry2d(scan_type, num_det, det_size, iso_source, source_detector, binshift)

        scan_type = _SCAN_ALIASES.get(geometry.scan_type.lower().strip(), geometry.scan_type.lower().strip())
        filter_type = filter_type.lower().strip()
        if scan_type not in {"parallel", "flat", "arc"}:
            raise ValueError(f"Unsupported scan_type: {scan_type}")

        W, H = volume.size
        sx, sy = volume.spacing
        shx, shy = volume.shift

        self.image_size = (W, H)
        self.pix_size = (sx, sy)
        
        self.volume = volume
        self.geometry = geometry
        self.num_det = geometry.detector.count
        self.scan_type = scan_type
        self.filter_type = filter_type
        self.iso_source = float(geometry.iso_source)
        self.source_detector = float(geometry.source_detector)

        self._default_dtype = dtype or torch.float32

        det_size = geometry.effective_det_spacing
        binshift = geometry.effective_det_shift
        shx = shx * sx
        shy = shy * sy

        self.det_size = det_size
        self.binshift = binshift
        self.pixshift = (shx, shy)

        options = torch.tensor(
            [W, H, self.num_det, sx, sy, det_size,
             self.iso_source, self.source_detector, shx, shy, binshift, geometry.scan_id],
            dtype=self._default_dtype,
        )
        
        self.register_buffer("options", options, persistent=False)
        weight_1d = self._make_cosine_weight(dtype=self._default_dtype)
        if trainable:
            self.weight = nn.parameter.Parameter(weight_1d.clone())
        else:
            self.register_buffer("weight", weight_1d, persistent=False)

        filt_1d = self._filter_generate(filter_type=filter_type, dtype=self._default_dtype)
        if trainable:
            self.filter = nn.parameter.Parameter(filt_1d.clone())
        else:
            self.register_buffer("filter", filt_1d, persistent=False)

    # ---------- public API ----------
    def forward(self, image: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        return self.projection(image, angles)
        
    def projection(self, image: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        _, _, h, w = image.shape
        assert (w, h) == self.image_size, f"Expected input image size {self.image_size[::-1]}, but got {(h, w)}"
        angles = self._prepare_angles(angles, image)
        return Projection2d.apply(image, self.options, angles)

    def backprojection(self, sino: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        _, _, h, w = sino.shape
        v, = angles.shape
        assert (h, w) == (v, self.num_det), f"Expected input sino size {(v, self.num_det)}, but got {(h, w)}"
        angles = self._prepare_angles(angles, sino)
        return BackProjection2d.apply(sino, self.options, angles)

    def backprojection_weighted(self, sino: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        _, _, h, w = sino.shape
        v, = angles.shape
        assert (h, w) == (v, self.num_det), f"Expected input sino size {(v, self.num_det)}, but got {(h, w)}"
        angles = self._prepare_angles(angles, sino)
        return WeightedBackProjection2d.apply(sino, self.options, angles)

    def projection_t(self, sino: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        self._check_sino_angles(sino, angles)
        angles = self._prepare_angles(angles, sino)
        return ProjectionTranspose2d.apply(sino, self.options, angles)

    def backprojection_t(self, image: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        _, _, h, w = image.shape
        assert (w, h) == self.image_size, f"Expected input image size {self.image_size[::-1]}, but got {(h, w)}"
        angles = self._prepare_angles(angles, image)
        return BackProjectionTranspose2d.apply(image, self.options, angles)

    def backprojection_weighted_t(self, image: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        _, _, h, w = image.shape
        assert (w, h) == self.image_size, f"Expected input image size {self.image_size[::-1]}, but got {(h, w)}"
        angles = self._prepare_angles(angles, image)
        return WeightedBackProjectionTranspose2d.apply(image, self.options, angles)

    projection_transpose = projection_t
    backprojection_transpose = backprojection_t
    weighted_backprojection_transpose = backprojection_weighted_t

    def system_matrix(self, angles: torch.Tensor) -> torch.Tensor:
        angles = angles.to(device=self.options.device, dtype=self.options.dtype).contiguous()
        return system_matrix2d(self.options.cpu().contiguous(), angles.cpu())

    get_sys_matrix = system_matrix
    
    def filtered_backprojection(
        self, sino: Tensor, 
        angles: Tensor, 
        filter: Union[Tensor, None] = None, 
        filter_type: Union[str, None] = None, 
        redundant: Optional[bool] = None,
        parker: bool = False,
    ) -> Tensor:
        self._check_sino_angles(sino, angles)
        
        if filter is not None:
            _filter = filter
        else:             
            if filter_type is None or filter_type.lower() == self.filter_type:
                _filter = self.filter
            else:
                _filter = self._filter_generate(filter_type=filter_type.lower(), dtype=sino.dtype).to(sino.device)
        if parker:
            sino = self.apply_parker_weight(sino, angles)
            if redundant is None:
                redundant = False
        elif redundant is None:
            redundant = True

        sino = sino * self.weight.to(device=sino.device, dtype=sino.dtype)
        filtered_sino = nn.functional.conv2d(sino, _filter, padding=(0, self.num_det - 1))
        recon = self.backprojection_weighted(filtered_sino, angles)
        recon = recon * self._mean_angle_step(angles).to(device=recon.device, dtype=recon.dtype)
        if redundant:
            recon = recon / 2
        return recon

    def apply_parker_weight(self, sino: Tensor, angles: Tensor) -> Tensor:
        return sino * self.parker_weight(angles).to(device=sino.device, dtype=sino.dtype)

    def parker_weight(self, angles: Tensor) -> Tensor:
        if self.scan_type not in {"flat", "arc"}:
            return torch.ones((1, 1, angles.numel(), self.num_det), device=angles.device, dtype=angles.dtype)
        if angles.numel() < 2:
            return torch.ones((1, 1, angles.numel(), self.num_det), device=angles.device, dtype=angles.dtype)

        beta = angles - angles[0]
        scan_range = beta[-1].abs()
        gamma = self._fan_angles(device=angles.device, dtype=angles.dtype)
        gamma_max = gamma.abs().max()
        if scan_range >= (2 * math.pi - 1e-4):
            return torch.ones((1, 1, angles.numel(), self.num_det), device=angles.device, dtype=angles.dtype)

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
        w = torch.ones((angles.numel(), self.num_det), device=angles.device, dtype=angles.dtype)
        left_mask = beta2d < left[None, :]
        right_mask = beta2d > right[None, :]
        left_den = torch.clamp(left, min=eps)
        right_den = torch.clamp(scan_range - right, min=eps)
        w_left = torch.sin(0.5 * math.pi * beta2d / left_den[None, :]).pow(2)
        w_right = torch.sin(0.5 * math.pi * (scan_range - beta2d) / right_den[None, :]).pow(2)
        w = torch.where(left_mask, w_left, w)
        w = torch.where(right_mask, w_right, w)
        return w.clamp_(0.0, 1.0).view(1, 1, angles.numel(), self.num_det)
    
    
    def _filter_generate(self, filter_type: str, dtype=None) -> Tensor:
        det_size = self.det_size
        num_det = self.num_det
        filter = filter_generate(num_det, det_size, filter_type, dtype)
        return filter[None, None, None, :]

    def _make_cosine_weight(
        self,
        dtype=None,
    ) -> torch.Tensor:
        
        det_size = self.det_size
        num_det = self.num_det
        binshift = self.binshift
        iso_source = self.iso_source
        dtype = dtype or torch.float32
        u = torch.arange(-num_det / 2 + 0.5, num_det / 2, 1.0, dtype=dtype)
        s = u * det_size + binshift

        if self.scan_type == "flat":
            w = iso_source / torch.sqrt(s * s + iso_source * iso_source)
        elif self.scan_type == "arc":
            w = torch.cos(s)
        elif self.scan_type == "parallel":
            w = torch.ones_like(s)
        else:
            raise ValueError(f"Unknown scan_type: {self.scan_type}")

        return w.view(1, 1, 1, num_det)

    def _fan_angles(self, device=None, dtype=None) -> torch.Tensor:
        dtype = dtype or torch.float32
        u = torch.arange(-self.num_det / 2 + 0.5, self.num_det / 2, 1.0, device=device, dtype=dtype)
        s = u * self.det_size + self.binshift
        if self.scan_type == "arc":
            return s
        if self.scan_type == "flat":
            return torch.atan2(s, torch.as_tensor(self.iso_source, device=device, dtype=dtype))
        return torch.zeros_like(s)

    @staticmethod
    def _prepare_angles(angles: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return angles.to(device=ref.device, dtype=ref.dtype).contiguous()

    def _check_sino_angles(self, sino: torch.Tensor, angles: torch.Tensor) -> None:
        if sino.ndim != 4:
            raise ValueError(f"Expected sino shape [B, C, V, Nu], got {tuple(sino.shape)}")
        if angles.ndim != 1:
            raise ValueError(f"Expected angles shape [V], got {tuple(angles.shape)}")
        if sino.shape[2] != angles.numel() or sino.shape[3] != self.num_det:
            raise ValueError(
                f"Expected sino shape [B, C, {angles.numel()}, {self.num_det}], got {tuple(sino.shape)}"
            )

    @staticmethod
    def _mean_angle_step(angles: torch.Tensor) -> torch.Tensor:
        """
        Mean wrapped angular step in radians.
        """
        if angles.numel() < 2:
            return angles.new_tensor(0.0)
        a = angles
        diffs = torch.diff(a)
        wrap = (diffs + math.pi) % (2 * math.pi) - math.pi
        return wrap.abs().mean()


projector2d = Projector2d
