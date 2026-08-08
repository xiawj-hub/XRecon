from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Union

Number = Union[int, float]
PairLike = Union[Number, Tuple[Number, Number], List[Number]]
TripleLike = Union[Number, Tuple[Number, Number, Number], List[Number]]


def to_2tuple(value: PairLike, name: str) -> Tuple[Number, Number]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"{name} should be a scalar or a 2-tuple/list")
        return value[0], value[1]
    return value, value


def to_3tuple(value: TripleLike, name: str) -> Tuple[Number, Number, Number]:
    if isinstance(value, (tuple, list)):
        if len(value) != 3:
            raise ValueError(f"{name} should be a scalar or a 3-tuple/list")
        return value[0], value[1], value[2]
    return value, value, value


@dataclass(frozen=True)
class Volume2d:
    size: Tuple[int, int]
    spacing: Tuple[float, float]
    shift: Tuple[float, float] = (0.0, 0.0)

    def __init__(self, size: PairLike, spacing: PairLike, shift: PairLike = 0.0) -> None:
        sx, sy = to_2tuple(size, "size")
        dx, dy = to_2tuple(spacing, "spacing")
        shx, shy = to_2tuple(shift, "shift")
        object.__setattr__(self, "size", (int(sx), int(sy)))
        object.__setattr__(self, "spacing", (float(dx), float(dy)))
        object.__setattr__(self, "shift", (float(shx), float(shy)))


@dataclass(frozen=True)
class Detector2d:
    count: int
    spacing: float
    shift: float = 0.0

    def __init__(self, count: int, spacing: float, shift: float = 0.0) -> None:
        object.__setattr__(self, "count", int(count))
        object.__setattr__(self, "spacing", float(spacing))
        object.__setattr__(self, "shift", float(shift))


@dataclass(frozen=True)
class ParallelBeam2d:
    detector: Detector2d
    scan_type: str = "parallel"
    iso_source: float = 0.0
    source_detector: float = 0.0

    @property
    def scan_id(self) -> int:
        return 0

    @property
    def effective_det_spacing(self) -> float:
        return self.detector.spacing

    @property
    def effective_det_shift(self) -> float:
        return self.detector.shift * self.effective_det_spacing


@dataclass(frozen=True)
class FanBeam2d:
    detector: Detector2d
    iso_source: float
    source_detector: float
    detector_type: str = "flat"

    def __init__(
        self,
        detector: Detector2d,
        iso_source: float,
        source_detector: float,
        detector_type: str = "flat",
    ) -> None:
        detector_type = detector_type.lower().strip()
        if detector_type not in {"flat", "arc"}:
            raise ValueError("detector_type should be 'flat' or 'arc'")
        object.__setattr__(self, "detector", detector)
        object.__setattr__(self, "iso_source", float(iso_source))
        object.__setattr__(self, "source_detector", float(source_detector))
        object.__setattr__(self, "detector_type", detector_type)

    @property
    def scan_type(self) -> str:
        return self.detector_type

    @property
    def scan_id(self) -> int:
        return {"flat": 1, "arc": 2}[self.detector_type]

    @property
    def effective_det_spacing(self) -> float:
        if self.detector_type == "flat":
            return self.detector.spacing * self.iso_source / self.source_detector
        return 2.0 * math.atan(self.detector.spacing / self.source_detector / 2.0)

    @property
    def effective_det_shift(self) -> float:
        return self.detector.shift * self.effective_det_spacing


Geometry2d = Union[ParallelBeam2d, FanBeam2d]


def make_geometry2d(
    scan_type: str,
    num_det: int,
    det_size: float,
    iso_source: float,
    source_detector: float,
    binshift: float = 0.0,
) -> Geometry2d:
    scan_type = scan_type.lower().strip()
    if scan_type == "para":
        scan_type = "parallel"
    detector = Detector2d(num_det, det_size, binshift)
    if scan_type == "parallel":
        return ParallelBeam2d(detector)
    if scan_type in {"flat", "arc"}:
        return FanBeam2d(detector, iso_source, source_detector, scan_type)
    raise ValueError(f"Unsupported scan_type: {scan_type}")


@dataclass(frozen=True)
class Volume3d:
    size: Tuple[int, int, int]
    spacing: Tuple[float, float, float]
    shift: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __init__(self, size: TripleLike, spacing: TripleLike, shift: TripleLike = 0.0) -> None:
        sx, sy, sz = to_3tuple(size, "size")
        dx, dy, dz = to_3tuple(spacing, "spacing")
        shx, shy, shz = to_3tuple(shift, "shift")
        object.__setattr__(self, "size", (int(sx), int(sy), int(sz)))
        object.__setattr__(self, "spacing", (float(dx), float(dy), float(dz)))
        object.__setattr__(self, "shift", (float(shx), float(shy), float(shz)))


@dataclass(frozen=True)
class Detector3d:
    count: Tuple[int, int]
    spacing: Tuple[float, float]
    shift: Tuple[float, float] = (0.0, 0.0)

    def __init__(self, count: PairLike, spacing: PairLike, shift: PairLike = 0.0) -> None:
        nu, nv = to_2tuple(count, "count")
        du, dv = to_2tuple(spacing, "spacing")
        shu, shv = to_2tuple(shift, "shift")
        object.__setattr__(self, "count", (int(nu), int(nv)))
        object.__setattr__(self, "spacing", (float(du), float(dv)))
        object.__setattr__(self, "shift", (float(shu), float(shv)))


@dataclass(frozen=True)
class ConeBeam3d:
    detector: Detector3d
    iso_source: float
    source_detector: float
    detector_type: str = "flat"

    def __init__(
        self,
        detector: Detector3d,
        iso_source: float,
        source_detector: float,
        detector_type: str = "flat",
    ) -> None:
        detector_type = detector_type.lower().strip()
        if detector_type not in {"flat", "arc"}:
            raise ValueError("detector_type should be 'flat' or 'arc'")
        object.__setattr__(self, "detector", detector)
        object.__setattr__(self, "iso_source", float(iso_source))
        object.__setattr__(self, "source_detector", float(source_detector))
        object.__setattr__(self, "detector_type", detector_type)

    @property
    def scan_type(self) -> str:
        return self.detector_type

    @property
    def scan_id(self) -> int:
        return {"flat": 1, "arc": 2}[self.detector_type]

    @property
    def effective_det_spacing(self) -> Tuple[float, float]:
        du, dv = self.detector.spacing
        if self.detector_type == "flat":
            return du * self.iso_source / self.source_detector, dv
        return du / self.source_detector, dv

    @property
    def effective_det_shift(self) -> Tuple[float, float]:
        du, dv = self.effective_det_spacing
        shu, shv = self.detector.shift
        return shu * du, shv * dv


@dataclass(frozen=True)
class ParallelBeam3d:
    detector: Detector3d
    scan_type: str = "parallel"

    @property
    def scan_id(self) -> int:
        return 0

    @property
    def effective_det_spacing(self) -> Tuple[float, float]:
        return self.detector.spacing

    @property
    def effective_det_shift(self) -> Tuple[float, float]:
        du, dv = self.detector.spacing
        shu, shv = self.detector.shift
        return shu * du, shv * dv


@dataclass(frozen=True)
class HelicalConeBeam3d(ConeBeam3d):
    pitch: float = 0.0
    z0: float = 0.0

    def __init__(
        self,
        detector: Detector3d,
        iso_source: float,
        source_detector: float,
        detector_type: str = "arc",
        pitch: float = 0.0,
        z0: float = 0.0,
    ) -> None:
        super().__init__(detector, iso_source, source_detector, detector_type)
        object.__setattr__(self, "pitch", float(pitch))
        object.__setattr__(self, "z0", float(z0))


Geometry3d = Union[ParallelBeam3d, ConeBeam3d, HelicalConeBeam3d]


def make_geometry3d(
    scan_type: str,
    num_det: PairLike,
    det_size: PairLike,
    iso_source: float,
    source_detector: float,
    binshift: PairLike = 0.0,
) -> Geometry3d:
    scan_type = scan_type.lower().strip()
    if scan_type == "para":
        scan_type = "parallel"
    detector = Detector3d(num_det, det_size, binshift)
    if scan_type == "parallel":
        return ParallelBeam3d(detector)
    if scan_type in {"flat", "arc"}:
        return ConeBeam3d(detector, iso_source, source_detector, scan_type)
    raise ValueError(f"Unsupported 3D scan_type: {scan_type}")
