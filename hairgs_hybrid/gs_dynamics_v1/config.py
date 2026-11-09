"""Editable, asset-coordinate presets; these are artistic, not measured materials."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math


@dataclass(frozen=True)
class DynamicsConfig:
    motion_profile: str = "expanded"
    controller_count: int = 128
    neighbors: int = 4
    fixed_dt: float = 1.0 / 120.0
    frequency_hz: float = 2.0
    damping_ratio: float = 0.72
    max_angle_degrees: float = 18.0
    wind_strength: float = 1.0
    wind_direction: tuple[float, float, float] = (0.75, 0.5, 0.12)
    wind_start: float = 0.25
    wind_stop: float = 3.5
    gust_hz: float = 0.7
    # Z-up lion in the supplied complete asset's coordinate system.
    mane_y_min: float = -0.83
    mane_y_max: float = -0.07
    mane_z_min: float = 0.42
    face_y_cut: float = -0.74
    crest_z_min: float = 0.94
    region_feather: float = 0.04
    tail_y_min: float = 0.90
    tail_z_min: float = 0.08
    tail_z_max: float = 0.32
    enable_tail: bool = True
    max_base_offset: float = 0.085
    max_node_extent: float = 0.23
    max_chain_length: float = 0.28
    max_gaussian_sigma: float = 0.05
    min_strand_outward_distance: float = -0.025
    shell_pin_distance: float = 0.005
    shell_full_distance: float = 0.025
    # Swing strands whose base floats above the skin around their skin anchor
    # (bend fraction measured from the skin), so thick mane layers move as a mass.
    skin_rooted_strands: bool = False

    @classmethod
    def for_profile(cls, name):
        """Calibrated coverage choices for the supplied Z-up lion asset."""
        if name == "expanded":
            return cls().validate()
        if name == "conservative":
            return cls(motion_profile=name, mane_y_min=-.74, mane_y_max=-.12,
                       mane_z_min=.50, face_y_cut=-.60, crest_z_min=.98,
                       max_base_offset=.06, max_node_extent=.20,
                       min_strand_outward_distance=-.005,
                       shell_full_distance=.04).validate()
        if name == "full":
            # Region and facial guard come from rig bone semantics when given
            # (see build_bindings); the box fields are the no-rig fallback.
            # Long/large mane strands are admitted; FK keeps roots fixed.
            return cls(motion_profile=name, controller_count=192,
                       max_base_offset=.12, max_node_extent=.30, max_chain_length=.28,
                       max_gaussian_sigma=.06, min_strand_outward_distance=-.06,
                       shell_pin_distance=.003, shell_full_distance=.02,
                       skin_rooted_strands=True).validate()
        raise ValueError("motion_profile must be conservative, expanded or full")

    def validate(self):
        if self.motion_profile not in ("conservative", "expanded", "full"):
            raise ValueError("motion_profile must be conservative, expanded or full")
        for key, value in asdict(self).items():
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if (type(self.controller_count) is not int or type(self.neighbors) is not int
                or self.controller_count < 1 or self.neighbors < 1):
            raise ValueError("controller_count and neighbors must be positive")
        if not 0 < self.fixed_dt <= 1 / 30 or self.frequency_hz <= 0:
            raise ValueError("invalid fixed timestep or frequency")
        if not 0 < self.damping_ratio < 1:
            raise ValueError("this version requires an underdamped ratio in (0,1)")
        if self.wind_stop <= self.wind_start or self.wind_start < 0:
            raise ValueError("wind_stop must follow nonnegative wind_start")
        if self.max_angle_degrees < 0 or self.region_feather <= 0:
            raise ValueError("invalid angle or region feather")
        if not isinstance(self.skin_rooted_strands, bool):
            raise ValueError("skin_rooted_strands must be true or false")
        if self.shell_full_distance <= self.shell_pin_distance:
            raise ValueError("shell_full_distance must exceed shell_pin_distance")
        if len(self.wind_direction) != 3 or not all(math.isfinite(x) for x in self.wind_direction):
            raise ValueError("wind_direction must be a finite 3-vector")
        if sum(x * x for x in self.wind_direction) <= 0:
            raise ValueError("wind_direction must be nonzero")
        for key in ["max_base_offset", "max_node_extent", "max_chain_length", "max_gaussian_sigma"]:
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        # Saved v1 offline configs had all mask parameters but no profile tag.
        # Keep their original facial guard when replaying an archived result.
        profile = value.get("motion_profile", "conservative" if "mane_y_min" in value else "expanded")
        return cls(**{**cls.for_profile(profile).to_dict(), **value}).validate()
