"""Bounded-memory, fixed-step live controls for the interactive GS viewer."""
from __future__ import annotations

import math
import numpy as np

from .config import DynamicsConfig


PARAMETER_RANGES = {
    "wind_strength": (0., 3.), "wind_azimuth": (-180., 180.),
    "wind_elevation": (-80., 80.), "gust_strength": (0., 1.),
    "gust_hz": (0., 5.), "max_angle_degrees": (0., 45.),
    "frequency_hz": (.5, 6.), "damping_ratio": (.1, .98), "speed": (0., 2.),
}
DEFAULT_PARAMETERS = dict(wind_strength=1., wind_azimuth=34., wind_elevation=8.,
                          gust_strength=.3, gust_hz=.7, max_angle_degrees=18.,
                          frequency_hz=2., damping_ratio=.72, speed=1.)


def validate_settings(value):
    if not isinstance(value, dict):
        raise ValueError("live settings must be an object")
    unknown = set(value)-set(PARAMETER_RANGES)
    if unknown:
        raise ValueError(f"unknown live settings: {sorted(unknown)}")
    result = DEFAULT_PARAMETERS.copy()
    for key, raw in value.items():
        if isinstance(raw, bool) or not isinstance(raw, (float, int)):
            raise ValueError(f"{key} must be a number")
        x = float(raw)
        low, high = PARAMETER_RANGES[key]
        if not math.isfinite(x) or not low <= x <= high:
            raise ValueError(f"{key} must be in [{low}, {high}]")
        result[key] = x
    return result


class LiveWindDriver:
    def __init__(self, asset, bindings, config=None):
        self.config = (config or DynamicsConfig()).validate()
        self.fixed_dt = self.config.fixed_dt
        roots = asset.guide_anchors[bindings.controller_guide_ids]
        self.normals = asset.guide_normals[bindings.controller_guide_ids].astype(np.float64)
        self.phase = roots @ np.array([7.3, 4.1, 3.7])
        self.angles = np.zeros((len(roots), 3), np.float64)
        self.velocity = np.zeros_like(self.angles)
        self.time = 0.
        self._step = 0
        self._accumulator = 0.
        self.dropped_time = 0.
        self.parameters = validate_settings({"wind_strength": self.config.wind_strength,
                                              "max_angle_degrees": self.config.max_angle_degrees,
                                              "frequency_hz": self.config.frequency_hz,
                                              "damping_ratio": self.config.damping_ratio,
                                              "gust_hz": self.config.gust_hz})
        self._update_coefficients()

    def _update_coefficients(self):
        p = self.parameters
        omega = (2*np.pi*p["frequency_hz"]*(1+.12*np.sin(self.phase)))[:, None]
        decay = p["damping_ratio"]*omega
        wd = omega*np.sqrt(1-p["damping_ratio"]**2)
        si, co = np.sin(wd*self.fixed_dt), np.cos(wd*self.fixed_dt)
        ex = np.exp(-decay*self.fixed_dt)
        self._a = ex*(co+decay/wd*si)
        self._b = ex*si/wd
        self._c = -ex*omega**2/wd*si
        self._d = ex*(co-decay/wd*si)

    def set_parameters(self, value):
        updated = validate_settings({**self.parameters, **value})
        if updated != self.parameters:
            self.parameters = updated
            self._update_coefficients()

    def _target(self, time):
        p = self.parameters
        az, el = np.deg2rad([p["wind_azimuth"], p["wind_elevation"]])
        direction = np.array([np.cos(az)*np.cos(el), np.sin(az)*np.cos(el), np.sin(el)])
        gust = 1+p["gust_strength"]*(.75*np.sin(2*np.pi*p["gust_hz"]*time+.25*self.phase)
                                     +.25*np.sin(2*np.pi*1.7*time+self.phase))
        wind = gust[:, None]*direction
        wind[:, 1] += .16*p["gust_strength"]*np.sin(2*np.pi*.45*time+self.phase)
        target = np.cross(self.normals, wind)*p["wind_strength"]*np.deg2rad(p["max_angle_degrees"])
        magnitude = np.linalg.norm(target, axis=1, keepdims=True)
        target *= np.minimum(1., np.deg2rad(p["max_angle_degrees"])/np.maximum(magnitude, 1e-15))
        return target

    def advance(self, elapsed_seconds, *, paused=False):
        dt = float(elapsed_seconds)
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("elapsed_seconds must be finite and nonnegative")
        if paused or self.parameters["speed"] == 0:
            return self.angles.copy()
        # Window drags/OS stalls must not trigger an unbounded catch-up loop.
        limited = min(dt, .25)
        self.dropped_time += dt-limited
        self._accumulator += limited*self.parameters["speed"]
        count = min(60, int((self._accumulator+1e-12)/self.fixed_dt))
        for _ in range(count):
            target = self._target((self._step+.5)*self.fixed_dt)
            y = self.angles-target
            angle = target+self._a*y+self._b*self.velocity
            self.velocity[:] = self._c*y+self._d*self.velocity
            self.angles[:] = angle
            self._step += 1
        self._accumulator = max(0., self._accumulator-count*self.fixed_dt)
        if self._accumulator >= self.fixed_dt:
            excess = math.floor(self._accumulator/self.fixed_dt)*self.fixed_dt
            self._accumulator -= excess
            self.dropped_time += excess/self.parameters["speed"]
        self.time = self._step*self.fixed_dt
        return self.angles.copy()

    def reset(self):
        self.angles.fill(0.)
        self.velocity.fill(0.)
        self.time = 0.
        self._step = 0
        self._accumulator = 0.
        self.dropped_time = 0.
