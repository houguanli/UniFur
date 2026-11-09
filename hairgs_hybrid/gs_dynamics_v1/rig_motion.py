"""Clip playback and bounded artistic wind/inertia in moving body frames."""
from __future__ import annotations

import math
import numpy as np

from .live_driver import LiveWindDriver


class AnimationClock:
    """Play a non-periodic source action without a discontinuous last/first jump.

    Ping-pong keeps authored source timing everywhere except a short raised-
    cosine speed ramp of ``turn_seconds`` at each end, so body velocity (and
    the hair inertia it drives) stays continuous when playback reverses.
    ``loop`` is only seamless for clips marked cyclic (e.g. in-place runs).
    """
    MODES = ("pingpong", "once", "loop")

    def __init__(self, rig, turn_seconds=.5):
        self.fps = float(rig.fps)
        self.first_frame = float(rig.frame_numbers[0])
        self.last_frame = float(rig.frame_numbers[-1])
        self.duration = (self.last_frame-self.first_frame)/self.fps
        if self.duration <= 0:
            raise ValueError("Animation needs at least two distinct source frames")
        if not math.isfinite(turn_seconds) or turn_seconds < 0:
            raise ValueError("Turnaround easing must be finite and nonnegative")
        self.turn = min(float(turn_seconds), .5*self.duration)
        # Phase length of one eased pass; each ramp covers half its length.
        self.leg = self.duration+self.turn
        self.phase = 0.
        self.mode = "pingpong"

    def _ramp(self, phase):
        return .5*phase-self.turn/(2*math.pi)*math.sin(math.pi*phase/self.turn)

    def _eased(self, phase):
        if self.turn == 0:
            return phase
        if phase < self.turn:
            return self._ramp(phase)
        if phase > self.duration:
            return self.duration-self._ramp(self.leg-phase)
        return phase-.5*self.turn

    def _eased_phase(self, time):
        """Inverse of ``_eased`` on one forward pass."""
        half = .5*self.turn
        if self.turn == 0 or half <= time <= self.duration-half:
            return time+half
        target = time if time < half else self.duration-time
        low, high = 0., self.turn
        for _ in range(60):
            middle = .5*(low+high)
            low, high = (middle, high) if self._ramp(middle) < target else (low, middle)
        phase = .5*(low+high)
        return phase if time < half else self.leg-phase

    @property
    def time(self):
        if self.mode == "once":
            return min(self.phase, self.duration)
        if self.mode == "loop":
            return self.phase % self.duration
        phase = self.phase % (2*self.leg)
        return self._eased(min(phase, 2*self.leg-phase))

    @property
    def rate(self):
        """Source seconds per playback second right now (signed, before speed)."""
        if self.mode == "loop":
            return 1.
        if self.mode == "once":
            return float(self.phase < self.duration)
        phase = self.phase % (2*self.leg)
        sign, phase = (1., phase) if phase < self.leg else (-1., 2*self.leg-phase)
        if self.turn == 0 or self.turn <= phase <= self.duration:
            return sign
        edge = phase if phase < self.turn else self.leg-phase
        return sign*.5*(1-math.cos(math.pi*edge/self.turn))

    @property
    def frame(self):
        return self.first_frame+self.time*self.fps

    def advance(self, elapsed, *, enabled=True, paused=False, speed=1., mode="pingpong"):
        if mode not in self.MODES:
            raise ValueError("Animation mode must be pingpong, once or loop")
        if not math.isfinite(elapsed) or elapsed < 0 or not math.isfinite(speed) or not 0 <= speed <= 2:
            raise ValueError("Invalid playback elapsed time or speed")
        if mode != self.mode:
            frame = self.frame
            self.mode = mode
            self.seek(frame)
        if enabled and not paused:
            self.phase += min(float(elapsed), .25)*speed
            period = {"once": None, "loop": self.duration}.get(self.mode, 2*self.leg)
            self.phase = min(self.phase, self.duration) if period is None else self.phase % period
        return self.time

    def seek(self, frame):
        frame = float(frame)
        if not math.isfinite(frame):
            raise ValueError("Animation frame must be finite")
        t = float(np.clip((frame-self.first_frame)/self.fps, 0., self.duration))
        self.phase = self._eased_phase(t) if self.mode == "pingpong" else t

    def reset(self):
        self.phase = 0.


def _angular_velocity(current, previous, dt):
    delta = current @ np.swapaxes(previous, -1, -2)
    angle = np.arccos(np.clip((np.trace(delta, axis1=1, axis2=2)-1)*.5, -1., 1.))
    skew = .5*np.stack((delta[:, 2, 1]-delta[:, 1, 2],
                         delta[:, 0, 2]-delta[:, 2, 0],
                         delta[:, 1, 0]-delta[:, 0, 1]), axis=1)
    factor = np.ones_like(angle)
    normal = angle > 1e-6
    factor[normal] = angle[normal]/np.maximum(np.sin(angle[normal]), 1e-8)
    result = skew*factor[:, None]/dt
    # Large discontinuities come from scrubbing, never continuous playback.
    # Bound them here as a second guard if a caller forgot reset_body_history.
    norm = np.linalg.norm(result, axis=1, keepdims=True)
    return result*np.minimum(1., 8./np.maximum(norm, 1e-12))


def _rotate(vectors, rotvec):
    angle = np.linalg.norm(rotvec, axis=1, keepdims=True)
    axis = rotvec/np.maximum(angle, 1e-15)
    cos, sin = np.cos(angle), np.sin(angle)
    return (vectors*cos+np.cross(axis, vectors)*sin
            + axis*np.sum(axis*vectors, axis=1, keepdims=True)*(1-cos))


def _fur_directions(asset, bindings, normals):
    """Controller-weighted mean root-to-Gaussian direction of the active fur."""
    rows = np.flatnonzero(~bindings.frozen_rows)
    guides = asset.gs_guide[rows]
    offsets = asset.xyz[rows].astype(np.float64)-asset.guide_anchors[guides]
    weights = bindings.control_weights[guides]*bindings.guide_weight[guides, None]
    total = np.zeros((len(bindings.controller_guide_ids), 3))
    np.add.at(total, bindings.control_ids[guides].ravel(), (weights[:, :, None]*offsets[:, None, :]).reshape(-1, 3))
    length = np.linalg.norm(total, axis=1, keepdims=True)
    direction = total/np.maximum(length, 1e-15)
    # Flat or cancelling regions (and controllers without fur) use the normal.
    bad = (length[:, 0] < 1e-9) | (np.sum(direction*normals, axis=1) < .05)
    direction[bad] = normals[bad]
    return direction


def _soft_limit(vectors, limit):
    """Nearly linear below ``limit`` and smoothly saturating at it."""
    if limit <= 0:
        return np.zeros_like(vectors)
    magnitude = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors/np.power(1+(magnitude/limit)**4, .25)


class RigWindDriver(LiveWindDriver):
    """Existing 120Hz damped controls driven by wind, drag, inertia and gravity.

    Angles live in each controller's trained-pose (local) frame; the rig
    renderer carries them with the animated surface. Air drag from wind and
    body velocity saturates softly at the bend limit (hair streams back).
    Root acceleration and gravity enter as a pendulum of length ``hair_length``
    on the controller's own spring (deflection a/(L*omega^2)), unclipped by the
    drag limit so gait bounce stays visible; the total is softly capped at
    twice the bend limit. Stiffness and length are artistic, not measured.
    Torques act on each controller's mean fur direction, so fur turns toward
    the flow and stops when aligned; a skin guard shrinks any target bend that
    would tilt that direction below ``skin_margin_degrees`` above the surface.
    """
    GRAVITY = np.array([0., 0., -9.81])  # source units are ~metres, +Z up

    def __init__(self, asset, bindings, config=None, *, hair_length=.15, skin_margin_degrees=5.):
        super().__init__(asset, bindings, config)
        self.fur_directions = _fur_directions(asset, bindings, self.normals)
        self.skin_margin = np.sin(np.deg2rad(skin_margin_degrees))
        k = len(self.angles)
        self.body_scale = max(float(np.linalg.norm(np.ptp(asset.body_vertices, axis=0))), 1e-6)
        self.hair_length = float(hair_length)
        self.body_rotations = np.tile(np.eye(3), (k, 1, 1))
        self.body_normals = self.normals.copy()
        self.root_velocity = np.zeros((k, 3))
        self.root_acceleration = np.zeros((k, 3))
        self.angular_velocity = np.zeros((k, 3))
        self.angular_acceleration = np.zeros((k, 3))
        self.frame_velocity = np.zeros(3)
        self.inertia_strength = 1.
        self._previous_roots = self._previous_rotations = None

    def reset_body_history(self):
        self._previous_roots = self._previous_rotations = None
        for value in (self.root_velocity, self.root_acceleration,
                      self.angular_velocity, self.angular_acceleration):
            value.fill(0.)

    def set_body_state(self, anchors, rotations, normals, elapsed, *, inertia_strength=1., paused=False,
                       frame_velocity=(0., 0., 0.)):
        """``frame_velocity``: world velocity removed from the displayed clip
        (treadmill playback); it adds air drag but no acceleration."""
        roots, rotation, normal = (np.asarray(value, np.float64) for value in (anchors, rotations, normals))
        carried = np.asarray(frame_velocity, np.float64)
        if roots.shape != self.angles.shape or normal.shape != roots.shape or rotation.shape != (len(roots), 3, 3):
            raise ValueError("Body controller frame shape mismatch")
        if carried.shape != (3,) or not all(np.isfinite(value).all() for value in (roots, rotation, normal, carried)):
            raise ValueError("Body controller frames must be finite")
        if not math.isfinite(inertia_strength) or not 0 <= inertia_strength <= 2:
            raise ValueError("Hair inertia must be between 0 and 2")
        self.inertia_strength = float(inertia_strength)
        self.body_rotations, self.body_normals = rotation.copy(), normal.copy()
        self.frame_velocity = carried.copy()
        if paused:
            return
        dt = float(elapsed)
        if self._previous_roots is not None and 1e-6 < dt <= .25:
            velocity = (roots-self._previous_roots)/dt
            omega = _angular_velocity(rotation, self._previous_rotations, dt)
            # Short enough to keep a ~2-3 Hz gait, long enough to hide the
            # piecewise-linear interpolation of 30 fps source frames.
            blend = -math.expm1(-dt/.04)
            old_v, old_w = self.root_velocity.copy(), self.angular_velocity.copy()
            self.root_velocity += blend*(velocity-self.root_velocity)
            self.angular_velocity += blend*(omega-self.angular_velocity)
            acc = (self.root_velocity-old_v)/dt
            alpha = (self.angular_velocity-old_w)/dt
            self.root_acceleration += blend*(acc-self.root_acceleration)
            self.angular_acceleration += blend*(alpha-self.angular_acceleration)
        elif self._previous_roots is not None:
            self.reset_body_history()
        self._previous_roots, self._previous_rotations = roots.copy(), rotation.copy()

    def _target(self, time):
        p = self.parameters
        limit = np.deg2rad(p["max_angle_degrees"])
        az, el = np.deg2rad([p["wind_azimuth"], p["wind_elevation"]])
        direction = np.array([np.cos(az)*np.cos(el), np.sin(az)*np.cos(el), np.sin(el)])
        gust = 1+p["gust_strength"]*(.75*np.sin(2*np.pi*p["gust_hz"]*time+.25*self.phase)
                                     +.25*np.sin(2*np.pi*1.7*time+self.phase))
        world_air = p["wind_strength"]*gust[:, None]*direction
        world_air[:, 1] += p["wind_strength"]*.16*p["gust_strength"]*np.sin(2*np.pi*.45*time+self.phase)
        world_air -= self.inertia_strength*.8*(self.root_velocity+self.frame_velocity)/self.body_scale
        inv = np.swapaxes(self.body_rotations, -1, -2)
        fur = self.fur_directions
        drag = _soft_limit(np.cross(fur, np.einsum("nij,nj->ni", inv, world_air))*limit, limit)
        # Base acceleration and gravity change, seen in the trained-pose frame.
        omega = 2*np.pi*p["frequency_hz"]*(1+.12*np.sin(self.phase))
        # Inertia 0 means rigid attachment, including no gravity re-sag.
        field = self.inertia_strength*(np.einsum("nij,nj->ni", inv, self.GRAVITY-self.root_acceleration)-self.GRAVITY)
        inertial = np.cross(fur, field)/(self.hair_length*omega**2)[:, None]
        inertial -= self.inertia_strength*np.einsum("nij,nj->ni", inv, self.angular_acceleration)/(omega**2)[:, None]
        return self._skin_guard(_soft_limit(drag+inertial, min(2*limit, np.deg2rad(75.))))

    def _skin_guard(self, target):
        """Largest fraction of each bend keeping the fur above the skin margin."""
        floor = np.minimum(np.sum(self.fur_directions*self.normals, axis=1), self.skin_margin)
        ok = np.sum(_rotate(self.fur_directions, target)*self.normals, axis=1) >= floor
        if ok.all():
            return target
        low, high = np.zeros(len(target)), np.ones(len(target))
        for _ in range(8):
            middle = .5*(low+high)
            safe = np.sum(_rotate(self.fur_directions, target*middle[:, None])*self.normals, axis=1) >= floor
            low, high = np.where(safe, middle, low), np.where(safe, high, middle)
        return target*np.where(ok, 1., low)[:, None]

    def reset(self):
        super().reset()
        self.reset_body_history()
