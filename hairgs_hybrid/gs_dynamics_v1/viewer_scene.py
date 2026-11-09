"""Worker-owned static scene, with lazy and optional UniFur fur simulation."""
from __future__ import annotations

import time
import numpy as np

from .bindings import build_bindings
from .config import DynamicsConfig
from .live_driver import LiveWindDriver, PARAMETER_RANGES


BACKGROUNDS = {"Light": (.94, .95, .97), "Dark": (.035, .045, .06),
               "Black": (0., 0., 0.), "White": (1., 1., 1.)}


class SimulationUnavailable(ValueError):
    pass


class ViewerScene:
    def __init__(self, asset, *, renderer_factory=None, rig_path=None):
        if renderer_factory is None:
            from .realtime_gpu import RealtimeGSRenderer
            renderer_factory = RealtimeGSRenderer
        self.asset, self._factory = asset, renderer_factory
        self.renderer = renderer_factory(asset)
        self.bindings = self.driver = self.binding_profile = None
        self.simulation_enabled = False
        self.rig = self.animation_clock = None
        self._last_rig_time = None
        self.last_stats = {}
        if rig_path is not None:
            try:
                self.attach_rig(rig_path)
            except Exception:
                self.renderer.close()
                raise

    def attach_rig(self, path):
        if not self.asset.can_simulate:
            raise ValueError("Body animation binding requires a complete UniFur NPZ")
        from .rig_asset import RigBinding
        from .rig_motion import AnimationClock, RigWindDriver
        from .rigged_gpu import RiggedGSRenderer
        rig = RigBinding.load(path, self.asset)
        profile = self.binding_profile or "expanded"
        config = DynamicsConfig.for_profile(profile)
        bindings = build_bindings(self.asset.complete_asset, config, semantics=rig.semantics)
        driver = RigWindDriver(self.asset.complete_asset, bindings, config)
        renderer = RiggedGSRenderer(self.asset.complete_asset, bindings, rig)
        previous = self.renderer
        self.rig, self.animation_clock = rig, AnimationClock(rig)
        self.renderer, self.bindings, self.driver = renderer, bindings, driver
        self.binding_profile, self._last_rig_time = profile, None
        previous.close()

    def clear_rig(self):
        if self.rig is None:
            return
        renderer = self._factory(self.asset)
        previous = self.renderer
        self.rig = self.animation_clock = self.bindings = self.driver = self.binding_profile = None
        self.renderer, self.simulation_enabled, self._last_rig_time = renderer, False, None
        previous.close()

    def seek_animation(self, frame):
        if self.animation_clock is None:
            raise ValueError("Load a body animation binding first")
        self.animation_clock.seek(frame)
        self._last_rig_time = None
        if self.driver is not None:
            self.driver.reset()

    def configure(self, state):
        enabled = state["simulation_enabled"]
        profile = state["motion_profile"]
        if enabled and not self.asset.can_simulate:
            raise SimulationUnavailable(self.asset.simulation_reason)
        if (enabled or self.rig is not None) and (self.bindings is None or self.binding_profile != profile):
            config = DynamicsConfig.for_profile(profile)
            try:
                bindings = build_bindings(self.asset.complete_asset, config,
                                          semantics=self.rig.semantics if self.rig is not None else None)
            except ValueError as error:
                raise SimulationUnavailable(f"This asset does not match the {profile} fur preset: {error}") from error
            if self.rig is None:
                driver = LiveWindDriver(self.asset.complete_asset, bindings, config)
                renderer = self._factory(self.asset.complete_asset, bindings)
            else:
                from .rig_motion import RigWindDriver
                from .rigged_gpu import RiggedGSRenderer
                driver = RigWindDriver(self.asset.complete_asset, bindings, config)
                renderer = RiggedGSRenderer(self.asset.complete_asset, bindings, self.rig)
                self._last_rig_time = None
            previous = self.renderer
            self.renderer, self.bindings, self.driver = renderer, bindings, driver
            self.binding_profile = profile
            previous.close()
        if not enabled and self.simulation_enabled and self.driver is not None:
            self.driver.reset()
        self.simulation_enabled = enabled
        if self.driver is not None:
            self.driver.set_parameters({key: state[key] for key in PARAMETER_RANGES})
        collision_enabled = enabled and state.get("collision_enabled", False)
        if collision_enabled:
            from .sdf_collision import FurBodyCollision
            settings = (state.get("collision_resolution", 128), state.get("collision_margin", .003))
            collision = self.renderer.collision
            if collision is None or (collision.field.resolution, collision.margin) != settings:
                self.renderer.collision = FurBodyCollision(self.asset.complete_asset, self.bindings,
                    resolution=settings[0], margin=settings[1], device=self.renderer.device)
        elif hasattr(self.renderer, "collision"):
            self.renderer.collision = None

    def render(self, state, camera, elapsed):
        self.configure(state)
        tick = time.perf_counter()
        if self.rig is not None:
            mode = state.get("animation_mode", "pingpong")
            if mode == "loop" and not self.rig.loop:
                mode = "pingpong"  # a raw loop of a non-cyclic clip jumps at the seam
            playing = state.get("animation_enabled", True) and not state["paused"]
            speed = state.get("animation_speed", 1.)
            clip_time = self.animation_clock.advance(elapsed, enabled=playing, speed=speed, mode=mode)
            if clip_time != self._last_rig_time:
                self.renderer.set_rig_pose(self.rig.sample(clip_time, mode="once"))
                self._last_rig_time = clip_time
            # In-place cycles: the removed travel velocity still blows the fur back.
            carried = self.rig.root_velocity*speed*self.animation_clock.rate if playing else np.zeros(3)
            self.driver.set_body_state(**self.renderer.controller_state(), elapsed=elapsed,
                inertia_strength=state.get("inertia_strength", 1.), paused=state["paused"], frame_velocity=carried)
        if self.driver is None:
            angles = None
        elif self.simulation_enabled:
            angles = self.driver.advance(elapsed, paused=state["paused"])
        else:
            angles = np.zeros_like(self.driver.angles)
        driver_ms = (time.perf_counter() - tick) * 1000
        mode = state["appearance"]
        if mode == "routes" and self.asset.route is None:
            mode = "appearance"
        if mode == "motion_mask" and not self.simulation_enabled:
            mode = "appearance"
        degree = self.asset.sh_degree if state["sh_degree"] == "Auto" else min(int(state["sh_degree"]), self.asset.sh_degree)
        image = self.renderer.render(angles, camera, mode=mode, sh_degree=degree,
                                     background=BACKGROUNDS[state["background"]],
                                     apply_motion=self.simulation_enabled)
        stats = dict(self.renderer.last_timings)
        count = self.bindings.diagnostics["active_gaussians"] if self.simulation_enabled else 0
        stats.update(simulation_ms=driver_ms + stats["deform_ms"], render_ms=stats["raster_ms"],
                     simulation_time=self.driver.time if self.simulation_enabled else 0.,
                     gaussians=len(self.asset.xyz), simulation_enabled=self.simulation_enabled,
                     motion_profile=state["motion_profile"], appearance=mode,
                     sh_degree=degree, resolution=[camera.width, camera.height],
                     parameters={key: state[key] for key in PARAMETER_RANGES}, paused=state["paused"],
                     bindings=self.bindings.diagnostics if self.bindings is not None else None,
                     status=f"{self.renderer.info['device']}  |  {len(self.asset.xyz):,} Gaussians  |  "
                            + (f"{count:,} fur / {state['motion_profile']}" if self.simulation_enabled else "fur simulation off"))
        stats.update(has_rig=self.rig is not None, animation_enabled=bool(self.rig is not None and state.get("animation_enabled", True)))
        if self.animation_clock is not None:
            stats.update(animation_frame=self.animation_clock.frame, animation_time=self.animation_clock.time,
                         animation_duration=self.animation_clock.duration)
            stats["status"] += f"  |  body frame {self.animation_clock.frame:.1f}"
        elif not self.simulation_enabled:
            stats["status"] += " / static GS"
        if state.get("collision_enabled", False) and self.simulation_enabled:
            contact = stats.get("collision", {})
            stats["status"] += f"  |  SDF {contact.get('collision_ms', 0):.0f} ms"
        self.last_stats = stats
        return image, stats

    def reset(self):
        if self.driver is not None:
            self.driver.reset()

    def snapshot(self):
        # Explicitly select rest buffers even if no static frame was drawn yet.
        angles = self.driver.angles if self.simulation_enabled else np.zeros((self.renderer.controller_count, 3))
        self.renderer.deform(angles, apply_motion=self.simulation_enabled)
        return self.renderer.snapshot()

    def metadata(self):
        return dict(self.asset.metadata(), renderer=self.renderer.info,
                    has_rig=self.rig is not None, rig=self.rig.metadata() if self.rig is not None else None,
                    simulation_enabled=self.simulation_enabled, motion_profile=self.binding_profile,
                    bindings=self.bindings.diagnostics if self.bindings is not None else None,
                    collision=getattr(getattr(self.renderer, "collision", None), "last_stats", None))

    def close(self):
        self.renderer.close()
