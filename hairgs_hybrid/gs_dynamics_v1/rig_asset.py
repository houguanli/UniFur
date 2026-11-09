"""Validated surface correspondence to baked Blender rig animation.

The cache preserves the trained pose using barycentric surface points plus
authored offsets. Source animation is evaluated by Blender once; the viewer
uses these compact surface maps and the shared vertex clips without Blender.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "unifur-blender-rig-binding-v1"
# v2: motion clips of another action of the same rig; the reference stays the
# trained pose, so clips need not start there. Optional loop / root velocity
# (in-place cycles) and per-Gaussian fur semantics from the rig's skinning.
SCHEMA_V2 = "unifur-blender-rig-binding-v2"
SEMANTICS = {"mane": "gs_semantic_mane", "tail": "gs_semantic_tail", "face": "gs_semantic_face"}


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


class RigBinding:
    REQUIRED = {"driver_vertices_rest", "driver_faces", "gs_face", "gs_bary", "gs_rest_offset",
                "body_vertices_rest", "body_faces", "body_driver_face", "body_driver_bary",
                "body_driver_offset", "manifest_json"}

    @classmethod
    def load(cls, path, asset):
        path = Path(path).expanduser().resolve()
        complete = getattr(asset, "complete_asset", asset)
        if complete is None:
            raise ValueError("Rig animation requires a complete UniFur asset")
        with np.load(path, allow_pickle=False) as archive:
            missing = cls.REQUIRED - set(archive.files)
            if missing:
                raise ValueError(f"Missing rig binding arrays: {sorted(missing)}")
            values = {key: np.array(archive[key], copy=True) for key in cls.REQUIRED}
            semantic = {name: np.array(archive[key], np.float32, copy=True)
                        for name, key in SEMANTICS.items() if key in archive.files}
        try:
            manifest = json.loads(str(values.pop("manifest_json").item()))
        except (ValueError, TypeError) as error:
            raise ValueError("Invalid rig binding manifest") from error
        version = {SCHEMA: 1, SCHEMA_V2: 2}.get(manifest.get("schema"))
        if version is None:
            raise ValueError("Unsupported rig binding schema")
        expected = manifest.get("target_asset_sha256")
        if not expected or getattr(asset, "source_sha256", None) != expected:
            raise ValueError("Rig binding target asset SHA-256 does not match the opened GS file")
        for key, value in values.items():
            if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError(f"Rig array {key} must contain finite numeric values")
        if not np.array_equal(values["body_vertices_rest"], complete.body_vertices):
            raise ValueError("Rig body rest vertices differ from the trained asset")
        if not np.array_equal(values["body_faces"], complete.body_faces):
            raise ValueError("Rig body topology differs from the trained asset")

        clips = manifest.get("clips")
        if not isinstance(clips, list) or not clips:
            raise ValueError("Rig manifest has no vertex clips")
        positions, reference, faces, ranges = [], [], [], []
        source_frames = None
        fps = None
        vertex_offset = 0
        for entry in clips:
            relative = Path(entry["file"])
            clip_path = (path.parent / relative).resolve()
            if clip_path.suffix.lower() != ".npz":
                raise ValueError("Rig vertex clips must be NPZ files")
            if not entry.get("sha256") or file_sha256(clip_path) != entry["sha256"]:
                raise ValueError(f"Rig clip hash mismatch: {clip_path.name}")
            with np.load(clip_path, allow_pickle=False) as clip:
                data = np.array(clip["vertex_positions_world"], dtype=np.float32, order="C", copy=True)
                triangle = np.array(clip["triangles"], copy=True)
                frames = np.array(clip["frame_numbers"], copy=True)
                rate = float(clip["fps"])
                rest_frame = int(clip["reference_frame"])
                mesh_name = str(clip["mesh_name"].item())
                source_hash = str(clip["source_sha256"].item())
            if (data.ndim != 3 or data.shape[2] != 3 or data.shape[0] < 2 or data.shape[1] < 1
                    or not np.isfinite(data).all()):
                raise ValueError(f"Invalid vertex animation array in {clip_path.name}")
            if (frames.shape != (len(data),) or frames.dtype.kind not in "iu"
                    or np.any(np.diff(frames) <= 0) or not np.isfinite(rate) or rate <= 0):
                raise ValueError(f"Invalid frame table or FPS in {clip_path.name}")
            if rest_frame != int(frames[0]) or (version == 1 and rest_frame != int(manifest.get("reference_frame", -1))):
                raise ValueError("Rig reference must be the first clip frame")
            if source_hash != manifest.get("source_blend_sha256") or mesh_name != entry["mesh_name"]:
                raise ValueError("Rig clip source identity does not match its manifest")
            cls._validate_faces(triangle, data.shape[1], f"clip {mesh_name}")
            if source_frames is None:
                source_frames, fps = frames, rate
            elif not np.array_equal(frames, source_frames) or rate != fps:
                raise ValueError("All rig clips must share identical frame numbers and FPS")
            positions.append(data)
            reference.append(data[0])
            faces.append(triangle.astype(np.int64) + vertex_offset)
            ranges.append(dict(mesh_name=mesh_name, vertex_start=vertex_offset, vertices=data.shape[1],
                               triangles=len(triangle), clip=str(clip_path)))
            vertex_offset += data.shape[1]
        driver_rest = np.concatenate(reference, axis=0)
        driver_faces = np.concatenate(faces, axis=0)
        if version == 1:
            if not np.array_equal(values["driver_vertices_rest"], driver_rest):
                raise ValueError("Rig rest vertices do not match the first source clip frame")
            if not np.array_equal(values["driver_faces"], driver_faces):
                raise ValueError("Rig source face order does not match the clip sequence")
        else:
            # Same vertex order as the bound reference; Blender may triangulate
            # posed quads differently, so the reference triangulation is used.
            expected = [(entry["mesh_name"], int(entry["vertices"])) for entry in manifest.get("driver_mesh_ranges", [])]
            if expected != [(entry["mesh_name"], entry["vertices"]) for entry in ranges]:
                raise ValueError("Rig motion clips do not match the bound reference meshes")
            driver_rest = values["driver_vertices_rest"]
            driver_faces = values["driver_faces"]
        n, b = len(complete.xyz), len(complete.body_vertices)
        cls._validate_map(values, "gs", n, len(driver_faces))
        cls._validate_map(values, "body_driver", b, len(driver_faces))
        cls._validate_faces(values["driver_faces"], len(driver_rest), "driver")
        cls._validate_faces(values["body_faces"], b, "body")
        reconstructed_gs = ((driver_rest[driver_faces[values["gs_face"]]] * values["gs_bary"][:, :, None]).sum(1)
                            + values["gs_rest_offset"])
        reconstructed_body = ((driver_rest[driver_faces[values["body_driver_face"]]]
                              * values["body_driver_bary"][:, :, None]).sum(1) + values["body_driver_offset"])
        scale = max(float(np.linalg.norm(np.ptp(complete.body_vertices, axis=0))), 1e-6)
        rest_error = float(np.abs(reconstructed_gs-complete.xyz).max())
        body_error = float(np.abs(reconstructed_body-complete.body_vertices).max())
        if rest_error > 1e-6*scale or body_error > 1e-6*scale:
            raise ValueError("Rig surface maps do not reproduce the trained rest pose")
        obj = cls()
        for key, value in values.items():
            value.setflags(write=False)
            setattr(obj, key, value)
        obj.frames_vertices = np.concatenate(positions, axis=1)
        obj.frames_vertices.setflags(write=False)
        obj.frame_numbers = source_frames.astype(np.int32)
        obj.frame_numbers.setflags(write=False)
        obj.fps = float(fps)
        obj.reference_frame = int(source_frames[0])
        obj.duration_seconds = float((source_frames[-1]-source_frames[0])/fps)
        obj._times = (source_frames.astype(np.float64)-source_frames[0])/fps
        obj.source_metadata, obj.driver_mesh_ranges = manifest, ranges
        obj.loop = bool(manifest.get("loop", False))
        root_velocity = np.asarray(manifest.get("root_velocity", (0., 0., 0.)), np.float64)
        if root_velocity.shape != (3,) or not np.isfinite(root_velocity).all():
            raise ValueError("Rig root velocity must be a finite 3-vector")
        obj.root_velocity = root_velocity
        for name, value in semantic.items():
            if value.shape != (n,) or not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
                raise ValueError(f"Rig {name} semantics must be [N] values in [0,1]")
            value.setflags(write=False)
        obj.semantics = semantic if len(semantic) == len(SEMANTICS) else None
        obj.source_path, obj.source_sha256 = path, file_sha256(path)
        obj.rest_max_error, obj.body_rest_max_error = rest_error, body_error
        return obj

    @staticmethod
    def _validate_faces(faces, vertex_count, label):
        if (faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu" or not len(faces)
                or faces.min() < 0 or faces.max() >= vertex_count):
            raise ValueError(f"Invalid {label} triangle indices")

    @staticmethod
    def _validate_map(values, prefix, count, face_count):
        face = values[f"{prefix}_face"]
        bary = values[f"{prefix}_bary"]
        offset_name = "gs_rest_offset" if prefix == "gs" else "body_driver_offset"
        offset = values[offset_name]
        if (face.shape != (count,) or face.dtype.kind not in "iu" or face.min() < 0 or face.max() >= face_count
                or bary.shape != (count, 3) or offset.shape != (count, 3)):
            raise ValueError(f"Invalid {prefix} surface correspondence shape or index")
        if np.any(bary < -1e-8) or np.any(bary > 1+1e-8) or np.max(np.abs(bary.sum(1)-1)) > 1e-7:
            raise ValueError(f"Invalid {prefix} barycentric coordinates")

    def sample(self, time_seconds, mode="once"):
        """Owned world-space source vertices, linearly interpolated in time.

        ``once`` clamps at the clip ends. Plain ``pingpong`` reverses time;
        callers wanting eased reversal can map their clock and use ``once``.
        ``loop`` is supported but the authored clip has an identified seam.
        """
        time_seconds = float(time_seconds)
        if not np.isfinite(time_seconds) or time_seconds < 0:
            raise ValueError("Rig time must be finite and nonnegative")
        duration = self.duration_seconds
        if mode == "once":
            time_seconds = min(time_seconds, duration)
        elif mode == "loop":
            time_seconds %= duration
        elif mode == "pingpong":
            time_seconds %= 2*duration
            time_seconds = min(time_seconds, 2*duration-time_seconds)
        else:
            raise ValueError("Rig playback mode must be once, loop or pingpong")
        upper = int(np.searchsorted(self._times, time_seconds, side="right"))
        if upper == 0:
            return self.frames_vertices[0].copy()
        if upper >= len(self._times):
            return self.frames_vertices[-1].copy()
        lower = upper-1
        weight = (time_seconds-self._times[lower])/(self._times[upper]-self._times[lower])
        if abs(weight) < 1e-12:
            return self.frames_vertices[lower].copy()
        return ((1-weight)*self.frames_vertices[lower] + weight*self.frames_vertices[upper]).astype(np.float32)

    def sample_frame(self, frame_number):
        return self.sample(max(0., (float(frame_number)-self.reference_frame)/self.fps), mode="once")

    def metadata(self):
        return dict(schema=self.source_metadata["schema"], binding=str(self.source_path), sha256=self.source_sha256,
                    source_blend=self.source_metadata["source_blend"],
                    source_blend_sha256=self.source_metadata["source_blend_sha256"],
                    action=self.source_metadata.get("source_action"),
                    loop=self.loop, root_velocity=self.root_velocity.tolist(), has_semantics=self.semantics is not None,
                    target_asset_sha256=self.source_metadata["target_asset_sha256"],
                    frames=len(self.frame_numbers), fps=self.fps, duration_seconds=self.duration_seconds,
                    reference_frame=self.reference_frame, driver_vertices=len(self.driver_vertices_rest),
                    driver_triangles=len(self.driver_faces), mesh_ranges=self.driver_mesh_ranges,
                    rest_max_error=self.rest_max_error, body_rest_max_error=self.body_rest_max_error,
                    playback=self.source_metadata.get("playback", {}))
