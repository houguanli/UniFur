"""Common static 3DGS data with optional validated UniFur dynamics metadata.

PLY follows the Graphdeco field convention: log-scales, opacity logits, wxyz
quaternions and channel-major SH storage. NPZ uses the project's canonical
xyz/scale/rotation/dc/rest/opacity arrays. No point-cloud parameters are guessed.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .asset import CompleteAsset


class GaussianAsset:
    REQUIRED = {"xyz", "scale", "rotation", "dc", "opacity"}

    @classmethod
    def load(cls, path):
        path = Path(path).expanduser().resolve()
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            kind = "UniFur complete NPZ" if CompleteAsset.REQUIRED <= arrays.keys() else "Gaussian NPZ"
        elif path.suffix.lower() == ".ply":
            arrays = cls._read_ply(path)
            kind = "3DGS PLY"
        else:
            raise ValueError("Open a standard 3DGS .ply or Gaussian .npz file")
        obj = cls.from_arrays(arrays, source_format=kind)
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        obj.source_path, obj.source_sha256 = path, digest.hexdigest()
        if obj.complete_asset is not None:
            obj.complete_asset.source_path = path
            obj.complete_asset.source_sha256 = obj.source_sha256
        return obj

    @classmethod
    def from_arrays(cls, arrays, *, source_format="Gaussian NPZ"):
        missing = cls.REQUIRED - arrays.keys()
        if missing:
            raise ValueError(f"Missing Gaussian arrays: {', '.join(sorted(missing))}")
        obj = cls()
        n = len(arrays["xyz"])
        values = {key: np.array(arrays[key], dtype=np.float32, order="C", copy=True) for key in cls.REQUIRED}
        values["rest"] = np.array(arrays.get("rest", np.empty((n, 0, 3))), dtype=np.float32, order="C", copy=True)
        # Flat DC/opacity arrays are common in portable NPZ exports.
        if values["dc"].shape == (n, 3):
            values["dc"] = values["dc"][:, None, :]
        if values["opacity"].shape == (n,):
            values["opacity"] = values["opacity"][:, None]
        shapes = {"xyz": (n, 3), "scale": (n, 3), "rotation": (n, 4), "dc": (n, 1, 3), "opacity": (n, 1)}
        if n < 1 or any(values[key].shape != shape for key, shape in shapes.items()):
            raise ValueError("Gaussian shape mismatch: need xyz/scale [N,3], rotation [N,4], dc [N,1,3], opacity [N,1]")
        rest = values["rest"]
        if rest.ndim != 3 or rest.shape[0] != n or rest.shape[2] != 3:
            raise ValueError("SH rest must have shape [N,K,3]")
        degree = math.isqrt(rest.shape[1] + 1) - 1
        if (degree + 1)**2 != rest.shape[1] + 1 or not 0 <= degree <= 3:
            raise ValueError("Supported SH degrees are 0, 1, 2, 3 (rest K = 0, 3, 8, 15)")
        if not all(np.isfinite(value).all() for value in values.values()):
            raise ValueError("Gaussian arrays must be finite float32 values")
        if np.any(np.linalg.norm(values["rotation"].astype(np.float64), axis=1) < 1e-10):
            raise ValueError("Gaussian quaternion cannot be zero")
        with np.errstate(over="ignore", under="ignore"):
            scales = np.exp(values["scale"])
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("Log-scales must exponentiate to finite positive float32 scales")
        obj.route = None
        if "route" in arrays:
            route = np.asarray(arrays["route"])
            if route.shape == (n,) and route.dtype.kind in "iu" and np.all((route >= 0) & (route <= 2)):
                obj.route = np.array(route, dtype=np.int64, copy=True)
                obj.route.setflags(write=False)
        obj.complete_asset = None
        missing_dynamics = CompleteAsset.REQUIRED - arrays.keys()
        if missing_dynamics:
            obj.simulation_reason = "Static asset: missing UniFur body mesh / routing / guide graph."
        else:
            try:
                obj.complete_asset = CompleteAsset.from_arrays(arrays)
                obj.simulation_reason = "UniFur guide dynamics available (asset-coordinate mane preset)."
            except (ValueError, TypeError, IndexError) as error:
                obj.simulation_reason = f"Static asset: invalid UniFur dynamics data ({error})."
        for value in values.values():
            value.setflags(write=False)
        obj.arrays = MappingProxyType(values)
        obj.source_format, obj.sh_degree = source_format, degree
        obj.source_path = obj.source_sha256 = None
        return obj

    def __getattr__(self, name):
        if name in self.__dict__.get("arrays", {}):
            return self.arrays[name]
        raise AttributeError(name)

    @property
    def can_simulate(self):
        return self.complete_asset is not None

    def metadata(self):
        return dict(asset=str(self.source_path) if self.source_path else None,
                    sha256=self.source_sha256, source_format=self.source_format,
                    gaussians=len(self.xyz), sh_degree=self.sh_degree,
                    can_simulate=self.can_simulate, has_routes=self.route is not None,
                    simulation_reason=self.simulation_reason)

    @staticmethod
    def _read_ply(path):
        from plyfile import PlyData
        ply = PlyData.read(str(path), mmap=False)
        if "vertex" not in ply:
            raise ValueError("3DGS PLY needs a vertex element")
        vertex = ply["vertex"]
        names = {prop.name for prop in vertex.properties}
        required = {"x", "y", "z", "opacity"} | {f"f_dc_{i}" for i in range(3)} | {f"scale_{i}" for i in range(3)} | {f"rot_{i}" for i in range(4)}
        missing = required - names
        if missing:
            raise ValueError("Not a standard 3DGS PLY; missing " + ", ".join(sorted(missing)))
        extra = [name for name in names if name.startswith("f_rest_")]
        if set(extra) != {f"f_rest_{i}" for i in range(len(extra))} or len(extra) % 3:
            raise ValueError("PLY f_rest_* fields must be contiguous and channel-major")
        def columns(keys):
            return np.stack([np.asarray(vertex[key]) for key in keys], axis=1)
        n = vertex.count
        rest = (columns([f"f_rest_{i}" for i in range(len(extra))]).reshape(n, 3, -1).transpose(0, 2, 1)
                if extra else np.empty((n, 0, 3), np.float32))
        return dict(xyz=columns(("x", "y", "z")), opacity=np.asarray(vertex["opacity"])[:, None],
                    scale=columns([f"scale_{i}" for i in range(3)]),
                    rotation=columns([f"rot_{i}" for i in range(4)]),
                    dc=columns([f"f_dc_{i}" for i in range(3)])[:, None], rest=rest)
