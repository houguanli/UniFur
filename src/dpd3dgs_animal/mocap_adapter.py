from __future__ import annotations

import os
import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import dump_yaml


@dataclass
class SkeletonPrior:
    joints: np.ndarray
    parents: np.ndarray
    joint_names: list[str]
    source_path: Path
    species: str
    coordinate_frame: str = "mocap_normalized"

    @property
    def num_frames(self) -> int:
        return int(self.joints.shape[0])

    @property
    def num_joints(self) -> int:
        return int(self.joints.shape[1])


class MocapAnythingAdapter:
    def __init__(
        self,
        mocap_root: str | Path = "/home/aoki/MocapAnything_inference_only",
        checkpoint_root: str | Path | None = None,
        zoo_root: str | Path | None = None,
    ) -> None:
        self.root = Path(mocap_root)
        self.checkpoint_root = Path(checkpoint_root) if checkpoint_root else self.root / "checkpoints"
        self.zoo_root = Path(zoo_root) if zoo_root else self.root / "zoo"

    def write_video2pose_config(
        self,
        image_root: str | Path,
        save_dir: str | Path,
        config_path: str | Path,
        ref_seq: str = "Dog#Dog-Galloping/y30",
        ref_idx: int = 0,
        base_config: str | Path | None = None,
    ) -> Path:
        base_config = (
            Path(base_config).resolve()
            if base_config
            else self.root / "configs/inference/inference_video2pose.yaml"
        )
        image_root = Path(image_root).resolve()
        save_dir = Path(save_dir).resolve()
        config_path = Path(config_path).resolve()
        with open(base_config, "r", encoding="utf-8") as f:
            cfg: dict[str, Any] = yaml.safe_load(f)

        cfg["weights"]["video2pose_ckpt_root"] = str(self.checkpoint_root / "video2pose")
        cfg["weights"]["triposg_weights_dir"] = str(self.checkpoint_root / "TripoSG")
        cfg["weights"]["rmbg_weights_dir"] = str(self.checkpoint_root / "RMBG-1.4")
        cfg["data"]["base_dir"] = str(self.zoo_root)
        cfg["data"]["bvh_roots"] = [str(self.zoo_root / "bvh")]
        cfg["data"]["image_roots"] = [str(image_root)]
        cfg["data"]["image_max_depth"] = 1
        cfg["data"]["retarget"]["toggle"] = True
        cfg["data"]["retarget"]["ref_seq"] = ref_seq
        cfg["data"]["retarget"]["ref_idx"] = int(ref_idx)
        cfg["data"]["wild_flag"] = True
        cfg["output"]["save_dir"] = str(save_dir)
        dump_yaml(cfg, config_path)
        return config_path

    def run_video2pose(self, config_path: str | Path, python: str | None = None) -> None:
        python = python or os.environ.get("PYTHON") or sys.executable
        config_path = Path(config_path).resolve()
        env = os.environ.copy()
        extra = [str(self.root), str(self.root / "TripoSG")]
        env["PYTHONPATH"] = os.pathsep.join(extra + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        subprocess.run(
            [python, "-m", "inference.video2pose", "--config", str(config_path)],
            cwd=str(self.root),
            env=env,
            check=True,
        )

    def write_video2pose2rot_config(
        self,
        image_root: str | Path,
        save_dir: str | Path,
        config_path: str | Path,
        ref_seq: str,
        ref_idx: int = 0,
    ) -> Path:
        base_config = self.root / "configs/inference/inference_video2pose2rot_v2_zoo.yaml"
        with open(base_config, "r", encoding="utf-8") as f:
            cfg: dict[str, Any] = yaml.safe_load(f)
        image_root = Path(image_root).resolve()
        save_dir = Path(save_dir).resolve()
        config_path = Path(config_path).resolve()
        cache_dir = self.zoo_root / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        empty_memory = cache_dir / "empty_pose_rotation_memory.pkl"
        if not empty_memory.exists():
            with open(empty_memory, "wb") as f:
                pickle.dump({}, f)

        cfg["weights"]["video2pose_ckpt_root"] = str(
            self.checkpoint_root / "video2pose2rot"
        )
        cfg["weights"]["triposg_weights_dir"] = str(self.checkpoint_root / "TripoSG")
        cfg["weights"]["rmbg_weights_dir"] = str(self.checkpoint_root / "RMBG-1.4")
        cfg["data"]["base_dir"] = str(self.zoo_root)
        cfg["data"]["memory_pkl_path"] = str(empty_memory)
        cfg["data"]["bvh_roots"] = [str(self.zoo_root / "bvh")]
        cfg["data"]["image_roots"] = [str(image_root)]
        cfg["data"]["image_max_depth"] = 1
        cfg["data"]["split_json"] = None
        cfg["data"]["retarget"]["toggle"] = True
        cfg["data"]["retarget"]["ref_seq"] = ref_seq
        cfg["data"]["retarget"]["ref_idx"] = int(ref_idx)
        cfg["data"]["wild_flag"] = True
        cfg["output"]["save_dir"] = str(save_dir)
        cfg["output"]["export_gt_mesh"] = False
        cfg["output"]["export_gt_video"] = False
        dump_yaml(cfg, config_path)
        return config_path

    def run_video2pose2rot(
        self,
        config_path: str | Path,
        python: str | None = None,
    ) -> None:
        python = python or os.environ.get("PYTHON") or sys.executable
        config_path = Path(config_path).resolve()
        env = os.environ.copy()
        extra = [str(self.root), str(self.root / "TripoSG")]
        env["PYTHONPATH"] = os.pathsep.join(
            extra + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        subprocess.run(
            [python, "-m", "inference.video2pose2rot", "--config", str(config_path)],
            cwd=str(self.root),
            env=env,
            check=True,
        )

    def find_pose_rotation_predictions(
        self,
        save_dir: str | Path,
    ) -> tuple[Path, Path]:
        root = Path(save_dir)
        pose = sorted(root.rglob("*_pose_pred.npy"))
        rotation = sorted(root.rglob("*_rot6d_pred.npy"))
        if not pose or not rotation:
            raise FileNotFoundError(
                f"No video2pose2rot pose/rotation predictions found under {root}"
            )
        return pose[-1], rotation[-1]

    def find_prediction(self, save_dir: str | Path, exp: str, seq_name: str) -> Path:
        root = Path(save_dir) / exp / seq_name
        matches = sorted(root.glob("*_pred.npy"))
        if not matches:
            raise FileNotFoundError(f"No MocapAnything prediction found under {root}")
        return matches[-1]

    def load_skeleton_prior(self, prediction_path: str | Path, ref_seq: str) -> SkeletonPrior:
        prediction_path = Path(prediction_path)
        joints = np.load(prediction_path).astype(np.float32)
        species = ref_seq.split("#", 1)[0]
        info = self._species_info(species)
        joint_names = [str(x) for x in info.get("joints_name", [])]
        n = min(joints.shape[1], len(joint_names) if joint_names else joints.shape[1])
        joints = joints[:, :n, :]
        parents = self._parents_from_reference_bvh(ref_seq, joint_names[:n])
        if parents is None:
            parents = parents_from_joint_relation(info.get("joint_relation"), n)
        return SkeletonPrior(joints, parents, joint_names[:n], prediction_path, species)

    def _species_info(self, species: str) -> dict[str, Any]:
        path = self.zoo_root / "species_info_dict.npy"
        data = np.load(path, allow_pickle=True).item()
        if species not in data:
            raise KeyError(f"Species {species!r} not found in {path}")
        return data[species]

    def _parents_from_reference_bvh(
        self,
        ref_seq: str,
        expected_names: list[str],
    ) -> np.ndarray | None:
        path = self.zoo_root / "bvh" / f"{ref_seq}.bvh"
        if not path.exists():
            return None
        names, parents = parse_bvh_hierarchy(path)
        n = len(expected_names)
        if len(names) < n or names[:n] != expected_names:
            return None
        return np.asarray(parents[:n], dtype=np.int64)


def parse_bvh_hierarchy(path: str | Path) -> tuple[list[str], list[int]]:
    """Read joint names and parents from the BVH hierarchy section."""

    names: list[str] = []
    parents: list[int] = []
    stack: list[int | None] = []
    pending: int | None | str = None
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line == "MOTION":
                break
            if line.startswith("ROOT ") or line.startswith("JOINT "):
                name = line.split(maxsplit=1)[1]
                parent = next((x for x in reversed(stack) if x is not None), -1)
                names.append(name)
                parents.append(int(parent))
                pending = len(names) - 1
            elif line == "End Site":
                pending = "end"
            elif line == "{":
                stack.append(pending if isinstance(pending, int) else None)
                pending = None
            elif line == "}":
                if stack:
                    stack.pop()
    return names, parents


def parents_from_joint_relation(joint_relation: np.ndarray | None, n: int) -> np.ndarray:
    parents = np.full((n,), -1, dtype=np.int64)
    if joint_relation is None:
        parents[1:] = 0
        return parents
    relation = np.asarray(joint_relation)[:n, :n]
    positive = relation[relation > 0]
    if positive.size == 0:
        parents[1:] = 0
        return parents
    # MocapAnything stores all-pairs graph distance here, not a binary
    # adjacency matrix. Only the minimum positive value represents an edge.
    rel = np.isclose(relation, float(positive.min()))
    visited = np.zeros((n,), dtype=bool)
    queue = [0]
    visited[0] = True
    while queue:
        cur = queue.pop(0)
        neighbors = np.flatnonzero(rel[cur] | rel[:, cur])
        for nb in neighbors:
            nb = int(nb)
            if nb == cur or visited[nb]:
                continue
            parents[nb] = cur
            visited[nb] = True
            queue.append(nb)
    for idx in range(1, n):
        if parents[idx] < 0:
            parents[idx] = 0
    return parents
