"""CPU-only release pins and dependency checks; never load observation data."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from hairgs_hybrid.rgb_only_init_v1.core import sha
from hairgs_hybrid.scaffold_io import load_scaffold, subset, write_scaffold
from hairgs_hybrid.teacher_free_shell_strand_v1.run import runtime_manifest


def test_completed_run_source_pins():
    root = Path(__file__).resolve().parents[1]
    config = root / "hairgs_hybrid/teacher_free_shell_strand_v1/run_formal8k_v1.json"
    frozen = runtime_manifest(
        config, "f7b030c4afb1e53559f4fe0aea4dc9bd5dd9c0c9616c6455b3bebbc43ab608d1"
    )
    source = root / "hairgs_hybrid/rgb_only_init_v1/source_tail_v3.json"
    assert sha(source) == frozen["source_lock_sha256"]
    assert frozen["status"] == "approved_for_fresh_formal_training"


@pytest.mark.parametrize("module", [
    "hairgs_hybrid.prepare_a050_hairgs_rgb_only",
    "hairgs_hybrid.export_teacher_free_shell_strand",
    "hairgs_hybrid.compress_hairgs_teacher",
    "hairgs_hybrid.convert_gaussian_to_hairgs",
    "hairgs_hybrid.benchmark_render",
    "hairgs_hybrid.train_native",
    "scripts.export_hairgs_external_renders",
])
def test_release_entrypoint_imports_without_observations(module):
    importlib.import_module(module)


def test_scaffold_subset_preserves_shared_endpoints(tmp_path):
    points = np.asarray([[0., 0., 0.], [1., 0., 0.], [2., .5, 0.]])
    pairs = np.asarray([[0, 1], [1, 2]])
    vector = points[pairs[:, 0]] - points[pairs[:, 1]]
    length = np.linalg.norm(vector, axis=1)
    fields = np.zeros((2, 6))
    fields[:, 3:5] = 1.
    fields[:, 5] = np.log(.02)
    data = dict(
        centers=.5 * (points[pairs[:, 0]] + points[pairs[:, 1]]),
        direction=vector / length[:, None], half_length=.5 * length,
        width=np.full(2, .02), fields=fields,
        field_names=("f_dc_0", "f_dc_1", "f_dc_2", "opacity", "mask", "width"),
        roots=np.zeros((2, 3)), endpoints=points, endpoint_pairs=pairs,
        root_endpoint_indices=np.asarray([0]), ref_roots=np.zeros((1, 3)),
    )
    path = tmp_path / "curve.ply"
    write_scaffold(path, data)
    compact = subset(load_scaffold(path), np.asarray([0, 1]))
    assert compact["endpoints"].shape == (3, 3)
    assert compact["endpoint_pairs"][0, 1] == compact["endpoint_pairs"][1, 0]
    assert len(compact["root_endpoint_indices"]) == 1
