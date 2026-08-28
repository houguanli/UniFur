from __future__ import annotations

import numpy as np

from hairgs_hybrid.build_topology_candidate_bank import balanced_spatial_partition
from hairgs_hybrid.harden_topology_checkpoint import select_exact_budget


def test_balanced_spatial_partition_is_complete_and_balanced() -> None:
    rng = np.random.default_rng(4)
    points = rng.normal(size=(101, 3))
    groups = balanced_spatial_partition(points, np.arange(101), 10)
    assert len(groups) == 10
    assert sorted(np.concatenate(groups).tolist()) == list(range(101))
    assert max(map(len, groups)) - min(map(len, groups)) <= 1


def test_exact_budget_assignment_uses_real_route_counts() -> None:
    rng = np.random.default_rng(9)
    logits = rng.normal(size=(80, 3))
    costs = np.tile(np.asarray([[1, 2, 4]]), (80, 1))
    for target in (80, 121, 160, 219, 320):
        selected = select_exact_budget(logits, costs, target)
        assert int(costs[np.arange(80), selected].sum()) == target
        assert selected.shape == (80,)


def test_route_shuffle_preserves_active_budget() -> None:
    rng = np.random.default_rng(3)
    selected = np.asarray([0] * 19 + [1] * 13 + [2] * 8, dtype=np.int8)
    shuffled = rng.permutation(selected)
    costs = np.asarray([1, 2, 4])
    assert int(costs[selected].sum()) == int(costs[shuffled].sum())
    assert np.array_equal(np.bincount(selected, minlength=3), np.bincount(shuffled, minlength=3))
