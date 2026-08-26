from pathlib import Path

import numpy as np
import pytest

from hairgs_hybrid.compact_state import (
    load_compact_hard_state,
    pack_route_ids,
    save_compact_hard_state,
    unpack_route_ids,
)


@pytest.mark.parametrize("count", [0, 1, 3, 4, 5, 31, 32, 33])
def test_route_ids_round_trip_four_per_byte(count: int) -> None:
    route_ids = np.resize(np.asarray([-1, 0, 1, 2], dtype=np.int8), count)
    packed = pack_route_ids(route_ids)
    assert packed.nbytes == (count + 3) // 4
    np.testing.assert_array_equal(unpack_route_ids(packed, count), route_ids)


def test_compact_hard_state_round_trip_and_parameter_elision(tmp_path: Path) -> None:
    route_ids = np.asarray([-1, 0, 2, 1, 0, 2, 1, -1, 0], dtype=np.int8)
    shell = np.asarray([-1.1, 0.2, 1.7], dtype=np.float32)
    fin = np.asarray([0.4, -0.8], dtype=np.float32)
    roll = np.asarray([0.1, -0.2, 0.3, -0.4, 0.5], dtype=np.float32)
    path = tmp_path / "hybrid_state.npz"

    save_compact_hard_state(path, route_ids, shell, fin, roll)
    decoded = load_compact_hard_state(path)

    np.testing.assert_array_equal(decoded["route_id"], route_ids)
    np.testing.assert_allclose(decoded["shell_strength_raw"], shell, atol=5e-4)
    np.testing.assert_allclose(decoded["fin_strength_raw"], fin, atol=5e-4)
    np.testing.assert_allclose(decoded["non_strand_roll_residual"], roll, atol=5e-4)
    with np.load(path, allow_pickle=False) as state:
        assert "route_logits" not in state.files
        assert "shape_strength_raw" not in state.files
        assert state["route_code_packed"].nbytes == 3


def test_compact_hard_state_rejects_route_parameter_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="parameter counts"):
        save_compact_hard_state(
            tmp_path / "bad.npz",
            np.asarray([0, 1, 2], dtype=np.int8),
            np.asarray([], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([0.0, 0.0], dtype=np.float32),
        )
