"""Compact serialization for fixed shell/fin/strand ownership.

The soft checkpoint stores three FP32 route logits, two FP32 shape values, and
one FP32 roll value for every segment.  A hard-route checkpoint only needs the
2-bit ownership plus the parameters that are active for that ownership:

* shell: one shape value and one roll value;
* fin: one shape value and one roll value;
* strand/base: no additional shape or roll value.

The compact arrays are ordered by the decoded route indices, so no redundant
index arrays are serialized.  FP16 is sufficient for the bounded shape and
roll residuals and the saved model is evaluated after the same decode path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


SCHEMA = "hairgs-hybrid-hard-state-v2"
_VALID_ROUTES = np.asarray([-1, 0, 1, 2], dtype=np.int8)


def pack_route_ids(route_ids: np.ndarray) -> np.ndarray:
    """Pack four route ids into one byte using ``route_id + 1`` codes."""

    route_ids = np.asarray(route_ids, dtype=np.int8)
    if route_ids.ndim != 1 or not np.all(np.isin(route_ids, _VALID_ROUTES)):
        raise ValueError("route_ids must be a 1-D array containing only -1/0/1/2")
    codes = (route_ids.astype(np.uint8) + 1).astype(np.uint8)
    padding = (-len(codes)) % 4
    if padding:
        codes = np.pad(codes, (0, padding), constant_values=0)
    codes = codes.reshape(-1, 4)
    return (
        codes[:, 0]
        | (codes[:, 1] << 2)
        | (codes[:, 2] << 4)
        | (codes[:, 3] << 6)
    ).astype(np.uint8)


def unpack_route_ids(packed: np.ndarray, count: int) -> np.ndarray:
    """Decode a packed route array and validate the requested segment count."""

    packed = np.asarray(packed, dtype=np.uint8).reshape(-1)
    if count < 0 or packed.size != (count + 3) // 4:
        raise ValueError("Packed route byte count does not match segment_count")
    codes = np.empty(packed.size * 4, dtype=np.uint8)
    for offset in range(4):
        codes[offset::4] = (packed >> (2 * offset)) & 0x03
    return codes[:count].astype(np.int8) - 1


def save_compact_hard_state(
    path: Path,
    route_ids: np.ndarray,
    shell_strength_raw: np.ndarray,
    fin_strength_raw: np.ndarray,
    non_strand_roll_residual: np.ndarray,
) -> None:
    """Write a self-contained hard-route state next to the native HairGS PLY."""

    path = Path(path)
    route_ids = np.asarray(route_ids, dtype=np.int8).reshape(-1)
    shell = np.asarray(shell_strength_raw, dtype=np.float32).reshape(-1)
    fin = np.asarray(fin_strength_raw, dtype=np.float32).reshape(-1)
    roll = np.asarray(non_strand_roll_residual, dtype=np.float32).reshape(-1)
    expected = (
        int(np.count_nonzero(route_ids == 0)),
        int(np.count_nonzero(route_ids == 1)),
        int(np.count_nonzero(np.logical_or(route_ids == 0, route_ids == 1))),
    )
    actual = (shell.size, fin.size, roll.size)
    if actual != expected:
        raise ValueError(f"Hard-state parameter counts {actual} != expected {expected}")
    if not all(np.all(np.isfinite(value)) for value in (shell, fin, roll)):
        raise ValueError("Hard-state parameters contain non-finite values")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema=np.asarray(SCHEMA),
        segment_count=np.asarray(route_ids.size, dtype=np.uint32),
        route_code_packed=pack_route_ids(route_ids),
        shell_strength_raw=shell.astype(np.float16),
        fin_strength_raw=fin.astype(np.float16),
        non_strand_roll_residual=roll.astype(np.float16),
    )


def load_compact_hard_state(path: Path) -> dict[str, np.ndarray]:
    """Load and validate a compact state, returning FP32 trainable arrays."""

    with np.load(Path(path), allow_pickle=False) as state:
        schema = str(np.asarray(state["schema"]).item())
        if schema != SCHEMA:
            raise ValueError(f"Unsupported compact hard-state schema: {schema}")
        count = int(np.asarray(state["segment_count"]).item())
        route_ids = unpack_route_ids(state["route_code_packed"], count)
        shell = np.asarray(state["shell_strength_raw"], dtype=np.float32).reshape(-1)
        fin = np.asarray(state["fin_strength_raw"], dtype=np.float32).reshape(-1)
        roll = np.asarray(
            state["non_strand_roll_residual"], dtype=np.float32
        ).reshape(-1)
    expected = (
        int(np.count_nonzero(route_ids == 0)),
        int(np.count_nonzero(route_ids == 1)),
        int(np.count_nonzero(np.logical_or(route_ids == 0, route_ids == 1))),
    )
    if (shell.size, fin.size, roll.size) != expected:
        raise ValueError("Compact hard-state route/parameter counts are inconsistent")
    if not all(np.all(np.isfinite(value)) for value in (shell, fin, roll)):
        raise ValueError("Compact hard-state parameters contain non-finite values")
    return {
        "route_id": route_ids,
        "shell_strength_raw": shell,
        "fin_strength_raw": fin,
        "non_strand_roll_residual": roll,
    }
