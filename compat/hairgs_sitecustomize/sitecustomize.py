"""Runtime-only compatibility aliases for the pinned, unmodified HairGS code."""

import numpy as np


if "bool" not in np.__dict__:
    np.bool = np.bool_  # type: ignore[attr-defined]
