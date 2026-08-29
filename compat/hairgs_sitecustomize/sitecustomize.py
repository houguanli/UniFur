"""Runtime-only compatibility aliases for the pinned, unmodified HairGS code."""

try:
    import numpy as np
except ModuleNotFoundError:
    # This module is also loaded by the brand-new interpreter used to install
    # NumPy itself. Compatibility aliases only matter after NumPy exists.
    pass
else:
    if "bool" not in np.__dict__:
        np.bool = np.bool_  # type: ignore[attr-defined]
