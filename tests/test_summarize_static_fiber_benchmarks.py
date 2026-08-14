import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_static_fiber_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("summarize_static_fiber_benchmarks", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _evaluation(protocol="same", size=(512, 512), views=(1, 3)):
    return {
        "protocol": protocol,
        "render_size": list(size),
        "image_count": len(views),
        "ground_truth_dir": "/tmp/gt",
        "per_frame": [
            {"image": f"{view:04d}.png", "view_index": view} for view in views
        ],
    }


def test_validate_group_accepts_identical_heldout_protocol():
    MODULE._validate_group(
        "hair",
        [
            {"method": "a", "evaluation": _evaluation()},
            {"method": "b", "evaluation": _evaluation()},
        ],
    )


def test_validate_group_rejects_different_heldout_views():
    with pytest.raises(ValueError, match="mixes incompatible"):
        MODULE._validate_group(
            "hair",
            [
                {"method": "a", "evaluation": _evaluation()},
                {"method": "b", "evaluation": _evaluation(views=(1, 4))},
            ],
        )
