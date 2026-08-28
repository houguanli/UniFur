from argparse import Namespace

import pytest

from hairgs_hybrid.inference_config import completed_route_schedule


def test_completed_route_schedule_uses_final_temperature_and_full_activation() -> None:
    schedule = completed_route_schedule(
        Namespace(
            hybrid_temperature=1.5,
            hybrid_temperature_final=0.5,
            hybrid_temperature_anneal_steps=4000,
            hybrid_neutral_activation_steps=1000,
        )
    )
    assert schedule == {
        "temperature": 0.5,
        "temperature_final": 0.5,
        "temperature_anneal_steps": 0,
        "neutral_activation_steps": 0,
    }


def test_completed_route_schedule_supports_legacy_checkpoints() -> None:
    schedule = completed_route_schedule(Namespace(hybrid_temperature=0.75))
    assert schedule["temperature"] == 0.75
    assert schedule["neutral_activation_steps"] == 0


def test_completed_route_schedule_rejects_invalid_temperature() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        completed_route_schedule(
            Namespace(hybrid_temperature=1.5, hybrid_temperature_final=-0.5)
        )
