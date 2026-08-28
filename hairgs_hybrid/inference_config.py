"""Pure helpers for loading a completed soft-routing checkpoint."""

from __future__ import annotations


def completed_route_schedule(args) -> dict[str, float | int]:
    """Resolve training-time routing fields to their inference-time values.

    A renderer does not execute optimizer iterations, so replaying the saved
    warm-up schedule would leave a neutral model at activation zero.  A saved
    checkpoint is evaluated at the completed schedule instead.
    """

    start = getattr(args, "hybrid_temperature", None) or 0.75
    final = getattr(args, "hybrid_temperature_final", None) or start
    if final <= 0:
        raise ValueError("hybrid_temperature_final must be positive")
    return {
        "temperature": float(final),
        "temperature_final": float(final),
        "temperature_anneal_steps": 0,
        "neutral_activation_steps": 0,
    }
