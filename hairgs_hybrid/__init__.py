"""HairGS-compatible adaptive shell/fin/strand experimental model."""

__all__ = ["HybridHairGaussianModel", "HybridModelConfig"]


def __getattr__(name: str):
    """Avoid importing the external HairGS runtime for codec-only utilities."""

    if name in __all__:
        from .hybrid_model import HybridHairGaussianModel, HybridModelConfig

        return {
            "HybridHairGaussianModel": HybridHairGaussianModel,
            "HybridModelConfig": HybridModelConfig,
        }[name]
    raise AttributeError(name)
