#!/usr/bin/env python3
"""Verify that UniFur runs only against its pinned in-repository HairGS."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

from packaging.requirements import Requirement


EXPECTED_HAIRGS_COMMIT = "16588656b1f6f048bc3bc83f3cb98c2da8596754"


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _module_path(name: str) -> Path:
    module = importlib.import_module(name)
    location = getattr(module, "__file__", None)
    if not location:
        raise RuntimeError(f"Module {name} has no filesystem location")
    return Path(location).resolve()


def _verify_pins(root: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for relative in (
        "requirements/runtime.txt",
        "requirements/cuda118.txt",
        "requirements/pytorch3d-cu118.txt",
        "requirements/dev.txt",
    ):
        for raw in (root / relative).read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith(("#", "-")):
                continue
            requirement = Requirement(line)
            version = importlib.metadata.version(requirement.name)
            if version not in requirement.specifier:
                raise RuntimeError(
                    f"{requirement.name} {version} violates pin {requirement.specifier}"
                )
            versions[requirement.name] = version
    return versions


def _verify_local_build_origins(hairgs: Path) -> dict[str, str]:
    origins: dict[str, str] = {}
    for distribution_name in (
        "c_utils",
        "diff_gaussian_rasterization",
        "simple_knn",
        "hair-gs",
    ):
        distribution = importlib.metadata.distribution(distribution_name)
        payload = distribution.read_text("direct_url.json")
        if payload is None:
            raise RuntimeError(f"{distribution_name} has no PEP 610 build origin")
        url = json.loads(payload).get("url", "")
        parsed = urlparse(url)
        source = Path(unquote(parsed.path)).resolve()
        if parsed.scheme != "file" or not _inside(source, hairgs):
            raise RuntimeError(
                f"{distribution_name} was not built from the pinned submodule: {url}"
            )
        origins[distribution_name] = str(source)
    return origins


def _cuda_roundtrip() -> dict[str, object]:
    import torch
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda")
    means3d = torch.tensor([[0.0, 0.0, 1.0]], device=device, requires_grad=True)
    means2d = torch.zeros((1, 3), device=device, requires_grad=True)
    colors = torch.tensor([[0.8, 0.4, 0.2]], device=device, requires_grad=True)
    opacity = torch.tensor([[0.8]], device=device, requires_grad=True)
    scales = torch.tensor([[0.1, 0.1, 0.1]], device=device, requires_grad=True)
    rotations = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]], device=device, requires_grad=True
    )
    identity = torch.eye(4, device=device)
    settings = GaussianRasterizationSettings(
        image_height=16,
        image_width=16,
        tanfovx=1.0,
        tanfovy=1.0,
        bg=torch.zeros(3, device=device),
        scale_modifier=1.0,
        viewmatrix=identity,
        projmatrix=identity,
        sh_degree=0,
        campos=torch.zeros(3, device=device),
        prefiltered=False,
        debug=False,
    )
    image, radii = GaussianRasterizer(settings)(
        means3D=means3d,
        means2D=means2d,
        colors_precomp=colors,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
    )
    image.sum().backward()
    if not torch.isfinite(image).all() or means3d.grad is None:
        raise RuntimeError("CUDA rasterizer forward/backward produced invalid tensors")
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "visible_gaussians": int((radii > 0).sum().item()),
        "image_sum": float(image.detach().sum().cpu()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--skip-cuda", action="store_true")
    args = parser.parse_args()

    root = args.repository_root.resolve()
    hairgs = (root / "third_party/hair-gs").resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(hairgs), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != EXPECTED_HAIRGS_COMMIT:
        raise RuntimeError(f"HairGS commit mismatch: {commit}")

    module_paths = {
        name: _module_path(name)
        for name in (
            "arguments",
            "gaussian_renderer",
            "scene.hair_gaussian_model",
            "hairgs_hybrid.hybrid_model",
            "c_utils",
            "diff_gaussian_rasterization",
            "diff_gaussian_rasterization._C",
            "simple_knn._C",
            "pytorch3d",
        )
    }
    for name in ("arguments", "gaussian_renderer", "scene.hair_gaussian_model"):
        if not _inside(module_paths[name], hairgs):
            raise RuntimeError(f"{name} was imported outside the submodule: {module_paths[name]}")
    if not _inside(module_paths["hairgs_hybrid.hybrid_model"], root):
        raise RuntimeError("UniFur package was imported outside the repository")

    report: dict[str, object] = {
        "schema": "unifur-reproducibility-check-v1",
        "python": sys.version.split()[0],
        "repository_root": str(root),
        "hairgs_commit": commit,
        "module_paths": {name: str(path) for name, path in module_paths.items()},
        "native_build_origins": _verify_local_build_origins(hairgs),
        "pinned_versions": _verify_pins(root),
    }
    if not args.skip_cuda:
        report["cuda_roundtrip"] = _cuda_roundtrip()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
