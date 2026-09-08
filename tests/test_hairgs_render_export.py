from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image

import scripts.export_hairgs_external_renders as exporter


def _install_fake_colmap(monkeypatch: pytest.MonkeyPatch) -> None:
    data = ModuleType("data")
    colmap = ModuleType("data.colmap")

    def read_cameras(_path: Path) -> dict[int, SimpleNamespace]:
        return {1: SimpleNamespace(name="view.png")}

    colmap.read_extrinsics_binary = read_cameras
    colmap.read_extrinsics_text = read_cameras
    data.colmap = colmap
    monkeypatch.setitem(sys.modules, "data", data)
    monkeypatch.setitem(sys.modules, "data.colmap", colmap)


def _protocol(tmp_path: Path, split: str, view_index: int) -> tuple[Path, Path]:
    dataset = tmp_path / split
    sparse = dataset / "sparse/0"
    sparse.mkdir(parents=True)
    (sparse / "images.txt").write_text("fake", encoding="utf-8")
    manifest = dataset / "camera_manifest.json"
    manifest.write_text(
        json.dumps(
            {"observations": [{"image": "view.png", "view_index": view_index}]}
        ),
        encoding="utf-8",
    )
    return dataset, manifest


def _write_render(model: Path, color: tuple[int, int, int], mask: bool = True) -> Path:
    root = model / "render/train/iteration_8000"
    rgb = root / "renders/rgb"
    rgb.mkdir(parents=True)
    Image.new("RGB", (2, 2), color).save(rgb / "00000.png")
    if mask:
        mask_dir = root / "renders/mask_foreground"
        mask_dir.mkdir(parents=True)
        Image.new("L", (2, 2), 255).save(mask_dir / "00000.png")
    return root


def _argv(
    model: Path, dataset: Path, camera_manifest: Path, output: Path
) -> list[str]:
    return [
        "--model-dir",
        str(model),
        "--camera-dataset",
        str(dataset),
        "--camera-manifest",
        str(camera_manifest),
        "--output-dir",
        str(output),
        "--iteration",
        "8000",
        "--output-size",
        "2",
        "2",
        "--hairgs-root",
        str(model.parent / "fake-hairgs"),
    ]


def test_real_png_main_keeps_same_iteration_splits_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_colmap(monkeypatch)
    model = tmp_path / "model"
    train_dataset, train_camera = _protocol(tmp_path, "train", 7)
    test_dataset, test_camera = _protocol(tmp_path, "test", 9)
    train_output = tmp_path / "evaluation_train/external_renders"
    test_output = tmp_path / "evaluation_test/external_renders"

    mutable = _write_render(model, (10, 20, 30))
    assert exporter.main(_argv(model, train_dataset, train_camera, train_output)) == 0
    train_source = train_output / "source_render/iteration_8000/renders/rgb/00000.png"
    train_hash = exporter._sha256_file(train_source)

    assert _write_render(model, (100, 110, 120)) == mutable
    assert exporter.main(_argv(model, test_dataset, test_camera, test_output)) == 0
    test_source = test_output / "source_render/iteration_8000/renders/rgb/00000.png"

    assert train_source != test_source
    assert exporter._sha256_file(train_source) == train_hash
    assert exporter._sha256_file(test_source) != train_hash
    manifest = json.loads((train_output / "render_manifest.json").read_text())
    assert manifest["observations"][0]["source_rgb_sha256"] == train_hash


def test_missing_mask_fails_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_colmap(monkeypatch)
    model = tmp_path / "model"
    dataset, camera = _protocol(tmp_path, "train", 7)
    output = tmp_path / "evaluation_train/external_renders"
    mutable = _write_render(model, (10, 20, 30), mask=False)

    with pytest.raises(ValueError, match="render set mismatch"):
        exporter.main(_argv(model, dataset, camera, output))

    assert mutable.is_dir()
    assert not output.exists()
    assert not output.with_name(output.name + ".incomplete").exists()


def test_failure_after_claim_resumes_without_rerender(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_colmap(monkeypatch)
    model = tmp_path / "model"
    dataset, camera = _protocol(tmp_path, "train", 7)
    output = tmp_path / "evaluation_train/external_renders"
    mutable = _write_render(model, (10, 20, 30))
    argv = _argv(model, dataset, camera, output)
    real_save = exporter.np.savez_compressed

    def injected_failure(*_args, **_kwargs) -> None:
        raise RuntimeError("injected after claim")

    monkeypatch.setattr(exporter.np, "savez_compressed", injected_failure)
    with pytest.raises(RuntimeError, match="injected after claim"):
        exporter.main(argv)
    stage = output.with_name(output.name + ".incomplete")
    assert not mutable.exists()
    assert (stage / "export_claim.json").is_file()
    assert (stage / "source_render/iteration_8000").is_dir()
    assert not output.exists()

    monkeypatch.setattr(exporter.np, "savez_compressed", real_save)
    camera.write_text(
        json.dumps({"observations": [{"image": "view.png", "view_index": 8}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="claim differs"):
        exporter.main([*argv, "--resume-existing"])
    assert (stage / "source_render/iteration_8000").is_dir()
    camera.write_text(
        json.dumps({"observations": [{"image": "view.png", "view_index": 7}]}),
        encoding="utf-8",
    )
    assert exporter.main([*argv, "--resume-existing"]) == 0
    assert output.is_dir()
    assert not stage.exists()


def test_complete_export_resumes_evaluator_without_rerender(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_colmap(monkeypatch)
    model = tmp_path / "model"
    dataset, camera = _protocol(tmp_path, "train", 7)
    output = tmp_path / "evaluation_train/external_renders"
    argv = _argv(model, dataset, camera, output)
    assert exporter.main([*argv, "--resume-existing"]) == exporter.NEEDS_RENDER
    assert not output.with_name(output.name + ".incomplete").exists()
    mutable = _write_render(model, (10, 20, 30))
    mutable_rgb = mutable / "renders/rgb/00000.png"
    mutable_hash = exporter._sha256_file(mutable_rgb)
    with pytest.raises(FileExistsError, match="unknown protocol content"):
        exporter.main([*argv, "--resume-existing"])
    assert exporter._sha256_file(mutable_rgb) == mutable_hash
    assert exporter.main(argv) == 0
    manifest = output / "render_manifest.json"
    before = exporter._sha256_file(manifest)
    assert not mutable.exists()

    assert exporter.main([*argv, "--resume-existing"]) == 0

    assert exporter._sha256_file(manifest) == before
    assert not mutable.exists()
