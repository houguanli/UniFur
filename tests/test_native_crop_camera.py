from types import SimpleNamespace

import numpy as np
import torch

from hairgs_hybrid import train_native


def test_native_crop_preserves_pixel_pitch_and_updates_off_axis_projection(monkeypatch):
    # Foreground branch, then deterministic center x/y.
    choices = iter((1, 1536, 512))
    monkeypatch.setattr(train_native, "randint", lambda _lo, _hi: next(choices))
    image = torch.arange(3 * 2048 * 2048, dtype=torch.float32).reshape(3, 2048, 2048)
    mask = torch.ones((2048, 2048), dtype=torch.bool)
    camera = SimpleNamespace(
        image_height=2048,
        image_width=2048,
        original_image=image,
        mask=mask,
        float_mask=mask.float(),
        masked_image=image,
        orientation_field=None,
        orientation_confidence=None,
        projection_matrix=torch.eye(4),
        full_proj_transform=torch.eye(4),
        FoVx=np.deg2rad(60.0),
        FoVy=np.deg2rad(60.0),
        image_name="view",
    )

    cropped = train_native._native_crop_camera(camera, 1024)

    # x is clamped to the right tile (1024), y to the top tile (0).
    assert cropped.native_crop_origin == (1024, 0)
    assert cropped.original_image.shape == (3, 1024, 1024)
    assert torch.equal(cropped.original_image, image[:, :1024, 1024:])
    assert cropped.full_proj_transform[0, 0].item() == 2.0
    assert cropped.full_proj_transform[1, 1].item() == 2.0
    assert cropped.full_proj_transform[3, 0].item() == -1.0
    assert cropped.full_proj_transform[3, 1].item() == 1.0
    assert np.isclose(np.tan(cropped.FoVx / 2), np.tan(camera.FoVx / 2) / 2)


def test_resume_keeps_stage_lr_and_advances_hybrid_curriculum():
    class FakeGaussians:
        _hybrid_iteration = 0

        def update_learning_rate(self, iteration):
            self.lr_iteration = iteration
            self._hybrid_iteration = iteration

        def _apply_handoff_learning_rates(self):
            self.handoff_lr_iteration = self._hybrid_iteration

    gaussians = FakeGaussians()
    train_native._update_learning_rate(gaussians, 1, 6501)

    assert gaussians.lr_iteration == 1
    assert gaussians._hybrid_iteration == 6501
    assert gaussians.handoff_lr_iteration == 6501
