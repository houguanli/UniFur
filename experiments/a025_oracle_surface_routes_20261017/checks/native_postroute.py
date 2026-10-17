"""Independent post-fit decoder/backward/reload check. No formal model is touched."""
from pathlib import Path
import sys
import tempfile
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run as study
from geometry import postroute


def main():
    study.source_lock(); m = study.native(); views = study.load_views(limit=1)
    model, cameras, radius = study.initial(m, views, 'free_then_route', count=1024)
    model.active_sh_degree = 3
    with torch.no_grad():
        model._scaling[:64, 2] += np.log(5)
        before = study.rgb_alpha(m, cameras[0], model)
        model.attachment = dict(prior='metadata only')
        after = study.rgb_alpha(m, cameras[0], model)
        assert torch.equal(before[0]['render'], after[0]['render']) and torch.equal(before[1], after[1])
    a, binding, report = postroute(study.arrays(model), study.body(), study.CONFIG)
    assert len(binding['strand_rows']) >= 64
    study.assign(model, a); connected = study.bind_connected(m, model, binding); connected.active_sh_degree = 3
    connected.training_setup(m.opt)
    pkg, alpha = study.rgb_alpha(m, cameras[0], connected)
    loss = (pkg['render'] - cameras[0].original_image).square().mean() + (alpha - cameras[0].target_alpha).square().mean()
    loss.backward()
    assert connected.nodes.grad is not None and torch.isfinite(connected.nodes.grad).all() and connected.nodes.grad.abs().sum() > 0
    for field in study.FIELDS.values():
        parameter = getattr(connected, field)
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), field
    actual = study.arrays(connected)
    actual['xyz'] = connected.get_xyz.detach().cpu().numpy()
    actual['scale'] = connected.get_scaling.detach().log().cpu().numpy()
    actual['rotation'] = connected.get_rotation.detach().cpu().numpy()
    reloaded = m.GaussianModel(3); study.assign(reloaded, actual); reloaded.active_sh_degree = 3
    with torch.no_grad():
        frozen = study.rgb_alpha(m, cameras[0], connected); replay = study.rgb_alpha(m, cameras[0], reloaded)
        error = max(float((frozen[0]['render'] - replay[0]['render']).abs().max()), float((frozen[1] - replay[1]).abs().max()))
        assert error < 2e-6
        original = connected.nodes.detach().clone(); connected.nodes[:, 0].add_(.001)
        moved = study.rgb_alpha(m, cameras[0], connected)
        response = float((moved[0]['render'] - frozen[0]['render']).abs().max())
        assert response > 0
        connected.nodes.copy_(original)
    study.write(study.HERE / 'preflight/native_postroute.json', dict(status='passed', source_lock=study.source_lock(),
        native_resolution=[2048, 2048], sampled_GS=1024, shared_node_gradients=True, actual_image_response=response,
        materialized_decoder_reload_RGB_alpha_max_error=error, metadata_bind_error=0, report=report,
        synthetic_elongation_for_branch_coverage=True, no_formal_model_modified=True, heldout_read=False))


if __name__ == '__main__': main()
