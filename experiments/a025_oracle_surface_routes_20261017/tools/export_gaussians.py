"""Native 3DGS PLY export, all trained fields, without route/group metadata."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch
from plyfile import PlyData


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(source_root, output_root, native_source):
    sys.path.insert(0, str(native_source.resolve()))
    from scene import GaussianModel
    import scene.gaussian_model as native_module
    assert Path(native_module.__file__).resolve() == (native_source / 'scene/gaussian_model.py').resolve()
    fields = dict(xyz='_xyz', dc='_features_dc', rest='_features_rest',
                  opacity='_opacity', scale='_scaling', rotation='_rotation')
    models = {
        'free_then_route': (215495, 'bf04de7d72c24e179d2fe5dd45ee573c3dae4ad631fb66bfd3ecf591062af242'),
        'born_routed': (122767, '876f5da3770e0ba6f9f99c1fb144ba4578d89edcdd619a11ff3749336e25af5b'),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    receipt = {}
    for arm, (count, expected_sha) in models.items():
        source = source_root / arm / 'final.npz'
        assert sha(source) == expected_sha, 'Frozen source hash mismatch'
        with np.load(source, allow_pickle=False) as model:
            original = {k:model[k].copy() for k in fields}
        assert original['xyz'].shape == (count, 3)
        assert original['dc'].shape == (count, 1, 3) and original['rest'].shape == (count, 15, 3)
        assert all(v.dtype == np.float32 and np.isfinite(v).all() for v in original.values())
        native = GaussianModel(3)
        for key, attr in fields.items():
            setattr(native, attr, torch.from_numpy(original[key]))
        native.active_sh_degree = 3
        output = output_root / (arm + '_3dgs.ply')
        assert not output.exists(), 'Refusing to overwrite an existing full GS export'
        native.save_ply(str(output))
        data = PlyData.read(output)
        names = native.construct_list_of_attributes()
        assert len(data.elements) == 1 and data['vertex'].count == count
        assert list(data['vertex'].data.dtype.names) == names and len(names) == 62
        expected = np.concatenate((original['xyz'], np.zeros_like(original['xyz']),
            original['dc'].transpose(0,2,1).reshape(count,-1),
            original['rest'].transpose(0,2,1).reshape(count,-1),
            original['opacity'], original['scale'], original['rotation']),axis=1)
        assert all(np.array_equal(data['vertex'][name], expected[:,i]) for i,name in enumerate(names)), 'Native PLY field round-trip mismatch'
        assert sha(source) == expected_sha, 'Original model changed'
        receipt[arm] = dict(source=str(source), source_sha256=expected_sha, output=str(output),
                            vertices=count, properties=names, bytes=output.stat().st_size, sha256=sha(output),
                            all_trained_fields_exact=True, SH_degree=3, group_route_binding_properties=False,
                            native_exporter_sha256=sha(Path(native_module.__file__)),
                            CPU_only=True, axis_scale_or_filter_changes=False)
    (output_root / 'EXPORT.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
