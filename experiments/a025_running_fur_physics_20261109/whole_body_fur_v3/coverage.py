"""Graded fur coverage, retaining accepted mane gains and cleanup exclusions."""
import numpy as np


def ramp(lo, hi, value):
    t = np.clip((value-lo)/(hi-lo), 0., 1.)
    return t*t*(3-2*t)


def classify(asset, semantics, patch, accepted_rows, accepted_gain):
    n = len(asset.xyz)
    anchors = asset.guide_anchors[asset.gs_guide]
    height = np.einsum('ij,ij->i', asset.xyz-anchors, asset.guide_normals[asset.gs_guide])
    length = asset.guide_lengths[asset.gs_guide]
    face = semantics['face']
    features = (face >= .65) | ((face > .3) & (asset.xyz[:, 1] < -.78) & (np.abs(asset.xyz[:, 0]) < .13))
    # Near-ground claws and the innermost shell retain body skinning.
    features |= asset.xyz[:, 2] < .025
    valid = (~features & ~patch['selected'] & (length < .4)
             & (np.exp(asset.scale).max(1) < .08)
             & (np.linalg.norm(asset.xyz-anchors, axis=1) < .35))
    region = np.ones(n, np.int64)  # 0 mane, 1 torso, 2 limbs, 3 tail
    region[anchors[:, 2] < .43] = 2
    tail = semantics['tail'] > .3
    # Tail-bone weights include the shaft and rump. Only the distal tuft
    # should receive long-fur material; shaft fur uses the short coat tier.
    tuft = tail & (anchors[:,1] > .88) & (anchors[:,2] < .40)
    region[tail & ~tuft] = 1
    region[tuft] = 3
    region[(semantics['mane'] > .3) & (face < .45)] = 0
    gain = np.zeros(n, np.float32)
    flexible = ramp(.006, .07, length)
    for group, low, high in ((0, .55, .9), (1, .2, .65), (2, .12, .35), (3, .5, .9)):
        mask = region == group
        gain[mask] = low+(high-low)*flexible[mask]
    gain *= 1-ramp(.15, .65, face)
    shell = asset.route == 0
    gain[shell] *= ramp(.001, .02, height[shell])
    gain[~valid] = 0.
    gain[gain < 1e-4] = 0.
    region[accepted_rows] = 0
    gain[accepted_rows] = accepted_gain
    notes = dict(accepted_mane_gaussians=len(accepted_rows), graded_active_gaussians=int((gain > 0).sum()),
        protected_feature_gaussians=int(features.sum()), cleanup_exclusions=int(patch['selected'].sum()),
        source_gaussians=n, region_names=['mane', 'torso', 'limbs', 'tail'],
        policy='All eligible fur gets graded motion; inner shell, features, cleanup outliers and invalid supports remain body-bound')
    accepted=np.zeros(n,bool);accepted[accepted_rows]=True
    notes['inherited_dynamic_cleanup_candidates']=int(((gain>0)&patch['selected']&accepted).sum())
    notes['new_dynamic_cleanup_candidates']=int(((gain>0)&patch['selected']&~accepted).sum())
    notes['regions'] = {name:dict(gaussians=int(((region == group) & (gain > 0)).sum()),
        shell=int(((region == group) & (gain > 0) & shell).sum()),
        strand=int(((region == group) & (gain > 0) & ~shell).sum()))
        for group, name in enumerate(notes['region_names'])}
    return region, gain, notes
