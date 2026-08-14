# Hair structure constraints used by UniFur

This note records which parts of the current implementation are controlled
adaptations of published hair-reconstruction ideas. It is not a claim that the
referenced systems use the same representation.

| UniFur component | Closest published design principle | Our adaptation |
|---|---|---|
| Scalp-only barycentric roots | Neural Strands anchors a neural scalp texture to scalp UV and creates roots for uniform scalp coverage; Dr.Hair explicitly reconstructs scalp-connected strands. | Roots are restricted to the official subject scalp vertex/face subset and stored as barycentric coordinates in the normalized head frame. |
| Root-to-tip optimization | Neural Strands uses coarse-to-fine and root-to-tip optimization; Dr.Hair uses guide/child hierarchy and global orientation initialization. | Shell/strand are exact zero-initialized residual-teacher copies. Learned per-source geometry gains unfold them continuously, preserving attachment. |
| Fixed photometric scaffold | GaussianHaircut fits classical strands against a multi-view unstructured Gaussian scene; Neural Haircut first obtains a coarse implicit hair/head volume. | A completed Residual-only 3DGS is frozen as teacher. A held-out hinge loss prevents the routed representation from exceeding the teacher loss. |
| Signed route evidence | Mixture specialization is not directly provided by the cited hair methods. | Leave-one-route-out damage is the only positive route-mass reward. If ablation improves held-out loss, current mass on that route is directly penalized. |
| Multi-view support | Hair-GS jointly supervises RGB, masks and 2D orientations, then merges/refines short Gaussian segments; Neural Haircut reconciles coarse volume/silhouette evidence with strand priors. | Calibrated hair masks form a conservative multi-view support gate. Unsupported strand samples are hidden and a soft projection loss supplies gradients; root-to-tip prefix connectivity prevents floating distal fragments. |
| Silhouette fins | UnityFurURP emits segmented fin cross-sections only for grazing face/view configurations and parameterizes them root-to-tip. | Shell samples become thin anisotropic Gaussian ribbons with a differentiable grazing-angle opacity gate and multi-view silhouette-band activation. The complete residual teacher stays additive and frozen. |
| Single-view protocol | Im2Haircut combines a learned global prior with local image-space optimization, using coarse-to-fine PCA stages. | It is evaluated separately as a single-view baseline; its learned prior is not mixed into the calibrated multi-view table. |

Primary sources:

- Neural Strands (ECCV 2022): <https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930070.pdf>
- Neural Haircut (ICCV 2023): <https://openaccess.thecvf.com/content/ICCV2023/papers/Sklyarova_Neural_Haircut_Prior-Guided_Strand-Based_Hair_Reconstruction_ICCV_2023_paper.pdf>
- Dr.Hair (CVPR 2024): <https://openaccess.thecvf.com/content/CVPR2024/papers/Takimoto_Dr.Hair_Reconstructing_Scalp-Connected_Hair_Strands_without_Pre-Training_via_Differentiable_Rendering_CVPR_2024_paper.pdf>
- GaussianHaircut: <https://arxiv.org/abs/2409.14778>
- Hair-GS: <https://yimin-pan.github.io/hair-gs/>
- Im2Haircut: <https://im2haircut.is.tue.mpg.de/>
- UnityFurURP Fin source: <https://github.com/hecomi/UnityFurURP/blob/main/Assets/Fur/Shaders/Fin/Lit.hlsl>

Protocol warning: the released NeuralHaircut `person_0/ckpt_final.pth` is a
subject-specific reconstruction trained with the released person views. It
must not be placed in the strict odd-fit/even-held-out table unless retrained
using only odd views. GaussianHaircut runs initialized from that checkpoint
must carry the same leakage warning.
