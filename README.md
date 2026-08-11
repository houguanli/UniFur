# UniFur: Unified Fur/Hair Reconstruction with Differentiable 3D Gaussian Rendering

> Research code snapshot, August 2026.  UniFur is an optimization-first model
> that routes surface-anchored Gaussian primitives through shell, strand, and
> residual representations.  It is intended for reproducible research rather
> than a claim of exact per-fibre recovery from monocular video.

## Current reproducible benchmark snapshot

The frozen **DFA Panda walk monocular** protocol fits one view across 32 motion
states and evaluates eight held-out views at 512 x 288 (256 images).  The
reported `Unified-balanced` configuration reaches foreground PSNR **12.233**,
masked PSNR **19.889**, masked LPIPS **0.1293**, and mask IoU **0.7696** under
`scripts/evaluate_external_renders.py`.  It improves the residual-only
ablation in appearance metrics while GART-DFA retains a higher silhouette IoU.

GART-DFA and 4D-Animal-DFA results are data-boundary adapters around their
respective upstream models, not official paper numbers.  Datasets, upstream
repositories, and model checkpoints are deliberately excluded from this repo;
see `EXTERNAL_BASELINES.md` and the scripts for setup details.

> 2026-08 update: an optimization-first unified fur/hair prototype is now
> available through `fiber-stage2`.  It starts from an ordinary Gaussian
> scaffold and learns surface-anchored `shell / strand / residual` routing for
> each sequence.  See `docs/unified_fur_hair_design.md`.

本工程将 SAM 3D Objects、MocapAnything、ElasticSimulator 和可微点渲染整合到同一个
WSL 工程中，用于研究单目动物视频的三维重建、骨架驱动物理表面、3D Gaussian
渲染和逐帧骨架优化。

工程路径：

```text
/home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction
```

## 当前结论

Cat 样例已经逐阶段审计。输入、mask、SAM3D 静态重建、相机和 Mocap 骨架本身基本正常。
腿部严重变形首次出现在 **Mocap 骨架自动绑定到 SAM3D 四面体** 的阶段。

当前自动 rigging 仍不满足最终目标：

- SAM3D mesh 和 Mocap 参考角色不是同一个 rig。
- 一部分骨段没有嵌入 SAM3D 四面体内部。
- 当前版本仍有四面体测地权重和 DQS 作为 baseline。
- 已有两条物理路径：ElasticSimulator C++/CUDA constrained FEM forward，
  以及 PyTorch 线性四面体 constrained FEM 可微 CG 反传路径。
- 图像 loss 可以下降，但可能通过错误的非物理形变下降。

因此，现阶段结果应理解为 **可微集成原型和问题定位结果**，不是已经完成的物理动物动作恢复。

## 代码与数据布局

```text
.
|-- src/dpd3dgs_animal/          集成管线、坐标变换、渲染、优化和诊断代码
|-- third_party/
|   |-- sam3d_objects/           SAM 3D Objects 源码
|   |-- mocap_anything/          MocapAnything、TripoSG 源码
|   `-- elastic_simulator/       ElasticSimulator、TetGen 及子模块
|-- checkpoints/
|   |-- sam3d/hf/                SAM3D checkpoint
|   `-- mocap_anything/
|       |-- RMBG-1.4/
|       |-- TripoSG/
|       |-- video2pose/
|       `-- video2pose2rot/
|-- samples/mocap_anything/zoo/  Cat 视频、BVH、pose、图像特征及缓存
|-- configs/
|-- scripts/
|-- tests/
|-- docs/
`-- output/                       运行结果，不纳入 Git
```

checkpoint 目录由 `.gitignore` 排除，结构说明见 `checkpoints/README.md`。

## 整体 Pipeline

```text
单目 MP4
  |
  +-- RMBG-1.4
  |     -> RGB frames
  |     -> soft alpha RGBA
  |     -> binary masks
  |
  +-- 第 0 帧 RGB + mask
  |     -> SAM3D mesh
  |     -> SAM3D Gaussian cloud
  |     -> SAM3D pose、point map、camera intrinsics
  |     -> OpenCV camera-space mesh/GS
  |
  +-- 全部 RGBA frames
  |     -> MocapAnything video2pose
  |     -> 每帧 3D skeleton joints
  |     -> 参考 BVH hierarchy
  |     -> 可选 video2pose2rot rotation prior
  |
  +-- mesh + skeleton
  |     -> 坐标轴变换和尺度对齐
  |     -> TetGen tetrahedralization
  |     -> tet boundary surface extraction
  |     -> 自动 rigging weights
  |     -> skeleton -> tet -> surface
  |
  +-- SAM3D GS + tet surface
  |     -> GS 投影到物理表面
  |     -> 邻接 surface vertices 加权绑定
  |     -> deformed surface -> deformed GS
  |
  +-- native-resolution differentiable renderer
        -> color loss
        -> high-weight mask loss
        -> elastic/bone/temporal/prior regularizers
        -> gradient to per-frame skeleton joint coordinates
```

## 与最初六项要求的对应关系

### 1. 整体环境和第三方依赖

已有 conda 环境：

```bash
cd /home/aoki/Differentiable_Physical_Driven_3DGS_for_monocular_Animal_Motion_Reconstruction
bash scripts/setup_wsl_env.sh
conda activate dpd3dgs-animal
```

三个上游项目、ElasticSimulator 子模块、样例数据和 checkpoint 均已放入本工程。
运行时不依赖 C 盘或并列源码目录。

### 2. 视频到 SAM3D 和 Mocap 先验

预处理直接从 MP4 读取，不再依赖 zoo PNG 的背景颜色阈值：

```bash
dpd3dgs-animal --config configs/cat_camera_aligned.yaml prepare-video \
  --video "samples/mocap_anything/zoo/video/Cat#Cat-Walk/y30.mp4" \
  --out-dir output/cat
```

输出：

```text
output/cat/raw_frames/
output/cat/rgba_frames/
output/cat/masks/
output/cat/sam3d_ref/
```

Mocap 目前支持：

- `video2pose`：每帧 3D joint positions。
- `video2pose2rot`：每帧 pose 和 rot6d。该模型已在 Cat 审计中跑通 NPY 输出。
- 父子关系直接读取参考 BVH，不再把 `joint_relation` 图距离矩阵误当邻接矩阵。

### 3. 四面体、骨架驱动和 GS 绑定

当前 Python 管线使用 ElasticSimulator 的 `tetrahedralize`、`extract_surface`
和 `build_surface_index_map` 构建四面体及其边界。

注意：上游 `third_party/elastic_simulator` 的 C++/CUDA solver 并不止于
TetGen。`Solver.h` 暴露 `AddKinematicCylinder`、`AttachKinematicConstraints`
和 `UpdateKinematicConstraints`，`Example/ArmBendGPU/main.cpp` 已演示
kinematic cylinder 驱动的 sparse implicit FEM。详见
`docs/elastic_simulator_constraint_audit.md`。

当前临时自动 rigging：

1. 每根骨段沿线在 tet 图中设置种子。
2. 通过四面体边图最短路径计算测地权重。
3. 使用 DQS 混合骨段刚体变换。
4. surface vertex 由对应 tet node 直接取得。

GS 绑定：

1. 每个 SAM3D Gaussian 找到最近表面投影。
2. 在最近三角面的局部 one-ring 中选多个顶点。
3. 使用加权顶点位移驱动 Gaussian，而不是一对一顶点绑定。
4. 静止帧精确保持 Gaussian 的表面投影位置。

**当前缺口：**

自动 rigging 不能代替 skeleton embedding 和带约束 FEM。Cat 的部分骨段位于 tet
体外，导致腿部权重和控制区域不可靠。

### 4. 原生分辨率渲染和 loss

渲染和 loss 默认使用输入视频分辨率，本例为 `1920x1080`。

```text
L_render = 1.0 * L_color + 10.0 * L_mask
```

- `L_color`：GT 前景区域上的 RGB L1。
- `L_mask`：预测 mask 与 GT mask 的 0/1 error，使用 STE 提供梯度。
- 另有 tet edge、tet volume、bone length、temporal 和 Mocap prior 正则项。

### 5. 反向传播到骨架节点

Stage 2 的优化变量是：

```text
[frame_count, joint_count, 3]
```

当前反向链：

```text
per-frame joints
  -> differentiable DQS
  -> tet nodes
  -> boundary surface
  -> attached GS
  -> soft splat render
  -> image losses
```

该 DQS baseline 梯度链可以运行，但不是物理平衡方程的求解结果。

### 5b. ElasticSimulator 约束 FEM forward

现在新增了一个真实 ElasticSimulator forward 路径：

```text
Stage1 NPZ
  -> TetGen .node/.ele
  -> Mocap joints + BVH parents
  -> ElasticSimulator kinematic cylinders
  -> sparse implicit FEM
  -> tet vertices per frame
  -> surface vertices
  -> attached GS
  -> native-resolution render/loss
```

入口：

```bash
PYTHONPATH=src python -m dpd3dgs_animal.cli --config configs/cat_camera_aligned.yaml \
  elastic-forward \
  --stage1-npz output/run/stage1/stage1_tet_skeleton_surface.npz \
  --gaussian-ply output/run/stage1/sam3d/sam3d_gaussian_camera.ply \
  --frame-dir output/run/preprocess/rgba_frames \
  --out-dir output/run/elastic_forward
```

实现位置：

- `third_party/elastic_simulator/Example/HeadlessConstrainedFEM/main.cpp`
- `src/dpd3dgs_animal/elastic_bridge.py`

该路径已经使用 ElasticSimulator 的 kinematic/Dirichlet constraints，但仍是
forward-only；优化骨架节点仍需要对该 constrained solve 做 implicit
differentiation 或稳定 unroll。

### 5c. 可微 constrained FEM stage2

新增 `elastic-stage2` 路径：

```text
per-frame joints
  -> skeleton segment soft Dirichlet targets
  -> linear tetrahedral FEM stiffness K
  -> differentiable conjugate-gradient solve
  -> tet nodes
  -> boundary surface
  -> attached GS
  -> native-resolution soft splat render
  -> color + high-weight mask loss
  -> gradients to per-frame joint coordinates
```

入口：

```bash
PYTHONPATH=src python -m dpd3dgs_animal.cli --config configs/cat_camera_aligned.yaml \
  elastic-stage2 \
  --stage1-npz output/run/stage1/stage1_tet_skeleton_surface.npz \
  --gaussian-ply output/run/stage1/sam3d/sam3d_gaussian_camera.ply \
  --frame-dir output/run/preprocess/rgba_frames \
  --out-dir output/run/elastic_stage2 \
  --steps 100
```

实现位置：

- `src/dpd3dgs_animal/fem_optimize.py`
- CLI: `dpd3dgs-animal elastic-stage2`

注意：该路径不是直接对 ElasticSimulator CUDA kernel 做 adjoint，而是在 PyTorch
中装配同一 tet mesh 的线性四面体 FEM 刚度矩阵，并 unroll CG 求解，因此梯度可以
回到骨架节点。它用于补齐当前端到端优化链路，C++ ElasticSimulator forward 仍是
物理 forward 对照。

### 6. 坐标系

统一基准：

```text
OpenCV camera space
x: right
y: down
z: forward
```

- SAM3D local mesh/GS 根据 SAM3D pose 转换到 OpenCV camera space。
- 相机 `world_to_camera` 在该空间中为单位矩阵。
- Mocap/BVH 的 Y-up 坐标通过 `mocap_to_opencv` 转为 Y-down。
- camera intrinsics、extrinsics 和 image-y convention 保存到 Stage 1 NPZ。
- Stage 2 直接复用 Stage 1 相机，不重新启发式拟合。

Cat 相机：

```text
resolution: 1920x1080
fx = fy = 1570.368
cx = 949
cy = 543
```

## 标准运行命令

### Stage 1

```bash
dpd3dgs-animal --config configs/cat_camera_aligned.yaml stage1 \
  --video output/cat/rgba_frames \
  --work-dir output/cat/stage1 \
  --mask output/cat/sam3d_ref/cat_ref_mask.png
```

复用已有 SAM3D/Mocap 结果：

```bash
dpd3dgs-animal --config configs/cat_camera_aligned.yaml stage1 \
  --video output/cat/rgba_frames \
  --work-dir output/cat/stage1 \
  --mesh output/cat/sam3d/sam3d_mesh_camera_simplified_5k_meshfix.obj \
  --gaussian-ply output/cat/sam3d/sam3d_gaussian_camera.ply \
  --sam3d-metadata output/cat/sam3d/sam3d_camera.json \
  --mocap-prediction output/cat/mocap/Cat_pred.npy \
  --skip-sam3d --skip-mocap
```

### Stage 2

```bash
dpd3dgs-animal --config configs/cat_camera_aligned.yaml stage2 \
  --stage1-npz output/cat/stage1/stage1_tet_skeleton_surface.npz \
  --gaussian-ply output/cat/sam3d/sam3d_gaussian_camera.ply \
  --frame-dir output/cat/rgba_frames \
  --out-dir output/cat/stage2 \
  --steps 100 \
  --lr 0.001
```

### Elastic Stage 2

```bash
dpd3dgs-animal --config configs/cat_camera_aligned.yaml elastic-stage2 \
  --stage1-npz output/cat/stage1/stage1_tet_skeleton_surface.npz \
  --gaussian-ply output/cat/sam3d/sam3d_gaussian_camera.ply \
  --frame-dir output/cat/rgba_frames \
  --out-dir output/cat/elastic_stage2 \
  --steps 100
```

### 分阶段诊断

```bash
dpd3dgs-animal --config configs/cat_camera_aligned.yaml diagnose \
  --frame-dir output/cat/rgba_frames \
  --gaussian-ply output/cat/sam3d/sam3d_gaussian_camera.ply \
  --stage1-npz output/cat/stage1/stage1_tet_skeleton_surface.npz \
  --optimized-joints output/cat/stage2/optimized_joints.npy \
  --out-dir output/cat/diagnostics \
  --frames 0,30,60,89
```

诊断输出：

```text
01_input_and_mask.png
02_sam3d_static_frame0.png
03_mocap_skeleton_overlay.png
04_surface_skinning_weights.png
05_motion_stage_comparison.png
06_tet_surface_vs_gs.png
pipeline_diagnostics.json
```

## 2026-06-19 Cat 逐阶段审计

审计根目录：

```text
output/cat_pipeline_audit_20260619
```

| 阶段 | 结果 | 结论 |
|---|---|---|
| MP4/RMBG | 通过 | alpha 和前景连续，没有旧阈值产生的身体孔洞 |
| SAM3D static | 通过 | 相机方向正确，point-map 中位重投影误差约 1.29 px |
| Mocap positions | 通过 | 正确 BVH hierarchy 后，四肢节点与视频动作基本对应 |
| Mocap rotations | 通过输出 | `video2pose2rot` 产生 90 帧 rot6d，样例平均角误差约 13.9° |
| TetGen | 通过 | 11,705 nodes，43,101 tetrahedra |
| Skeleton embedding | 未通过 | 多个骨段在 tet 体外，头/尾辅助骨最近距离最高约 0.48 |
| Automatic rigging | 未通过 | 第 30/60 帧 tet 表面腿部已经压缩 |
| GS surface binding | 通过局部检查 | GS 跟随 tet 表面，未额外引入主要腿部错误 |
| Differentiable render | 通过 | 原生分辨率、120k GS 可计算 color/mask loss |
| Skeleton optimization | 梯度通过、几何仍需改进 | DQS 与 constrained FEM loss 均可下降；腿部和外点问题还需更强 rigging/GS 清理 |

关键可视化：

- 正确 BVH 骨架：
  `output/cat_pipeline_audit_20260619/02_skeleton/skeleton_overlay_00030.png`
- tet 表面与 GS 对比：
  `output/cat_pipeline_audit_20260619/07_tet_vs_gs/tet_surface_vs_bound_gs.png`
- 最终阶段对比：
  `output/cat_pipeline_audit_20260619/14_diagnostics_final/05_motion_stage_comparison.png`
- 完整诊断指标：
  `output/cat_pipeline_audit_20260619/14_diagnostics_final/pipeline_diagnostics.json`
- Mocap pose2rot：
  `output/cat_pipeline_audit_20260619/10_mocap_pose2rot/`

10 epoch DQS 审计：

```text
total:      0.4891 -> 0.3823
color:      0.1453 -> 0.1277
mask error: 0.03438 -> 0.02546
```

虽然 loss 下降，但第 30/60 帧仍有严重腿部压缩。选定帧的 tet edge distortion
也没有同步改善，所以不能把该 loss 下降解释为物理动作恢复成功。

## 2026-06-19 Cat constrained-FEM 反传测试

输出目录：

```text
output/cat_elastic_stage2_full100_20260619
```

配置：

- 90 frames
- 原生 `1920x1080` 渲染和 loss
- 120k SAM3D Gaussian points
- 13,671 tet nodes / 50,700 tets
- 41 skeleton segment handles
- 3,744 constrained tet nodes
- `gravity=0`
- `fem_cg_iters=16`
- 100 optimization steps

结果：

| 方法 | color | mask | total/render |
|---|---:|---:|---:|
| Stage1 DQS baseline, 90-frame mean | 0.14363 | 0.03524 | 0.49605 |
| ElasticSimulator constrained FEM forward, 90-frame mean | 0.14003 | 0.03167 | 0.45673 |
| PyTorch constrained FEM stage2, step 0 | 0.14020 | 0.03163 | 0.45653 |
| PyTorch constrained FEM stage2, final eval | 0.12589 | 0.02591 | 0.38502 |

Loss history:

```text
step 0:   render 0.45653, color 0.14020, mask 0.03163
step 49:  render 0.39797, color 0.12918, mask 0.02688
step 99:  render 0.38518, color 0.12593, mask 0.02592
final:    render 0.38502, color 0.12589, mask 0.02591
```

Artifacts:

- `output/cat_elastic_stage2_full100_20260619/elastic_stage2_losses.json`
- `output/cat_elastic_stage2_full100_20260619/elastic_stage2_optimized_joints.npy`
- `output/cat_elastic_stage2_full100_20260619/elastic_stage2_optimized_state.npz`
- `output/cat_elastic_stage2_full100_20260619/previews/elastic_optimized_00000.png`
- `output/cat_elastic_stage2_full100_20260619/previews/elastic_optimized_00030.png`
- `output/cat_elastic_stage2_full100_20260619/previews/elastic_optimized_00060.png`
- `output/cat_elastic_stage2_full100_20260619/previews/elastic_optimized_00089.png`

结论：可微 constrained-FEM 反传链路已经打通，且 90 帧 Cat case 上
2D loss 明显下降；但 `cat_elastic_stage2_full100_20260619` 的 preview
视觉质量不合格，腿部发生糊连/塌缩，不能作为成功训练结果。当前视觉上更合理的是
`output/cat_elastic_stage2_20260619` 的 30 帧、20 step 短跑结果。full100
结果说明当前 loss/正则组合仍会过拟合二维 mask/color，需要用几何质量、骨长漂移、
分支约束和视觉 QA 共同做 early stopping/模型选择，而不能只看 render loss。
下一步应处理 skeleton embedding、体积权重分支约束和 Gaussian outlier/filtering。

## 腿部变形的根因

按影响顺序：

1. 旧代码将 Mocap 的全对图距离矩阵当成邻接矩阵，生成星形 parents。该问题已修复。
2. SAM3D 新 mesh 没有对应的角色 rig 和 skinning weights。
3. Mocap 骨架只做全局相似变换，没有执行骨架嵌入、关节中心拟合和肢体对应。
4. 一部分骨段位于四面体外，自动权重种子落到错误解剖区域。
5. 当前可微 constrained FEM 已能把渲染梯度传回骨架节点，但仍是线性 FEM
   PyTorch unroll，不是 ElasticSimulator CUDA kernel 的 adjoint。
6. color/mask loss 只约束二维投影，允许通过三维塌缩降低轮廓误差。

## 下一步必须完成的工作

正确方案应替换当前临时 rigging：

1. **Skeleton embedding**
   - 将关节投影/优化到 tet 内部。
   - 对四肢、脊柱、尾巴建立明确的解剖对应。
   - 排除 beard、tongue、nub 等不应控制大面积表面的辅助骨。

2. **Volumetric skinning**
   - 使用 bounded biharmonic weights、heat diffusion 或同类体积权重。
   - 权重约束在正确骨骼分支内，避免左右腿和头/前腿串权。

3. **Constrained FEM**
   - 已接入 ElasticSimulator kinematic cylinder forward solver。
   - 已接入 PyTorch 线性四面体 constrained FEM 可微 CG 反传。
   - 下一步需要把 cylinder/capsule 绑定从启发式半径升级为解剖分支内的
     skeleton embedding。
   - 重力保持为 0。

4. **Differentiation through physics**
   - 当前已对稳定 CG 求解迭代做 unroll。
   - 后续可替换为 ElasticSimulator sparse implicit 方程的 implicit differentiation。
   - 渲染梯度先到平衡后的 tet，再通过物理求解器回到 skeleton nodes。

5. **Loss 和验证**
   - 保留 color + high-weight mask。
   - 增加 tet inversion、ARAP、contact、bone capsule 和 temporal constraints。
   - 除图像 loss 外，必须报告 tet inversion 数、edge/volume distortion 和骨长误差。

在上述 skeleton embedding 和 constrained FEM 完成前，不建议继续增加 Stage 2 epoch；
更多 epoch 只能进一步优化当前错误 rigging 下的二维投影。

## 测试

```bash
PYTHONPATH=src python -m pytest -q tests/test_skinning.py tests/test_fem_optimize.py
```

当前测试覆盖：

- Cat BVH hierarchy。
- LBS/DQS 静止姿态不漂移。
- NumPy/Torch DQS 单骨骼刚体旋转一致性。
- 可微 constrained FEM solve 对 skeleton joints 有非零梯度。

开发和实验记录见 `docs/progress_log.md`。
