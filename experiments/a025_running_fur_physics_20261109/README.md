# A025 跑动毛发物理 v2／v3

使用 A025 strict-SMAL 的 `final_prior` 混合高斯资产，按 Lion_Run 动作驱动身体并离线渲染。v2 模拟鬃毛及其填充高斯。v3 保留 v2 鬃毛的代理束和参数，并加入躯干、四肢和尾端的分级响应。两版共用渲染后处理、静息 SDF 碰撞与固定根部约束。默认按 30 FPS 采样和编码。

## 1. 准备训练导出的资产和跑步动作

- **高斯输入**使用 A800 上 `outputs/a025_afp_prior_free_gs_20261105/runs/final_prior/complete_asset.npz`。如果完整资产尚未导出，在包含 `final.npz`、`final_binding.npz` 和训练回执的 UniFur 训练环境运行 `python -B -m hairgs_hybrid.afp_prior_gs_v1.asset --run final_prior --prediction final28` 导出。现成资产可直接复制，也可以用旧查看器发布包的 `assets/models/afp_guided_free_then_route_complete_asset.npz`。两者的 SHA-256 都是 `3dacec0103f22eea97d47a11c3ac6dec2de8f329e9d7d2c04471c39fef13074d`。
- **动作输入**使用用户已有的 `UniFur_GS_Viewer_Standalone_20261105.zip` 中 `assets/animations/binding_run.npz` 及同目录下的 `motion_run_inplace/`，保留目录结构。绑定的 SHA-256 为 `08ac61c6e61c469ee8eef95793cedb2b2ba71572b4def5a3bd0a9f53de38fab1`。它包含 GS 与身体的蒙皮对应、鬃毛/尾巴/面部语义和动作清单，加载时还会校验三个网格动作缓存的哈希。
- **环境**复用现有 WSL `unifur` 环境及其 CUDA Gaussian rasterizer。已使用 Python 3.10、PyTorch 2.1.2+cu118、NumPy、SciPy、Pillow 和 trimesh。视频编码与验证分别使用 `ffmpeg` 和 `ffprobe`。

这个入口锁定了 A025 的资产和配套绑定。训练导出的 `complete_asset.npz` 带有 Shell／Strand、身体网格和绑定信息，静态 PLY 或交付包里已形变、清理和分裂的逐帧高斯用于查看或外部渲染。

## 2. 在 WSL 生成 v2 与 v3 跑动视频

在分支检出目录运行。将 `VIEWER`、`PY` 和 `OUT` 换成实际路径。单独复制训练导出时，将 `ASSET` 指到复制后的 `complete_asset.npz`。

```bash
VIEWER=/mnt/f/UniFur/virtualization/UniFur_GS_Viewer_Standalone_20261105
PY=/home/aoki/miniconda3/envs/unifur/bin/python
ASSET="$VIEWER/assets/models/afp_guided_free_then_route_complete_asset.npz"
RIG="$VIEWER/assets/animations/binding_run.npz"
OUT=/mnt/f/UniFur/virtualization/a025_rods_replay
EXP=experiments/a025_running_fur_physics_20261109

export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

"$PY" -B "$EXP/render_run.py" --version v2 --asset "$ASSET" --rig "$RIG" --output "$OUT/v2"
"$PY" -B "$EXP/render_run.py" --version v3 --asset "$ASSET" --rig "$RIG" --output "$OUT/v3" --reference "$OUT/v2"
"$PY" -B "$EXP/validate_output.py" "$OUT/v2" --mane-crop
"$PY" -B "$EXP/validate_output.py" "$OUT/v3" --mane-crop
```

每个版本使用新输出目录。v2 预设生成 390 帧（13 秒），v3 预设生成 195 帧（6.5 秒，一个跑步周期）。想先试跑时，给两个渲染命令都加 `--frames 3`，仍然执行默认的 60 帧静态预热和 60 帧跑动预热。每个版本的主视频在 `optimized/preview.mp4`。`comparison/preview.mp4` 显示同一相机的对照，`diagnostic/preview.mp4` 显示静息与变形毛束，`scene_manifest.json` 和 `report.json` 记录实际参数、覆盖与约束统计。

`--reference` 接受带有 `report.json`、`matched_gs_motion.npz` 和 `optimized/frames/` 的已渲染目录。v2 可传旧 `mane_rods_v1/result`，v3 可传上面生成的 v2 或原始 `mane_rods_v2/result`。省略它时，左侧由同步的简化风响应提供。主视频的模拟和画面由本次运行计算。

同一环境下，可以给 `validate_output.py` 加 `--against /path/to/archived/result`，逐像素核对输出的主视频 PNG 帧。验证会检查所有编码视频的帧数、播放帧率、完整解码、固定根部误差、段长误差、清理核心像素以及 v3 区域运动。

## 3. 调整参数

两套可执行预设在 [v2 JSON](presets/v2.json) 和 [v3 JSON](presets/v3.json)。复制所需预设，修改后通过 `--preset /path/to/custom.json` 加载。

- **风和动作**由 `controls.wind_strength`（2.4）、`gust_strength`（0.85）、`gust_hz`（0.65）、`wind_azimuth`（70°）和 `wind_elevation`（10°）控制。跑步速度为 `animation_speed=0.4`，风和物理时钟为 `speed=1.0`。
- **鬃毛与求解器**由 `rod` 控制。384 条鬃毛代理束，每束 12 节点，前 2 节点固定。弯曲柔度为 `1e-6`，束间柔度 `8e-5`，空气阻力 `1.2`，阻尼 `1.8`，速度扩散 `8`，根部过滤 `0.06` 秒，填充运动最大增益 `0.9`。每帧 8 子步，每子步 10 次约束迭代。
- **碰撞**使用 128 格分辨率和 `0.003` 接触边距。默认静息 SDF 随毛根局部坐标系运动，并与固定根部在同一子步更新。当前姿态 SDF 单独记录残差。该接触是跨关节近似，统计值相对于原静息嵌入允许量。
- **v3 分级与限幅**由 `regions` 控制。柔度和阻力乘数以鬃毛为基准。位移上限随代理毛长缩放，并受表中最大偏移和转角限制。已验收的 v2 鬃毛保留原响应，尾杆按短毛处理。

| v3 区域 | 新增代理束 | 柔度倍数 | 阻力倍数 | 阻尼 | 扩散 | 最大偏移 | 最大转角 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 躯干 | 192 | 0.15 | 0.55 | 4.5 | 12 | 0.018 | 0.35 rad |
| 四肢 | 128 | 0.06 | 0.35 | 6.0 | 16 | 0.008 | 0.20 rad |
| 尾端毛簇 | 96 | 0.65 | 0.80 | 3.0 | 10 | 0.045 | 0.60 rad |

v2 的独立毛束驱动覆盖 47,282 个父高斯。v3 从同一份资产得到 800 条代理束，共覆盖 167,715 个父高斯。Shell 的新增运动增益随离皮肤高度连续变化，脸部、贴肤底层和清理保护区域保留各自的身体绑定。风和材料参数使用模型单位。

## 4. 代码入口与检查

- [v2 场景](mane_rods_v2/rods_scene.py) 和 [v2 求解器](mane_rods_v2/solver.py) 保存多节点鬃毛、惯性驱动和局部接触。
- [v3 场景](whole_body_fur_v3/rods_scene.py)、[分级与保护](whole_body_fur_v3/coverage.py) 和 [v3 求解器](whole_body_fur_v3/solver.py) 增加分区材料、区域绑定和连续限幅。
- [SDF 实现](../../hairgs_hybrid/gs_dynamics_v1/sdf_collision.py) 及同包的 rig、CUDA 渲染模块来自原查看器。毛发弯曲与清理模块沿用交付逻辑。清理的 17 邻点从训练高斯重新计算，测量行从同一组鬃毛种子推导。

```bash
repo=$(pwd)
cd "$EXP"
PYTHONPATH="$repo" "$PY" -B -m mane_rods_v2.check_solver
PYTHONPATH="$repo" "$PY" -B -m whole_body_fur_v3.check_solver
PYTHONPATH="$repo" "$PY" -B -m whole_body_fur_v3.check_grading
```
