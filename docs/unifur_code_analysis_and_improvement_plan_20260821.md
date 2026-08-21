# UniFur 代码实现分析与改进计划

> 审查日期：2026-08-21  
> 审查目录：`E:\ComputerGraphics\code\UniFur`  
> 审查提交：`bbbdad3`（`main`，工作树干净）  
> 文档类型：代码实现审查、问题诊断与实验改进计划  
> 验证状态：`ANALYZED`——本次结论来自代码、配置、测试定义和仓库实验记录的只读核查；未重新运行 CUDA/HairGS 训练或完整测试。

## 1. 执行摘要

UniFur 已经形成了一套较完整的统一结构化毛发 Gaussian 原型：它从普通 3DGS/HairGS 毛发点云出发，将共享 Gaussian source 迁移到 `shell / strand / residual` 三类渲染专家，并在统一的 HairGS differentiable rasterizer 中联合优化几何、外观、路由和结构部署。代码还实现了固定 head/body 底座、表面 root atlas、多视角方向 lifting、方向场传播、visual hull、双向 mask、最大孔洞约束、teacher non-regression、带局部 warmup 和回滚的 topology event，以及面向编辑/模拟的 carrier assignment。

但是，目前系统主要解决的是“训练可见区域中的结构迁移和局部优化”，尚未真正解决“把已观测区域的结构知识传播到完全不可见区域”。当前传播只覆盖方向场，并且依赖已有、可见、mask-supported 的 surface root；topology birth 也只是激活预分配 row，不能创建新的三维 root。因而无监督背面仍可能缺少 Gaussian 覆盖，或者继承错误的 root 深度、密度与排布。

当前最重要的改进不是继续叠加二维 loss，而是完成以下三个结构闭环：

1. 建立独立于 Stage-I Gaussian 密度的 canonical scalp/fur atlas；
2. 区分 foreground、background 和 unknown，使完全不可见区域由先验传播而不是被 visual hull 裁掉；
3. 实现能够新增或重新绑定 `face_index/barycentric` 的真实动态三维 root topology。

## 2. 审查范围与证据边界

### 2.1 审查内容

本次主要检查了：

- `src/dpd3dgs_animal/fiber.py`：统一表示、路由、shell/strand 几何、carrier 和正则；
- `src/dpd3dgs_animal/fiber_optimize.py`：训练流程、损失、多视角传播、visual hull 和 topology；
- `src/dpd3dgs_animal/hairgs_renderer.py`：HairGS CUDA rasterizer 适配；
- `src/dpd3dgs_animal/scaffold.py`：静态/动态表面 scaffold 和渲染损失；
- `src/dpd3dgs_animal/gaussian.py`：Gaussian PLY 与表面绑定；
- `src/dpd3dgs_animal/fiber_evaluate.py`：统一评测；
- `src/dpd3dgs_animal/fiber_route_audit.py`：leave-one-route-out 路由审计；
- wCurly、Panda 等维护配置和运行脚本；
- 仓库内测试定义、里程碑记录与实验评估文档。

### 2.2 证据边界

- 当前仓库可复现配置主要到 wCurly v21；仓库评估文档还记录了 v22/v23 的外部实验结果。
- 评估文档明确说明 v23 当时来自未提交工作树，缺少精确可获取的 Git commit。因此 v23 指标可以作为历史实验依据，但不能视为当前 `bbbdad3` 的一键可复现结果。
- 仓库文档记录测试状态为 `99 passed, 1 skipped`；本次没有重新运行依赖 CUDA/HairGS 环境的测试和训练。

## 3. 当前实现采用的方法

### 3.1 输入、初始化与表面绑定

UniFur 是 per-scene optimization 系统，不是 feed-forward 毛发重建网络。输入包括：

- 标定多视角 RGB；
- hair/fur mask；
- 可选的 HairGS/GaussianHaircut orientation map；
- surface/motion scaffold；
- 初始 Gaussian PLY；
- 可选固定 head/body Gaussian PLY；
- residual-only bootstrap checkpoint。

毛发 Gaussian 可通过 `nearest_vertex` 或 `closest_surface` 绑定到三角形表面，并保存：

- `face_index`；
- barycentric 坐标；
- tangent、bitangent、normal 局部坐标系；
- residual local offset；
- 邻域图和 scalp occupancy。

当前 wCurly v21 配置仍使用 `nearest_vertex` 绑定，但会在初始化传播阶段将部分重复 source row 重新分配到面积分层的 scalp atlas。

### 3.2 统一 Gaussian 表示

每个 source 共享基础颜色、opacity 和初始协方差，同时拥有三类渲染路由：

1. `shell`：短毛/体积性毛发专家；
2. `strand`：长发和方向性纤维专家；
3. `residual`：普通各向异性 3DGS 专家或迁移 teacher。

此外，系统还维护与渲染 route 分离的 `surface / shell / strand` carrier assignment，用于后续编辑、形变和模拟。这样，负责渲染的 expert 和负责运动传播的 carrier 不必完全相同。

### 3.3 Shell/Fin expert

Shell 分支从 surface root 沿一个短方向生成若干 Gaussian。当前 wCurly v21 配置使用两个 sample。每个 Gaussian 为薄的各向异性 ribbon，其三个尺度轴分别对应：

- 毛发生长方向；
- Fin 横向宽度；
- Fin 厚度。

Shell 方向由 surface normal 与传播得到的毛流方向混合。渲染时还使用类似 Unity Fur Fin 的 grazing-angle gate：视线越接近表面切向，Fin opacity 越高。

因此当前 `shell` 更准确地说是“短毛 Gaussian + silhouette fin expert”，而不是 NeuralFur 那种真正多层、连续的 neural shell density field。

### 3.4 Strand expert

Strand 分支把每个 root 参数化为一条三次曲线：

```text
p(t) = root + height * normal
       + length * (t * direction + t^2 * bend2 + t^3 * bend3)
```

当前主要可学习参数包括：

- 3D root-to-tip 方向；
- 长度；
- quadratic bend；
- cubic bend；
- 半径；
- structured geometry gain；
- route-specific color 和高阶 SH；
- strand route probability。

wCurly 配置中每条曲线固定采样五个 Gaussian，所有 sample 使用相同横向半径。目前没有：

- root-to-tip taper；
- 自适应段数；
- strand split/merge；
- guide/child hierarchy；
- 不同 source strand 之间的显式连接拓扑。

### 3.5 自适应路由与守恒迁移

系统以 per-source logits 学习 shell、strand、residual 的 softmax 分布。residual trust 会把部分概率重新分配给 residual；训练中可以使用 soft mixture，部署时支持：

- argmax hard route；
- mass-preserving hard route；
- forced route；
- leave-one-route-out；
- route dropout。

结构化迁移采用零位移初始化：shell/strand 起初与 residual teacher 共点、共协方差，然后由 `structured_delta_gain` 逐渐展开到解析结构。opacity 采用透射率守恒拆分，使共点 soft routes 在初始化时近似保持 teacher 的总透射率。

wCurly Hair 配置使用 residual-free adaptive migration：residual checkpoint 仍提供初始化和校准 teacher，但最终可部署容量只保留 shell/strand。

### 3.6 多视角方向场和表面传播

系统从 HairGS 风格的二维轴向 orientation map 中恢复三维方向。每个视角提供一个投影约束，多视角置信度加权的 normal matrix 最小特征向量作为候选 3D 方向。随后执行：

- outward normal bias；
- root-to-tip 符号同步；
- 邻域方向平滑；
- normal-aligned parallel transport；
- shell 与 strand 不同强度的方向消费。

传播图目前由 rest-space root 的三维欧氏 KNN 构造，不是 mesh geodesic graph。

### 3.7 Root atlas 与重复 root 重分配

初始化传播阶段首先量化 `face_index + barycentric`，找出重复 anchor。系统选择低 opacity 的重复 source row，将其重新绑定到多视角 mask 支持的 face，并在 face 内进行面积分层采样。

Hair 配置进一步构建 scalp occupancy 和 area-stratified scalp atlas：

- 一部分 atlas row 初始化为 strand；
- 其余 row 作为 shell 或 topology grow reserve；
- residual local offset 被重新计算，以尽量保持初始化渲染等价。

这种方法能减少 Stage-I `nearest_vertex` 导致的完全重复 root，但 atlas 仍受预分配 source row 数量和支持 face 集合限制。

### 3.8 Visual hull 与可见性门控

Shell 和 strand 的完整解析目标都会投影到多视角 mask 中。系统统计：

- sample 在多少视角有效；
- mask 支持次数和支持率；
- 是否被表面或其他 sample 遮挡；
- root-to-tip prefix 是否连续。

不满足支持条件的 sample 被 persistent gate 隐藏。prefix `cumprod` 会在首个无支持 sample 后裁掉所有更远端 sample，避免浮空 tip。

### 3.9 Topology event

训练期间周期性测量：

- residual footprint support；
- structured support；
- mask deficit；
- silhouette boundary；
- 多视角可见次数。

对 grow 候选，系统从已有 surface root 沿邻域传播方向进行 ray marching，选择被多个 mask 支持的长度作为新的 root-to-tip proposal；然后激活该 row 的 strand route，设置长度、方向和 bend，并只对局部 cluster 进行短程优化。

事件最终在预留 calibration views 上比较前后逐视角损失。如果任何视角退化超过阈值，或平均改进不足，整个事件回滚。

### 3.10 损失函数

当前 loss 大致分为六组。

#### 外观损失

- foreground RGB L1；
- masked RGB gradient；
- route-specific color/SH 正则。

#### 双向轮廓损失

- soft alpha mask；
- foreground/background balanced mask；
- boundary loss；
- mask 内 coverage；
- mask 外 spill；
- structured-only coverage/spill；
- 最大连通孔洞的 soft top-k surrogate。

#### 二维方向损失

- HairGS 风格的二倍角 orientation；
- 四阶轴向矩，保留交叉/卷曲的局部分布信息；
- confidence-weighted orientation consistency。

#### 三维结构损失

- shell/strand visual hull；
- Fin silhouette support；
- strand support activation；
- strand field 邻域一致性；
- strand deployability 和有效 coverage；
- shell normal、长度、strand thinness、height、bend；
- root barycentric 正则。

#### 路由和 teacher 约束

- route entropy；
- route prior KL；
- route neighbor smoothness；
- route ablation risk；
- negative contribution penalty；
- residual teacher non-regression；
- calibration view event validation。

#### Carrier 约束

- carrier entropy/prior；
- 邻域一致性；
- carrier attachment；
- tip prior；
- carrier 与 render route family alignment。

## 4. 无监督视角覆盖失败的直接原因

### 4.1 Surface propagation 只在已观测支持区域重新分配 root

Face 只有在满足以下条件时才会进入传播 atlas：

- 至少在配置数量的视角中有效可见；
- 投影位于 hair/fur mask 的比例达到阈值。

当前 wCurly v21 要求至少三个视角、支持率至少 0.5。完全不可见的背面 face 不会进入 eligible set，因此不会获得重新分配的 root。

### 4.2 Topology grow 依赖已有、可见的 surface root

Grow 候选必须满足：

- 当前 row 还没有 active strand；
- root 在多个视角可见；
- 多个视角存在 mask deficit；
- 解析 strand 目标本身已获得足够 mask 支持。

这意味着系统只能在“已经存在合法 root，且训练视角能观察到该 root”的位置增长。若缺口来自完全未覆盖表面，或 root depth/face 一开始就错误，当前 grow 候选空间中没有正确解。

### 4.3 Visual hull 把完全不可见区域视为不支持

Occlusion-aware visual hull 将被遮挡视角视为 unknown，但如果一个 sample 在所有训练视角都没有有效观测，代码仍要求至少一次支持。结果是 valid count 为零的 sample gate 为零，最终被裁掉。

因此当前 unknown 不是可由先验补全的状态，而更接近“不可部署”。

### 4.4 Coverage seed 在当前 Hair 配置中关闭

wCurly v21 设置：

```yaml
fiber_coverage_seed_count: 0
```

所以 frozen-teacher deficit coverage seeding 没有参与该实验。即使开启，该机制仍从现有 surface root 出发，并要求多个已观测视角支持，不能独立解决完全未观测区域。

### 4.5 当前 birth 不是新的 root topology

Topology grow 实际执行的是：

- 修改 `route_active_gate`；
- 设置 strand route mass；
- 写入 strand length、direction、bend；
- 清理局部 Adam state。

它没有：

- append 新 source；
- 创建未绑定的 root；
- 在事件中更新 `face_index`；
- 在事件中重新求解 barycentric/scalp UV；
- 从相机缺口射线直接三角化 root depth。

所以所谓“3D birth”目前只对 root-to-tip ray proposal 成立，不对 root topology 成立。

### 4.6 当前传播只传播方向，不传播完整结构属性

仓库测试验证了一个无观测节点可以从邻居获得方向，但没有传播：

- root occupancy；
- shell/strand density；
- 长度；
- 曲率；
- taper；
- opacity；
- appearance/SH；
- uncertainty。

这不足以从局部观测完成整个对象的毛发结构。

## 5. 当前代码中的关键风险

### 5.1 P0：HairGS 方向角约定可能仍不一致

当前 renderer 输出：

```text
[y^2 - x^2, 2xy]
```

这对应 HairGS 从图像 y 轴计量的方向角，即二维 tangent 为：

```text
[sin(theta), cos(theta)]
```

但当前多视角 lifting 构造的垂直方向是：

```text
[-sin(theta), cos(theta)]
```

它对应的是 `[cos(theta), sin(theta)]` 约定。对于 HairGS 的 y-axis convention，正确垂线应为：

```text
[-cos(theta), sin(theta)]
```

或其反向。

仓库文档称 v23 已修复这个问题，但 v23 修改未包含在当前提交中。因此应先修正当前代码并加入合成多相机测试，再开展后续 topology 实验。

### 5.2 P0：Hair 路由存在 shell 逃逸

当方向、tip support 或长 strand 难以优化时，shell 更容易通过短、粗、局部 Gaussian 降低 RGB/mask loss。仓库记录中 wCurly v23 的有效 route mass 约为：

- shell：75.8%；
- strand：24.2%；
- residual：0。

这满足了 residual-free，但没有形成 strand-dominant 的 Hair 结构。二维外观可以改善，物理语义却可能退化成大量短针状 shell。

### 5.3 P0：Root 重复和 novel-view 缺口同时存在

历史 v16 中，108,981 个 source 通过 nearest-vertex binding 只得到 5,802 个 distinct root。后续 scalp atlas 已明显改善，但 v23 审计仍记录近重复 root 比例约 0.279，同时 novel view 仍有明显孔洞。

这说明问题主要是空间分布和深度错误，而不是 Gaussian 总数不足。

### 5.4 P1：欧氏 KNN 不等价于曲面邻域

`_surface_knn_indices` 使用三维 cKDTree。它可能连接：

- 耳朵两侧；
- 头皮薄层的相对表面；
- 空间上接近但拓扑不连通的部位；
- 卷发穿过或靠近头部的不同结构层。

方向传播、route smoothness、birth cluster 和符号同步都可能因此发生跨层泄漏。

### 5.5 P1：所谓方向场不是严格的切向场

传播的是完整三维单位方向，包括 normal component；可靠性不足的点直接回退到 normal，并额外加入 outward normal bias。这对短 fur 合理，但容易使 Hair 方向保持较大的径向成分。

更合适的参数化是分别建模：

- scalp tangent angle；
- outward elevation；
- root-to-tip polarity；
- length/curvature。

### 5.6 P1：Strand 表示容量不足

当前固定五段、恒定半径的 cubic strand 很难表示长卷发。历史审计记录：

- GT strand 中位 sample 数约 67；
- UniFur analytic strand 只有 5 个 sample；
- learned median strand length 远短于 GT；
- 没有 tip taper；
- 会产生短、粗、交叉和发尾过亮的问题。

### 5.7 P1：尺度与 opacity 存在病态解

仓库 v23 审计记录有效 Gaussian 最大 aspect ratio 约 262，p99 约 50。极端大协方差可以在一个视角填洞，却在其他视角造成 spill、粗大发尾和过亮轮廓。

仅依靠全局 soft regularizer 不足，应对每个 route 的 world-space scale、screen-space footprint、opacity-scale product 做显式限制和 split/merge。

### 5.8 P1：严格回滚可能形成过度保守的局部最优

Teacher non-regression 和 topology validation 能防止点数爆炸和明显退化，但如果 teacher/scaffold 本身在缺口区域错误，任何必要的结构重排都可能短期破坏若干 calibration view，从而被整体回滚。

历史 v23 记录 23 次 topology event、148 个 proposal、0 个 accepted，说明当前系统是“安全但无效”。

## 6. 建议的改进方案

### 6.1 P0：修正方向约定并锁定可复现版本

首先应：

1. 修复 y-axis HairGS orientation lifting；
2. 添加已知 3D direction 的多相机投影—恢复单元测试；
3. 将 v23 或其后续版本正式提交；
4. 为每个实验记录 commit、config hash、seed、protocol id 和 artifact hash；
5. 对 v21 与方向修正版进行严格单变量对照。

方向初始化是 atlas propagation 和 topology birth 的共同基础。该问题未固定前，继续调整 route/topology 会混入不可解释的变量。

### 6.2 P0：建立独立 canonical scalp/fur atlas

建议将 root topology 从 Stage-I Gaussian source 中解耦。每个 canonical atlas cell 可以维护：

```text
fur occupancy probability
shell density
strand density
tangent angle
outward elevation
strand length
curvature coefficients
root height
root/tip radius and taper
appearance feature
observation confidence / uncertainty
```

Stage-I Gaussian 只提供 radiance、opacity 和局部外观初始化，不再定义 root 数量与空间分布。

Atlas 应支持：

- mesh geodesic blue-noise/Poisson root sampling；
- root merge 和 repulsion；
- root 跨 face 迁移；
- scalp UV 上连续优化；
- shell 与 strand 分别维护容量；
- fur-bearing surface mask，而不是只依赖当前视角 visual hull。

### 6.3 P0：使用 foreground/background/unknown 三态可见性

当前 binary support 应升级为三态：

1. `observed foreground`：正监督；
2. `observed background`：强负监督；
3. `occluded/unobserved`：不施加 visual-hull 负监督，由结构先验和邻域传播决定。

对 unknown 区域应保留低置信结构，而不是直接将 opacity/gate 清零。每个 atlas cell 应显式记录：

- supporting view count；
- rejecting view count；
- unknown view count；
- completion uncertainty。

低置信度可以限制 opacity、长度和频率，但不能等价于“该处没有毛发”。

### 6.4 P0：实现真正动态的三维 root birth

建议的 birth 流程如下：

1. 在每个 calibration/proxy view 中提取最大的连通 false-negative 区域；
2. 使用相机标定、mask 边界和 orientation 建立跨视角缺口对应；
3. 从至少两个到三个一致视角发射相机 ray；
4. 联合求解 root、tip 和 depth；
5. 将 root 投影到最近的合法 scalp atlas，允许重新指定 `face_index` 和 barycentric/UV；
6. 从 geodesic 邻域初始化方向、长度、曲率、SH、opacity 和 taper；
7. 动态 append parameter row，或从“尚未绑定 face 的空 slot”分配新 row；
8. 重建邻域图并正确初始化 optimizer state；
9. 只优化 birth cluster 和一环邻域；
10. 使用最大孔洞、spill、方向重投影和多个 calibration view 联合验收。

验收不应只看平均 RGB/mask loss。更合理的条件包括：

- 最大连通孔洞必须下降；
- 至少两个 proxy/calibration view 获得正贡献；
- spill 不得超过阈值；
- root 必须落在合法 scalp 区域；
- tip 必须获得多视角支持；
- 新结构必须具有非零 deployed opacity-weighted mass。

### 6.5 P1：使用 geodesic graph 和真实曲面传播

将欧氏 KNN 替换为：

- mesh one-ring/multi-ring adjacency；
- scalp UV 邻域；
- heat-method/geodesic 距离；
- 沿 mesh path 累计的 parallel transport。

建议分别传播：

- tangent angle；
- elevation；
- log shell density；
- log strand density；
- log length；
- curvature；
- appearance latent；
- uncertainty。

方向等轴向变量应使用圆统计或复数二倍角表示，避免在 `theta` 和 `theta + pi` 之间平均出错误方向。

### 6.6 P1：增加 surface-field completion

#### 单对象、无额外训练集

可以先采用非学习式或弱学习式补全：

- confidence-weighted harmonic extension；
- Laplacian/Poisson completion；
- 左右对称先验；
- 分区域长度/密度统计；
- 训练时随机隐藏已观测 atlas patch，再要求从周围恢复属性。

这种方法能传播局部规律，但不能唯一推断完全不可见的大尺度发型。

#### 多对象训练集

若有多个动物/发型数据，可以训练共享的 atlas GNN、UV neural field 或 Transformer：

```text
multi-view observed features
        + canonical atlas coordinates
        + surface geometry
        + category/global latent
        -> density, route, direction, length, curvature, uncertainty
```

训练时使用 visibility dropout 或 surface-patch masking，让模型学习从局部可见区域补全人为遮挡区域。这才是真正意义上的“把已观测区域知识传播到整个对象”。

### 6.7 P1：增强 Strand 表示

建议：

- 每根 strand 使用自适应 8–16 段；
- 按弧长采样而不是固定参数 `t` 等距；
- 加入 root-to-tip 单调 taper；
- 根据 screen-space error 或曲率进行 segment split/merge；
- 增加 guide strand + child strand hierarchy；
- 监督整条 strand 的多视角投影曲线；
- 对长度和曲率使用分段/频谱先验；
- 对相邻 guide 的 root/tip continuity 做结构约束。

### 6.8 P1：将 Shell 升级为真正的短毛体积专家

当前两个 Fin sample 对密集 underfur 较弱。可以考虑：

- 多个法向 shell layer；
- 每层独立 density、length 和 orientation variance；
- shell/fin 混合：内部用体积 shell，silhouette 使用 grazing Fin；
- 对不同层进行 transmittance-conserving opacity budget；
- 根据视距和曲率自适应采样层数。

### 6.9 P1：允许同一区域的 underfur 与 guard hair 共存

当前 softmax 让 shell、strand、residual 竞争同一份概率质量；hard 部署时每个 source 更是只能选择一个 expert。动物毛发往往需要同一区域同时存在：

- 密集短 underfur；
- 稀疏长 guard hair/whisker。

建议将路由改为：

- 独立 `shell_presence`；
- 独立 `strand_presence`；
- 分别受总 optical-thickness budget 约束；
- residual 仅作为 teacher/scaffold，不作为同级部署 expert。

这样既允许共存，又避免 opacity 双计数。

### 6.10 P1：防止 Hair 退化到 shell

建议：

1. 分别预热 shell 和 strand expert，再开放 router；
2. router 使用共享局部网络，而不是每点完全独立 logits；
3. 输入 orientation confidence、长度证据、曲率、silhouette band、geodesic position 和 visibility；
4. 对 Hair 设置 opacity-weighted deployed strand contribution 下限；
5. 启用 leave-one-route-out 正贡献监督；
6. 不奖励只有名义 route mass、但 geometry gain/opacity 接近零的结构；
7. 在 tip support、root coverage 未达标前限制 shell 抢占长发区域。

## 7. 推荐实验与消融顺序

建议严格按单变量或小步组合推进。

| 实验 | 相对前一步的修改 | 核心目的 |
|---|---|---|
| E0 | 当前 v21 | 可复现基线 |
| E1 | 只修 HairGS y-axis orientation lifting | 验证方向初始化影响 |
| E2 | geodesic graph + root 去重/Poisson atlas | 验证空间分布影响 |
| E3 | foreground/background/unknown 三态 | 验证 unseen culling 是否为主要原因 |
| E4 | atlas property harmonic completion | 验证单对象知识传播 |
| E5 | 真正动态 root birth | 验证候选 topology 是否为核心瓶颈 |
| E6 | adaptive segment + taper | 验证 strand parameterization 上限 |
| E7 | 独立 shell/strand presence | 验证双层毛发和 route collapse |
| E8 | 多对象 completion network | 验证学习型未观测区域先验 |

### 7.1 Novel-view 设计

除均匀分布 held-out camera 外，还应增加：

- 连续相机弧段留出；
- 完整背面留出；
- 顶部/底部极角留出；
- 人为 atlas patch 遮挡；
- 稀疏训练视角递减实验。

均匀留出视角通常仍被相邻训练相机覆盖，不能充分验证“完全无图像监督区域”的补全能力。

### 7.2 图像指标

- foreground PSNR；
- masked SSIM；
- masked LPIPS；
- mask IoU/F1；
- mask MAE；
- background opacity；
- 最大连通孔洞；
- spill 面积和最坏视角指标。

### 7.3 三维结构指标

- geodesic root coverage；
- near-duplicate root fraction；
- root density coefficient of variation；
- strand tip 多视角 mask support；
- orientation reprojection error；
- strand arc-length 分布；
- curvature/taper 分布；
- inward fraction；
- screen-space footprint；
- route-specific aspect-ratio/opacity-scale product；
- topology proposal、accept 和 rollback 数量。

### 7.4 路由指标

- nominal route mass；
- deployed opacity-weighted route mass；
- leave-one-route-out PSNR/IoU drop；
- shell-only/strand-only render；
- Hair 和 Fur 的区域性 route map；
- soft/hard route 差距。

### 7.5 下游结构价值

至少建立：

1. 局部梳理/长度编辑：测量目标位移、非目标泄漏、root 滑移；
2. 风场/重力模拟：测量碰撞穿透、连续性、帧间闪烁；
3. 头部形变或姿态变化：测量附着误差、体积保持和 novel-view 质量。

## 8. 当前实验记录的含义

仓库评估文档记录：

- Panda held-out：UniFur v18 soft 的 FG PSNR 约 20.70 dB，低于 residual-only 的 24.26 dB；
- wCurly held-out：HairGS 约 16.46 dB / IoU 0.857；UniFur v23 soft 约 16.28 dB / IoU 0.793；
- wCurly v23：strand tip support 约 0.751，近重复 root 约 0.279；
- 23 次 topology event、148 个 proposal、0 accepted；
- 最大有效 Gaussian aspect ratio 约 262。

这些结果支持以下判断：

1. 当前问题不是 Gaussian 总数不足；
2. mask loss 和 visual hull 数量已经足够多；
3. 错误主要来自 root topology、深度、结构参数化和路由逃逸；
4. 继续只调整 loss 权重不太可能形成质变；
5. 真正值得形成论文贡献的是 canonical surface-field completion、dynamic 3D root topology 和 dual shell/strand Gaussian experts 的闭环。

## 9. 建议的实施顺序

### 第一阶段：正确性与可复现性

- 修复 orientation convention；
- 建立合成多相机方向测试；
- 正式提交并锁定最新实验版本；
- 重新运行统一评测和几何审计。

### 第二阶段：表面覆盖与 unknown 建模

- 使用 geodesic graph；
- 建立独立 canonical atlas；
- 实现三态 visibility；
- 对方向、密度和长度做 confidence-aware completion。

### 第三阶段：真实动态 topology

- 多视角缺口匹配；
- root/tip/depth 联合求解；
- 动态 append 或未绑定 slot；
- root 跨 face 重绑定；
- cluster-level warmup 和结构化验收。

### 第四阶段：表示和路由增强

- adaptive strand segments；
- taper 和 guide/child hierarchy；
- multi-layer shell；
- 独立 shell/strand presence；
- contribution-aware router。

### 第五阶段：完整论文证据

- Hair/Fur 分开报告；
- static/dynamic/single-view 分开报告；
- soft/hard 路由同时报告；
- 与 residual-only、HairGS、NeuralFur 做同协议比较；
- 将编辑与模拟指标升级为主要评价轴。

## 10. 最终判断

“把已观测区域学到的毛发结构传播给整个对象”在方法上是可行的，但需要明确传播的对象和可推断边界：

- 局部方向、密度、长度和短程纹理可以通过 canonical surface field、geodesic propagation 和 self-masking 学习传播；
- 完全不可见区域的复杂卷发形状无法仅凭单对象局部观测唯一恢复，需要对称、类别或跨对象生成先验；
- 所有补全结果都应伴随 uncertainty，而不是把推断结构当作真实观测；
- 当前 UniFur 已具备方向传播和结构化 expert 的基础，但 root topology、unknown 建模和学习型 completion 尚未闭环。

因此，下一版最合理的研究主线是：

> 在独立 canonical scalp/fur atlas 上学习带置信度的 shell/strand 结构场，通过可见区域自监督补全未观测区域，并以多视角 deficit triangulation 动态创建真实三维 root，使短密 fur 与长连续 hair 能在统一 Gaussian renderer 中共存和自适应部署。

