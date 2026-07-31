# stereo-depth pipeline

一条命令，从原始图像跑到过滤后的深度图。

只做这一件事。点云、融合、mesh、多模型对比（MonSter / DEFOM / SEDNet / Depth-Anything）
这些都已经移到 `_archive/`。

```
原始图像
   │
   │  ── COLMAP ──────────────────────────────────────
   │  colmap      SfM + 去畸变（一步跑完 4 条 colmap 命令）
   │  transforms  去畸变模型 -> transforms.json
   │
   │  ── nerfstudio ──────────────────────────────────
   │  train       训练辐射场
   │  export      从模型导出训练视角的相机位姿
   │  shift       在相机自身坐标系下平移出右目位姿
   │  campath     转成 nerfstudio 的 camera_path 格式
   │  render      渲染左目（训练视角）+ 右目
   │
   │  ── 立体匹配 ────────────────────────────────────
   │  rotate      旋转到立体匹配要求的朝向
   │  intrinsic   生成 K.txt（内参 + 换算成真实尺度的基线）
   │  stereo      FoundationStereo 推理
   │  depth       深度图转回原始朝向 + 存 16 位 PNG
   │  filter      多视图几何一致性过滤，剔除不可信深度
   ▼
深度图 (.npy 米制 / .png 16 位)
```

## 用法

```bash
./run_pipeline.sh -c config.json                      # 全流程
./run_pipeline.sh -c config.json --from render        # 从某步开始
./run_pipeline.sh -c config.json --only stereo,depth,filter
./run_pipeline.sh -c config.json --show               # 只看解析后的完整配置
```

**所有参数都在 `config.json` 里改**，命令行只有流程控制和四个常用覆盖
（`--shift` / `--shift-direction` / `--vis` / `--clean`）。

```json
{
  "project":    { "root": "/path/to/scene" },
  "colmap":     { "enabled": true, "image_dir": "images", "matcher": "exhaustive" },
  "nerfstudio": { "method": "nerfacto", "max_num_iterations": 30000 },
  "stereo":     { "shift_mode": "pixels", "shift_pixels": 0.1, "direction": "up" },
  "filter":     { "enabled": true, "max_depth_error": 0.01, "use_ncc": true },
  "output":     { "depth_png": true, "clean_intermediate": false, "visualize": false }
}
```

没写的项走 `lib/load_config.py` 里的默认值，所以配置文件只需要写要改的部分。
相对路径按 config.json 所在目录解析，配置可以跟着场景走。

已经有 `transforms.json` 的话把 `colmap.enabled` 设成 `false`；
已经训好模型的话用 `--from export`，或在配置里指定 `nerfstudio.config_path`。

## COLMAP 那一段

`colmap` 这一步把四条命令包成一条：

```
feature_extractor → exhaustive_matcher → mapper → image_undistorter
```

每步的产物存在就自动跳过，中断了可以直接重跑。视频抽帧的场景把
`colmap.matcher` 设成 `sequential` 会快很多。

**为什么一定要去畸变。** mapper 出来的相机模型（OPENCV / RADIAL 等）带畸变系数，
那些系数要一路传到训练、渲染、立体匹配，任何一环漏了就错。
`image_undistorter` 之后只剩 `fx/fy/cx/cy`，下游全都干净。
所以 `transforms` 这步默认要求输入是去畸变模型，带畸变会明确报错并提示怎么做。

转换是**直接读二进制**的（`lib/colmap_model.py`）。旧版是
`cameras.bin → cameras.txt →`（shell 出去调 instant-ngp 的 `colmap2nerf.py`）`→ json`，
绕了一圈还依赖外部脚本。

坐标约定：COLMAP 存的是 world-to-camera、OpenCV 系（x 右 y 下 z 前），
nerfstudio 要 camera-to-world、OpenGL 系（x 右 y 上 z 后），所以

```
transform_matrix = inv(w2c) @ diag(1, -1, -1, 1)
```

验证方式是双向的：造一个已知位姿的 COLMAP 二进制模型，
先让真的 `colmap model_analyzer` 读一遍（验证二进制格式），
再用转换器转回来跟已知的 transforms.json 比对——**最大误差 1.3e-15**。

## 目录

```
scripts/
├── run_pipeline.sh              唯一入口
├── config.json                  所有配置
├── lib/
│   ├── load_config.py           config.json -> shell 变量
│   ├── run_colmap.py            一步跑完 COLMAP SfM + 去畸变
│   ├── colmap_model.py          COLMAP 二进制模型读取
│   ├── colmap_to_transforms.py  -> transforms.json
│   ├── export_poses.py          从模型导出相机位姿
│   ├── stereo_shift.py          左目位姿 -> 右目位姿
│   ├── make_camera_path.py      -> nerfstudio camera_path 格式
│   ├── rotate_images.py         旋转左右目图像
│   ├── make_intrinsics.py       生成 K.txt
│   ├── depth_postprocess.py     深度转回朝向 + 存 PNG
│   ├── filter_depth.py          多视图几何一致性过滤
│   └── common.py                旋转对应表 + 内参旋转 + K.txt 读写
└── _archive/                    旧脚本，没删，需要时可以翻
```

场景目录：

```
<project.root>/
├── images/            原始图像（colmap.image_dir）
├── colmap/            SfM 产物
│   ├── database.db
│   ├── sparse/0/      带畸变的原始模型
│   └── undistorted/   去畸变图像 + 模型  <- 训练和渲染都用这份
├── transforms.json    由去畸变模型生成
├── outputs/           nerfstudio 训练输出
└── stereo_depth/      本 pipeline 的产物
```

产物都在 `WORK_DIR`（默认 `DATA_DIR/stereo_depth`）：

```
stereo_depth/
├── cam_path/     位姿、camera_path、K.txt
├── render/       left/ right/   渲染出的原始视角
├── rotated/      left/ right/   旋转后送进网络的
├── raw_depth/    FoundationStereo 的原始输出
├── depth/        深度图（.npy 米制 + .png 16 位）
├── depth_filtered/  过滤后的深度（不可信像素置 0）
└── vis_filter/   过滤可视化三联图（只在 --vis 时生成）
```

## 磁盘占用

深度图是 float32 的 `.npy`，很占地方。30 帧 640×480 一次跑下来约 108M，
其中一半以上是中间产物：

| 目录 | 占用 | 说明 |
|---|---|---|
| `raw_depth/` | 36M | FoundationStereo 原始输出，**和 `depth/` 的 .npy 只差一个朝向** |
| `depth/` .npy | 36M | 最终产物 |
| `depth/` .png | 2.9M | 16 位 PNG，纯可视化 / 喂 depth-nerfacto 用 |
| `rotated/` | 19M | 送进网络前的旋转副本，纯中间产物 |
| `render/` | 16M | 渲染的训练视角 + 双目视角，交付物 |

放大到 500 帧 1080p 大约是一次 10G。三个选项：

```json
"output": {
  "depth_png": false,          // 深度只存 .npy，不存 16 位 PNG
  "clean_intermediate": true,  // 跑完删掉 rotated/ 和 raw_depth/
  "visualize": false           // 过滤的可视化三联图
}
```

`clean_intermediate` 实测 108M → 55M。注意删掉 `raw_depth/` 之后就没法单独重跑
`--only depth` 了，要连 `stereo` 一起重跑。`depth_png: false` 只是不再新建 PNG，
不会去删已经存在的。

## 深度过滤

FoundationStereo 对每个像素都给稠密深度，不管那个像素是不是真的可信。
`filter` 步骤用相机位姿做多视图交叉验证——一个像素的深度如果是对的，
把它投到别的视角再投回来，应该能回到原处。

四项判据，前三项纯几何、第四项光度：

| # | 判据 | 参数 | 默认 |
|---|---|---|---|
| 1 | 前后向重投影误差 | `--filter-max-reproj` | 2.0 px |
| 2 | 相对深度误差 | `--filter-max-depth-error` | 0.01 |
| 3 | 三角化角 | （config.sh `FILTER_MIN_TRI_ANGLE`） | 3.0° |
| 4 | NCC 光度一致性 | `--filter-min-ncc` | 0.3 |
| | 至少几票才保留 | `--filter-min-consistent` | 2 |
| | 用几个源视角 | `--filter-num-src` | 8 |

判据 1 是**双向投影**：ref → src → ref 走一个来回，看能不能回到原处。
（实现上有个坑：反向反投影必须用正向落点那个**非整数**坐标构造射线，
不能用源视角的整数像素网格。）

判据 4 是后加的——前三项管不住"纹理很差、深度靠猜但恰好几何自洽"的区域。
NCC 用盒式滤波实现，不逐窗口循环：warp 一次源图到参考视角，然后
`NCC = (E[rw] - E[r]E[w]) / sqrt(Var[r]·Var[w])`，每一项都是 `avg_pool2d`。

**无纹理区 NCC 弃权而不是否决。** 分母趋零时 NCC 被噪声主导，
此时它不该表态——没纹理不代表深度错。参考窗口局部标准差低于
`FILTER_MIN_TEXTURE_STD`（默认 0.02）就跳过这一项。这不是可选的润色：
实测不弃权的话保留率从 50% 崩到 13.6%，而纯度反而更差。

过滤的全部参数在 `config.json` 的 `filter` 段；`--vis` 单独用命令行开关
（可视化很占地方，默认不生成）。

**不需要法向。** 法向是 PatchMatch *估计*深度时用的（倾斜支撑窗口）；
这里只验证已有深度的自洽性，而且 NCC 的 warp 用的是每个像素**自己的深度**
而不是拟合平面，斜面只要深度局部准确就能对上。

### 并行策略

PatchMatch 的深度估计有传播依赖（第 r 行要等第 r-1 行），COLMAP 因此只能
每列开一个线程串行扫——1600×1200 的图只有 1600 个线程，在 170 SM 的卡上
占用率不到 1%，120 个 SM 全程空闲。

但**过滤没有这个依赖**：每个 (像素, 源视角) 对都独立，可以整块张量一次算完。
这里就是纯 PyTorch 批量算子，480×640 配 8 个源视角 = 246 万个并行单元。
实测 **30 视角 0.35 秒（11.6 ms/视角）**。

### 实测效果

用解析合成场景光追出真值深度来验证（30 视角，640×480）：

| | 保留像素 | 中位相对误差 | δ<5% | δ<10% |
|---|---|---|---|---|
| 过滤前 | 100% | 1.08% | 82.6% | 91.5% |
| 过滤后（默认阈值） | 50.1% | **0.47%** | **98.7%** | **99.5%** |

把它当作"坏像素检测器"（坏 = 相对误差 >5%）：
**召回 96.2%**（坏的基本都抓到），**精确率 33.6%**（剔掉的里有 2/3 其实是好的），
留下来的**纯度 98.7%**。

也就是说默认阈值偏保守——和 COLMAP 一个取向，宁可给洞也不给错。
想要更高的覆盖就放松阈值：

| `max_depth_error` | `max_reproj` | `min_consistent` | 保留 | 召回 | 精确 | 纯度 |
|---|---|---|---|---|---|---|
| 0.005 | 1.0 | 3 | 37.7% | 99.4% | 27.8% | 99.7% |
| **0.01** | **2.0** | **2** | **50.1%** | **96.2%** | **33.6%** | **98.7%** |
| 0.02 | 3.0 | 2 | 58.9% | 91.3% | 38.7% | 97.4% |
| 0.05 | 4.0 | 2 | 71.4% | 76.5% | 46.6% | 94.3% |
| 0.05 | 4.0 | 1 | 85.4% | 53.6% | 63.7% | 90.5% |

跑完会打印逐判据的拒绝统计，方便定位是哪一关卡得太死。默认参数下
相对深度误差那关否掉 24.2%、重投影 16.9%、三角化角 11.7%、NCC 0.4%。

### 关于 NCC 的一个实测说明

在这个合成测试场景上 **NCC 几乎不加分**：

| | 保留 | 召回 | 精确 | 纯度 |
|---|---|---|---|---|
| 仅几何三项 | 50.1% | 96.2% | 33.6% | 98.7% |
| + NCC 0.3 | 49.8% | 96.5% | 33.4% | 98.8% |
| + NCC 0.3（**不弃权**） | 13.6% | 98.7% | 19.9% | 98.3% |

原因是这个场景 **69~73% 的区域局部标准差低于 0.02**（大面积平滑着色），
NCC 在那里本来就没信号——弃权比例 77.1%，只有 0.4% 真被 NCC 否决。
真实拍摄的纹理场景上这一项会比这里重要得多。

这也说明测试数据没法验证 NCC 的效果，所以默认阈值取的是保守的 0.3，
并且把 NCC 的有效均值和弃权比例都打印出来，方便你在自己的数据上标定。

## 基线怎么定

默认用**目标视差占图像宽度的百分比**，比 nerfstudio 归一化坐标系下的抽象数值直观：

```json
"stereo": { "shift_mode": "pixels", "shift_pixels": 0.1 }
```

`shift_pixels: 0.1` = 640 宽的图上视差 64 像素。

但要清楚一件事：**视差随深度变化**，`d = fx·B/Z`，一个基线不可能让全图都是 64 像素。
所以这个百分比是锚定在**场景参考深度**上的——近处视差更大、远处更小，
这是立体几何本身的性质，不是实现的将就。

参考深度从 **COLMAP 稀疏点云**统计（对每张图取它实际看到的 3D 点的深度，
再取分位数）。不需要渲染，读文件 + numpy 投影，毫秒级。

**默认取 25% 分位而不是中位数**——锚近处能把近平面的视差压住。测试场景实测：

| 锚定 | 基线 | Z=0.88m | Z=1.05m | Z=2.42m | Z=5.44m |
|---|---|---|---|---|---|
| 50% 分位 | 0.310 m | 175px (27.4%) | 147px (23.0%) | 64px (10.0%) | 28px (4.5%) |
| **25% 分位** | 0.135 m | 76px (11.9%) | 64px (10.0%) | 28px (4.3%) | 12px (1.9%) |

锚中位数的话近平面视差冲到 27% 宽度，容易超出立体匹配网络的视差范围、
遮挡也更严重；锚 25% 压到 12%，好处理得多。

想固定参考深度就写死：

```json
"stereo": { "reference_depth": 2.5, "reference_depth_percentile": 25 }
```

没有 COLMAP 点云又没写 `reference_depth` 会明确报错，不会静默取个默认值。

旧的直接给基线的方式还在：`"shift_mode": "baseline", "shift": 0.2`。

换算链路（`lib/resolve_shift.py` → `lib/make_intrinsics.py`）：

```
baseline_metric = shift_pixels · W · Z_ref / fx      # 目标视差 -> 真实基线
shift_norm      = baseline_metric · scale            # -> nerfstudio 归一化空间
                  ... 位姿平移在归一化空间发生 ...
baseline        = shift_norm / scale                 # -> 换回真实尺度给 FoundationStereo
```

实测闭环精确（5% / 10% / 20% 反推误差 0）；稀疏点云估的参考深度在测试场景上
与真值差 3.3%（中位数 2.343 vs 2.422 m）。

跑的时候会把各深度分位上的实际视差打出来，一眼能看到跨深度的分布是否合适。

## 一个容易踩的点

**基线尺度**。渲染用的位姿在 nerfstudio 的归一化坐标系里，而 FoundationStereo
要用真实尺度的基线算 `depth = fx * baseline / disp`。
`dataparser_transforms.json` 里的 `scale` 就是这两者的换算系数。
这一步弄错的话深度会整体差一个常数倍。

**旋转**。立体匹配网络要求视差是水平的。当基线不是沿 `left` 方向时，
渲染出的图必须先旋转到那个约定，跑完再把深度转回原始朝向；内参也得跟着转，
否则 fx 用错、主点偏移。对应关系只有一份，在 `lib/common.py`。

`direction` 写的是**右目画面相对左目往哪边移**——也就是深度转回原始朝向后，
你实际看到的位移方向。比如 `up` 就是右目画面整体向上平移：

| 方向 | 图像/内参旋转 | 深度转回 | 旧写法 |
|---|---|---|---|
| `up`    | 90° 逆时针 | 90° 顺时针 | `y`  |
| `down`  | 90° 顺时针 | 90° 逆时针 | `-y` |
| `left`  | 不转       | 不转       | `-x` |
| `right` | 180        | 180        | `x`  |

旧的 `x/-x/y/-y` 仍然接受，只是不再宣传：那套写的是相机在自身坐标系里往哪个轴平移，
而相机往下移画面内容是往上跑的，方向正好相反，很容易记反。

这张表是用投影几何算出来、并且和实际渲染对过的（`y`+0.2 实测右图内容上移 87 px）。

### 方向自动跟随轨迹

**NeRF 只在相机真正去过的视角附近训练充分。** 沿轨迹方向平移，虚拟右目落在
观测过的视角流形内；垂直于轨迹平移就是外推，渲染会明显变糊、出伪影。
极端情况：轨迹全是上下运动却选了左右平移，右目渲染的是 NeRF 从没见过的视角。

```json
"stereo": { "auto_direction": true, "auto_direction_min_dominance": 0.6 }
```

判定方式（`lib/auto_direction.py`）：对每帧取轨迹切向（相机中心的中心差分），
投影到该帧自己的相机系，看 x 还是 y 分量占优，按多数投票定一个全局方向。

各类轨迹的实测判定：

| 轨迹 | 主导性 | 符号一致性 | 判定 |
|---|---|---|---|
| 水平直线 | 99.2% | 100% | `left` |
| 垂直直线 | 100% | 100% | `down` |
| 绕圈 360° | 100% | 100% | `left` |
| 水平往返 | 99.1% | **53%** | `left`（轴对，正负取多数） |

绕圈之所以主导性 100%，是因为相机盯着中心绕行时切向在相机系里恒为水平——
本来就该判成水平。真正模糊的是手持自由移动，那种情况主导性会掉到 0.6 以下，
脚本会警告并退回配置里的 `direction`，同时建议分段跑。

**为什么只出一个全局方向**：不同方向对应不同的图像旋转和内参旋转，
而下游 `batch_process.py` 整批共用一份 K.txt，逐帧变方向就得逐帧一个 K 文件。

关掉：`"auto_direction": false`，或命令行 `--no-auto-direction`
（`--shift-direction` 也会自动关掉它）。

## 环境

pipeline 跨两个 conda 环境，脚本直接用各自的解释器绝对路径调用，
不做 `conda activate` / `deactivate` 来回切换：

| 步骤 | 环境 |
|---|---|
| `export` `render` | `nerfstudio_sm120` |
| 其余全部 | `nevstereo` |

FoundationStereo 的代码和权重都在项目内，不依赖外部盘：

```
Nevstereo/
├── third_party/FoundationStereo/          代码
└── models/foundation_stereo/
    ├── model_best_bp2.pth                 权重 3.1G
    └── cfg.yaml
```

### RTX 5090 适配

原来的 `foundation` / `foundation_stereo` 两个 conda 环境**在 5090 上跑不了**：
`torch 2.4.1+cu121` 和 `flash_attn` 的 `.so` 都只编到 `sm_90`，没有 `sm_120`。

现在改成在 `nevstereo` 环境跑（`torch 2.8.0+cu128`，含 `sm_120`），
flash-attn 则在 `third_party/FoundationStereo/core/submodule.py` 里改成可选：

仓库里 `flash_attn_func` 的所有调用点用的都是默认 `window_size=(-1,-1)`，
在 flash-attn 语义中就是不做滑窗的非因果全注意力，与
`torch.nn.functional.scaled_dot_product_attention` 数学等价
（SDPA 内部同样会选 FlashAttention 后端）。所以缺 flash-attn 时回退到 SDPA，
省掉了为 sm_120 从源码编译 flash-attn 的几个小时。装了 flash-attn 就照旧用它。

验证方式：把一张图整体水平平移 24 像素当右图，真值视差处处为 24。
模型估计出 **23.99 px（误差 0.01）**——注意力算错的话视差会是垃圾，
所以这同时验证了 SDPA 替换的正确性。

顺带说明：FoundationStereo 内部用 DepthAnything 的 ViT 当特征提取骨干
（`core/extractor.py`），那是模型结构的一部分，和被归档掉的独立脚本
`depth_anything_infer.py` 不是一回事，删不得。

## 相对旧版的改动

旧的 `process_stereo_foundation.sh` 是 804 行，其中几百行是内联 heredoc 生成的
临时 Python 脚本。主要问题和对应处理：

| 旧版 | 现在 |
|---|---|
| 路径硬编码 `/mnt/h/RGBD-500`、`/mnt/f/algorithm_F` | 全部走 `config.sh` + 命令行参数 |
| 左目渲染那行被注释掉了，只渲染右目 | 左右目都渲染 |
| `colmap_convert_to_camera_path.py` 收的是相对路径 `transforms.json`，依赖当前目录 | 传绝对路径 |
| 旋转对应表抄了三份（且注释成"与另一处相反"） | 收敛到 `lib/common.py` 一处 |
| 旋转方向靠 grep `DEFAULT_SHIFT=` 反推 | 直接传基线方向 |
| `conda activate` / `deactivate` + `cuda_switch.sh` 来回切 | 用解释器绝对路径 |
| 深度写回 `transforms.json` 时按文件名转 int 当帧号 | 按 `file_path` 的 stem 匹配 |
| 结尾接一段 depth-nerfacto 训练 | 去掉，到深度图为止 |
| 参数名 `--disp-dir`，但 `batch_process.py` 存的其实已经是深度 | 改叫 `--depth-in` |
