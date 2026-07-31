# Nevstereo — RTX 5090 (Blackwell / sm_120) 环境

两个 conda 环境分工，互不干扰：

| 环境 | 用途 | 位置 |
|---|---|---|
| `nerfstudio_sm120` | nerfstudio（nerfacto / splatfacto） | `~/miniconda3/envs/nerfstudio_sm120` |
| `nevstereo` | droid_metric（DROID-SLAM + Metric3D） | `~/miniconda3/envs/nevstereo` |

代码、权重、输出都在 `/mnt/g/algorithm_backup/Nevstereo/` 下。

```
Nevstereo/
├── install_nevstereo.sh      安装脚本（幂等，可重复跑）
├── env_droid.sh              source 它进 nevstereo 环境
├── env_nerfstudio.sh         source 它进 nerfstudio_sm120 环境
├── test_sm120_kernels.py     CUDA kernel 数值验证
├── testdata/synth/           合成测试序列（30 帧，带真值轨迹）
└── droid_metric/
    ├── weights/              预训练权重 (6.9G)
    └── modules/
        ├── droid_slam/       子模块，提供 CUDA 扩展源码
        └── metric3d/         子模块
```

## 用法

```bash
# 1. droid_metric：出深度 / 位姿 / mesh
source /mnt/g/algorithm_backup/Nevstereo/env_droid.sh
python reconstruct.py --input <图片目录或视频> --output <输出目录> --viz

# 2. nerfstudio：训练辐射场
source /mnt/g/algorithm_backup/Nevstereo/env_nerfstudio.sh
ns-train nerfacto --data <处理后的目录>
```

安装脚本支持分步执行：`bash install_nevstereo.sh {check|repo|patch|env|build|models|verify}`。

## 版本

**nevstereo**（本次新建）

| 组件 | 版本 |
|---|---|
| Python | 3.11 |
| torch | 2.8.0+cu128（`arch_list` 含 `sm_120`） |
| torchvision | 0.23.0+cu128 |
| torch_scatter | 2.1.2（源码编译） |
| numpy | 1.26.4 |
| droid_backends / lietorch | 本地编译 sm_120 |

**nerfstudio_sm120**（既有环境，安装脚本只读校验，不修改）

| 组件 | 版本 |
|---|---|
| Python | 3.10 |
| torch | 2.12.0.dev+cu128（`arch_list` 含 `sm_120`） |
| tinycudann | 2.0 |
| nerfacc | 0.5.2 |
| gsplat | 1.4.0 |
| nerfstudio | 1.1.5 |

## 打了哪些补丁

安装脚本的 `patch` 步骤会改 6 处，全部幂等，原文件都备份成 `*.orig`。
前三个是 5090/新 torch 必需，后三个是上游 bug。

| | 文件 | 问题 |
|---|---|---|
| **A** | `droid_slam/setup.py` | nvcc gencode 硬编码到 `sm_86`。5090 上编出的 `.so` 没有 sm_120 kernel image，运行时报 `no kernel image is available for execution on the device`。两处（`droid_backends` + `lietorch_backends`）都改成 `compute_120/sm_120` |
| **B** | `droid_slam/src/*.cu` | 2022 年的代码用 `AT_DISPATCH_xxx(tensor.type(), ...)`，新版 torch 里 `.type()` 返回 `at::DeprecatedTypeProperties`，而 `AT_DISPATCH` 要 `c10::ScalarType`，**编译直接失败**。改成 `.scalar_type()`，共 3 处 |
| **C** | `lietorch/include/dispatch.h`<br>`lietorch/src/*` | lietorch 自己的 `DISPATCH_GROUP_AND_FLOATING_TYPES` 宏同样传 `.type()`，且宏体调用 `::detail::scalar_type()` —— 那是旧 ATen 放在全局 `detail` 命名空间的辅助函数，新版 torch 已移除。改宏体 + 38 处调用点 |
| **D** | `metric3d/mono/utils/comm.py` | 硬 `import mmcv`，但唯一使用点（`collect_env()`）在源码里整段被注释掉了，是个死导入。同文件里 `get_git_hash` 已有 try/except 回退 mmengine，作者漏了这一个 |
| **E** | `depth.py` | argparse 定义的是 `--images`（→`args.images`），调用时却写 `args.rgb`，readme 里的分步用法直接 `AttributeError`（`reconstruct.py` 直接调 `depth.main()` 所以不受影响） |
| **F** | `metric3d/mono/utils/transform.py` | `matplotlib.cm.get_cmap()` 在 matplotlib 3.9 已移除，存深度彩色图时报错。改用 `matplotlib.colormaps[]` |

## 相对上游 requirements.txt 的改动

上游 `droid_metric/requirements.txt` 是按 2023 年的栈写的，直接装在 5090 上跑不起来：

| 上游 | 本环境 | 原因 |
|---|---|---|
| `torch==2.0.1` | `2.8.0+cu128` | 2.0.1 没有 sm_120 kernel |
| `torchvision==0.15.2` | `0.23.0+cu128` | 跟 torch 配套 |
| `numpy==1.26.1` | `1.26.4` | 同系列，保持 <2 |
| `torch-scatter` | **源码编译** | `data.pyg.org` 的 `pt28cu128` 预编译轮子链接 GLIBC 2.32，本机 Ubuntu 20.04 是 2.31，装得上但一 import 就 `OSError` |
| `xformers==0.0.21` | **不装** | 它死锁 torch 2.0.1。Metric3D 里三处 `import xformers` 都在 `try/except ImportError` 内，`XFORMERS_AVAILABLE=False` 时自动回退标准 attention |
| `mmcv` | **不装** | 见补丁 D，推理路径只需要 `mmengine`（纯 Python） |
| `gdown` | `gdown<6` | gdown 6.x 删掉了 `download(fuzzy=...)`，而 `download_models.py` 用了它，装 6.x 下载权重直接 `TypeError` |

不装 xformers 的代价是 ViT attention 走标准实现，显存和速度略差，但 5090 有 32GB 显存够用，
且省掉了为新 torch 编译 xformers 的麻烦。

## 本机环境的坑

**`/usr/local/cuda` 指向的是 CUDA 11.8，不是 12.8。**
Blackwell 最低要求 CUDA 12.8，所有脚本都显式设 `CUDA_HOME=/usr/local/cuda-12.8`。

**`MAX_JOBS` 不能开太大。**
本机 30GB 内存，nvcc 编译 CUDA 模板每个 job 能吃好几个 GB。`MAX_JOBS=16` 会被 OOM killer
杀掉，现象是 ninja 报 `Killed` 而不是编译错误——很容易误判成代码不兼容。脚本固定 `MAX_JOBS=4`。

**gsplat 首次调用会 JIT 编译（约 6 分钟）。**
`TORCH_CUDA_ARCH_LIST` 和 `MAX_JOBS` 每次必须一致，否则 ninja 认为编译配置变了会全量重编。
两个 `env_*.sh` 都固定了这两个变量。

## 验证结果

**CUDA kernel 数值验证** —— 补丁 A/B/C 动的正是 kernel 分发路径，光能 import 不够，
`test_sm120_kernels.py` 把每个 kernel 跟纯 PyTorch 参考实现对拍，10/10 通过：

```
correlation_kernels.cu   corr_index_forward   vs grid_sample 参考，最大误差 1.6e-06
                         corr_index_backward  梯度有限且非零
altcorr_kernel.cu        altcorr_forward      输出有限非零
lietorch_gpu.cu          群运算恒等式          SO3/RxSO3/SE3/Sim3 均 <2e-15
                                              (exp/log 往返, X·X⁻¹=I, 结合律)
                         SE3 伴随恒等式        Adj(X)·a == log(X·exp(a)·X⁻¹), 误差 1.2e-16
droid_kernels.cu         iproj / projmap / frame_distance / depth_filter / ba  全部有效
```

**端到端 pipeline** —— `testdata/synth/` 是用 Open3D 光线投射生成的 30 帧序列
（640×480，相机基线 2.4m，有真实视差，附真值轨迹）：

| 步骤 | 结果 |
|---|---|
| `depth.py`（Metric3D giant2） | 30/30 帧，2.08 it/s |
| `slam.py`（DROID-SLAM） | 20 轮 global BA 收敛，输出 30 帧位姿 |
| `mesh.py`（TSDF 融合） | 15MB mesh.ply |

轨迹精度（Umeyama 对齐后）：

```
真值轨迹总长   2.631 m       估计轨迹总长  2.343 m
Sim(3) 对齐    ATE RMSE 5.13 cm   恢复尺度 1.19
SE(3)  对齐    ATE RMSE 12.93 cm  (锁定尺度=1)
```

2.6m 轨迹上 5cm 的 ATE 属于正常水平。恢复尺度 1.19 而非 1.0 是因为 Metric3D
在真实照片上训练，这个光追合成场景属于分布外——是测试场景的属性，不是安装问题。

## zipnerf

`nerfstudio_sm120` 里 zipnerf 是可用的，已实测训练+渲染。

它不是 pip 装的，而是 editable 安装指向 **`/mnt/m/algorithm/zipnerf-pytorch`**
（`__editable__.zipnerf-0.1.0.pth`），通过 nerfstudio 插件入口 `zipnerf_ns` 注册。
**依赖 `/mnt/m` 挂载**，否则 `ns-train` 启动就会失败。

它的 CUDA 扩展 `extensions/cuda/_cuda_backend.*.so` 是按 **sm_120** 编的
（`cuobjdump --list-elf` 确认），hash grid 编码的前向/反向在 5090 上都正常。
注意 `gridencoder` 没有独立的 `.so`，走的是同一个 `_cuda_backend`。

实测（30 帧合成序列，600 步）：

```
Train PSNR      13.54 -> 29.48        Train Loss  0.166 -> 0.018
Eval  PSNR/SSIM 27.39 / 0.892
速度            ~82 ms/iter, ~50K rays/s, 显存约 4 GB
```

```bash
source /mnt/g/algorithm_backup/Nevstereo/env_nerfstudio.sh
ns-train zipnerf --data <transforms.json 所在目录>
```

复跑验证：

```bash
cd /mnt/g/algorithm_backup/Nevstereo

bash install_nevstereo.sh verify              # 两个环境的组件校验
source env_droid.sh                           # 进 nevstereo 环境
python ../test_sm120_kernels.py               # CUDA kernel 数值验证
```
