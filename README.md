# Nevstereo - RTX 5090 (Blackwell / sm_120) Environment

Two conda environments have separate responsibilities and do not interfere with each other:

| Environment | Purpose | Location |
|---|---|---|
| `nerfstudio_sm120` | nerfstudio (nerfacto / splatfacto) | `~/miniconda3/envs/nerfstudio_sm120` |
| `nevstereo` | droid_metric (DROID-SLAM + Metric3D) | `~/miniconda3/envs/nevstereo` |

All code, weights, and outputs are under `/mnt/g/algorithm_backup/Nevstereo/`.

```
Nevstereo/
├── install_nevstereo.sh      Idempotent installation script
├── env_droid.sh              Source to enter the nevstereo environment
├── env_nerfstudio.sh         Source to enter the nerfstudio_sm120 environment
├── test_sm120_kernels.py     CUDA kernel numerical validation
├── testdata/synth/           Synthetic test sequence (30 frames with ground-truth trajectory)
└── droid_metric/
    ├── weights/              Pretrained weights (6.9 GB)
    └── modules/
        ├── droid_slam/       Submodule containing CUDA extension sources
        └── metric3d/         Submodule
```

## Usage

```bash
# 1. droid_metric: generate depth, poses, and a mesh
source /mnt/g/algorithm_backup/Nevstereo/env_droid.sh
python reconstruct.py --input <image-directory-or-video> --output <output-directory> --viz

# 2. nerfstudio: train a radiance field
source /mnt/g/algorithm_backup/Nevstereo/env_nerfstudio.sh
ns-train nerfacto --data <processed-directory>
```

The installation script can run individual stages with `bash install_nevstereo.sh {check|repo|patch|env|build|models|verify}`.

## Versions

**nevstereo** (created by this project)

| Component | Version |
|---|---|
| Python | 3.11 |
| torch | 2.8.0+cu128 (`arch_list` includes `sm_120`) |
| torchvision | 0.23.0+cu128 |
| torch_scatter | 2.1.2 (built from source) |
| numpy | 1.26.4 |
| droid_backends / lietorch | Locally built for sm_120 |

**nerfstudio_sm120** (pre-existing environment; the installation script only validates it)

| Component | Version |
|---|---|
| Python | 3.10 |
| torch | 2.12.0.dev+cu128 (`arch_list` includes `sm_120`) |
| tinycudann | 2.0 |
| nerfacc | 0.5.2 |
| gsplat | 1.4.0 |
| nerfstudio | 1.1.5 |

## Applied Patches

The installation script's `patch` stage makes six idempotent changes and backs up each original file as `*.orig`.
The first three are required for the RTX 5090 and newer torch versions; the last three fix upstream bugs.

| | File | Problem |
|---|---|---|
| **A** | `droid_slam/setup.py` | nvcc gencode is hard-coded to `sm_86`. The resulting `.so` has no sm_120 kernel image and fails on a 5090 with `no kernel image is available for execution on the device`. Both `droid_backends` and `lietorch_backends` are changed to `compute_120/sm_120`. |
| **B** | `droid_slam/src/*.cu` | The 2022 code uses `AT_DISPATCH_xxx(tensor.type(), ...)`. In newer torch versions, `.type()` returns `at::DeprecatedTypeProperties`, while `AT_DISPATCH` requires `c10::ScalarType`, so compilation fails. Three calls are changed to `.scalar_type()`. |
| **C** | `lietorch/include/dispatch.h`<br>`lietorch/src/*` | lietorch's `DISPATCH_GROUP_AND_FLOATING_TYPES` macro also receives `.type()`, and its body calls `::detail::scalar_type()`, a helper that old ATen placed in the global `detail` namespace but newer torch versions removed. The macro body and 38 call sites are updated. |
| **D** | `metric3d/mono/utils/comm.py` | The file unconditionally imports `mmcv`, but its only use in `collect_env()` is commented out. This is a dead import. The same file already makes `get_git_hash` fall back to mmengine with try/except; this import was overlooked upstream. |
| **E** | `depth.py` | argparse defines `--images` as `args.images`, but the call site uses `args.rgb`. The documented staged command fails with `AttributeError`; `reconstruct.py` is unaffected because it calls `depth.main()` directly. |
| **F** | `metric3d/mono/utils/transform.py` | matplotlib 3.9 removed `matplotlib.cm.get_cmap()`, causing an error while saving colorized depth maps. It is replaced with `matplotlib.colormaps[]`. |

## Changes From Upstream requirements.txt

The upstream `droid_metric/requirements.txt` targets a 2023 software stack and does not run directly on an RTX 5090:

| Upstream | This environment | Reason |
|---|---|---|
| `torch==2.0.1` | `2.8.0+cu128` | torch 2.0.1 has no sm_120 kernel. |
| `torchvision==0.15.2` | `0.23.0+cu128` | Matches the torch version. |
| `numpy==1.26.1` | `1.26.4` | Stays in the same series and below 2. |
| `torch-scatter` | **Built from source** | The `pt28cu128` wheel from `data.pyg.org` links against GLIBC 2.32, while this Ubuntu 20.04 host has 2.31. Installation succeeds, but importing it raises `OSError`. |
| `xformers==0.0.21` | **Not installed** | It pins torch 2.0.1. All three `import xformers` statements in Metric3D are inside `try/except ImportError`; when `XFORMERS_AVAILABLE=False`, the code falls back to standard attention. |
| `mmcv` | **Not installed** | See patch D. The inference path only needs `mmengine`, which is pure Python. |
| `gdown` | `gdown<6` | gdown 6.x removed `download(fuzzy=...)`, which `download_models.py` uses. Installing 6.x makes weight downloads fail with `TypeError`. |

Without xformers, ViT attention uses the standard implementation, with slightly higher memory use and lower speed.
The RTX 5090 has enough 32 GB memory, and this avoids building xformers for a newer torch version.

## Host-Specific Caveats

**`/usr/local/cuda` points to CUDA 11.8, not 12.8.**
Blackwell requires CUDA 12.8 or newer, so every script explicitly sets `CUDA_HOME=/usr/local/cuda-12.8`.

**Do not set `MAX_JOBS` too high.**
This host has 30 GB of RAM, and each nvcc CUDA template compilation job can consume several GB. With `MAX_JOBS=16`, the OOM killer terminates jobs and ninja reports `Killed` rather than a compilation error, which can look like incompatible code. The scripts set `MAX_JOBS=4`.

**The first gsplat call performs a JIT compilation that takes about six minutes.**
`TORCH_CUDA_ARCH_LIST` and `MAX_JOBS` must remain consistent between invocations, or ninja considers the build configuration changed and performs a complete rebuild. Both `env_*.sh` files fix these variables.

## Validation Results

**CUDA kernel numerical validation:** Patches A, B, and C alter kernel dispatch paths, so importing the modules is not enough.
`test_sm120_kernels.py` compares every kernel against a pure PyTorch reference implementation; all 10 tests pass:

```
correlation_kernels.cu   corr_index_forward   vs grid_sample reference, max error 1.6e-06
                         corr_index_backward  finite, nonzero gradients
altcorr_kernel.cu        altcorr_forward      finite, nonzero output
lietorch_gpu.cu          group identities     SO3/RxSO3/SE3/Sim3 all <2e-15
                                              (exp/log round trip, X*X^-1=I, associativity)
                         SE3 adjoint identity  Adj(X)*a == log(X*exp(a)*X^-1), error 1.2e-16
droid_kernels.cu         iproj / projmap / frame_distance / depth_filter / ba all valid
```

**End-to-end pipeline:** `testdata/synth/` is a 30-frame sequence generated with Open3D ray casting
(640x480, 2.4 m camera baseline, real disparity, and a ground-truth trajectory):

| Stage | Result |
|---|---|
| `depth.py` (Metric3D giant2) | 30/30 frames at 2.08 it/s |
| `slam.py` (DROID-SLAM) | 20 rounds of global BA converged; output 30 frame poses |
| `mesh.py` (TSDF fusion) | 15 MB mesh.ply |

Trajectory accuracy after Umeyama alignment:

```
Ground-truth path length  2.631 m       Estimated path length  2.343 m
Sim(3) alignment          ATE RMSE 5.13 cm   Recovered scale 1.19
SE(3) alignment           ATE RMSE 12.93 cm  (scale fixed at 1)
```

An ATE of 5 cm over a 2.6 m trajectory is normal. The recovered scale is 1.19 rather than 1.0 because Metric3D is trained on real photographs and this ray-traced scene is out of distribution. This is a property of the test scene, not an installation problem.

## zipnerf

zipnerf is available in `nerfstudio_sm120` and has been tested for both training and rendering.

It is not installed from pip. Its editable installation points to **`/mnt/m/algorithm/zipnerf-pytorch`**
through `__editable__.zipnerf-0.1.0.pth`, and it is registered through the `zipnerf_ns` nerfstudio plugin entry point.
**It requires `/mnt/m` to be mounted**, or `ns-train` will fail at startup.

Its CUDA extension, `extensions/cuda/_cuda_backend.*.so`, is built for **sm_120**, as verified with `cuobjdump --list-elf`.
Both forward and backward hash-grid encoding work on the RTX 5090. `gridencoder` has no separate `.so`; it uses the same `_cuda_backend`.

Measured on the 30-frame synthetic sequence for 600 steps:

```
Train PSNR      13.54 -> 29.48        Train Loss  0.166 -> 0.018
Eval  PSNR/SSIM 27.39 / 0.892
Speed           ~82 ms/iter, ~50K rays/s, about 4 GB VRAM
```

```bash
source /mnt/g/algorithm_backup/Nevstereo/env_nerfstudio.sh
ns-train zipnerf --data <directory-containing-transforms.json>
```

Run validation again:

```bash
cd /mnt/g/algorithm_backup/Nevstereo

bash install_nevstereo.sh verify              # Validate components in both environments
source env_droid.sh                           # Enter the nevstereo environment
python ../test_sm120_kernels.py               # Validate CUDA kernel numerics
```
