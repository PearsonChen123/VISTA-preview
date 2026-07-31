#!/usr/bin/env bash
#==============================================================================
#  Nevstereo 安装脚本 —— RTX 5090 / Blackwell (sm_120) 适配
#==============================================================================
#
#  两个 conda 环境，各管一摊：
#
#    nerfstudio_sm120  →  nerfstudio (nerfacto / splatfacto)   [已存在，本脚本只校验，不修改]
#    nevstereo         →  droid_metric (DROID-SLAM + Metric3D) [本脚本创建]
#
#  典型流程：
#    conda activate nevstereo        # 1. 跑 droid_metric 出位姿/深度/mesh
#    conda activate nerfstudio_sm120 # 2. 跑 nerfstudio 训练辐射场
#
#  用法：
#    bash install_nevstereo.sh            # 全流程
#    bash install_nevstereo.sh check      # 只做环境预检
#    bash install_nevstereo.sh repo       # 只克隆/更新 droid_metric
#    bash install_nevstereo.sh patch      # 只打 sm_120 补丁
#    bash install_nevstereo.sh env        # 只建 conda 环境 + 装 Python 依赖
#    bash install_nevstereo.sh build      # 只编译 CUDA 扩展 (droid_backends / lietorch)
#    bash install_nevstereo.sh models     # 只下载预训练权重 (~7GB)
#    bash install_nevstereo.sh verify     # 只做安装后验证
#
#  脚本幂等：重复执行安全，已完成的步骤会跳过。
#==============================================================================

set -euo pipefail

#------------------------------------------------------------------------------
# 配置
#------------------------------------------------------------------------------
ROOT="/mnt/g/algorithm_backup/Nevstereo"        # 所有代码/权重/输出都在这里
REPO="${ROOT}/droid_metric"
CONDA_ROOT="/home/pengcc/miniconda3"

ENV_DROID="nevstereo"                            # droid_metric 环境
ENV_NERF="nerfstudio_sm120"                      # nerfstudio 环境（不动）

PY_VER="3.11"
TORCH_VER="2.8.0"
TV_VER="0.23.0"
CU_TAG="cu128"

# 关键：/usr/local/cuda 在本机指向 11.8，必须显式用 12.8（Blackwell 最低要求）
export CUDA_HOME="/usr/local/cuda-12.8"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# 5090 = compute capability 12.0
export TORCH_CUDA_ARCH_LIST="12.0"

# nvcc 编译 CUDA 模板极吃内存，本机 30GB；MAX_JOBS 太大会被 OOM killer 杀掉
# （表现为 ninja 报 "Killed" 而不是编译错误）
export MAX_JOBS="${MAX_JOBS:-4}"

PY="${CONDA_ROOT}/envs/${ENV_DROID}/bin/python"

#------------------------------------------------------------------------------
# 小工具
#------------------------------------------------------------------------------
c_info()  { printf '\033[1;34m[*]\033[0m %s\n' "$*"; }
c_ok()    { printf '\033[1;32m[✓]\033[0m %s\n' "$*"; }
c_warn()  { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }
c_err()   { printf '\033[1;31m[✗]\033[0m %s\n' "$*" >&2; }
c_step()  { printf '\n\033[1;36m===== %s =====\033[0m\n' "$*"; }

#------------------------------------------------------------------------------
# 1. 预检
#------------------------------------------------------------------------------
do_check() {
    c_step "预检"

    if ! command -v nvidia-smi >/dev/null; then
        c_err "找不到 nvidia-smi"; return 1
    fi
    local gpu cc
    gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
    c_info "GPU: ${gpu} (compute capability ${cc})"
    if [[ "${cc}" != "12.0" ]]; then
        c_warn "本脚本针对 sm_120 (5090/Blackwell)，当前是 ${cc}"
        c_warn "如需在别的卡上用，请改 TORCH_CUDA_ARCH_LIST"
    fi

    if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
        c_err "找不到 ${CUDA_HOME}/bin/nvcc —— Blackwell 需要 CUDA >= 12.8"; return 1
    fi
    c_info "nvcc: $(${CUDA_HOME}/bin/nvcc --version | grep -oE 'release [0-9.]+' | head -1)"

    if [[ ! -x "${CONDA_ROOT}/bin/conda" ]]; then
        c_err "找不到 conda: ${CONDA_ROOT}/bin/conda"; return 1
    fi

    c_info "gcc: $(gcc -dumpversion)  (CUDA 12.8 支持 gcc <= 14)"
    c_info "内存: $(free -g | awk '/^Mem:/{print $2}')GB, 核心: $(nproc), MAX_JOBS=${MAX_JOBS}"
    c_ok "预检通过"
}

#------------------------------------------------------------------------------
# 2. 克隆仓库
#------------------------------------------------------------------------------
do_repo() {
    c_step "克隆 droid_metric"
    mkdir -p "${ROOT}"
    if [[ -d "${REPO}/.git" ]]; then
        c_ok "已存在: ${REPO}"
        git -C "${REPO}" submodule update --init --recursive
    else
        git clone --recursive https://github.com/Jianxff/droid_metric.git "${REPO}"
    fi
    [[ -f "${REPO}/modules/droid_slam/setup.py" ]] || { c_err "子模块 droid_slam 缺失"; return 1; }
    [[ -d "${REPO}/modules/metric3d/mono"     ]] || { c_err "子模块 metric3d 缺失";   return 1; }
    c_ok "仓库就绪"
}

#------------------------------------------------------------------------------
# 3. 补丁
#------------------------------------------------------------------------------
#  补丁 A —— sm_120 (5090 必需)
#    DROID-SLAM 的 setup.py 把 nvcc gencode 硬编码到 sm_86 为止。
#    在 5090 上编出来的 .so 没有 sm_120 的 kernel image，运行时会报：
#       CUDA error: no kernel image is available for execution on the device
#    把两处 gencode 列表（droid_backends / lietorch_backends）都换成 compute_120/sm_120。
#
#  补丁 B —— 新版 torch 的 AT_DISPATCH (编译必需)
#    这份 DROID-SLAM 是 2022 年的代码，用的是 AT_DISPATCH_xxx(tensor.type(), ...)。
#    新版 torch 里 tensor.type() 返回 at::DeprecatedTypeProperties，而 AT_DISPATCH
#    要求 c10::ScalarType，编译直接报：
#       error: no suitable conversion function from "const at::DeprecatedTypeProperties"
#              to "c10::ScalarType" exists
#    改成 tensor.scalar_type()。共 3 处（correlation_kernels.cu ×2, altcorr_kernel.cu ×1）。
#
#  补丁 C —— lietorch 的自定义 dispatch 宏 (编译必需)
#    lietorch 有自己的 DISPATCH_GROUP_AND_FLOATING_TYPES 宏，同样传 X.type()，
#    而且宏体里调用 ::detail::scalar_type() —— 那是旧 ATen 放在全局 detail 命名空间
#    的辅助函数，新版 torch 已经没有了。
#    这里把宏体改成直接取 ScalarType，并把 38 处调用点（lietorch_gpu.cu 19 +
#    lietorch_cpu.cpp 19）的 .type() 换成 .scalar_type()。
#------------------------------------------------------------------------------
do_patch() {
    c_step "打补丁 (sm_120 + 新版 torch API)"
    local setup="${REPO}/modules/droid_slam/setup.py"
    [[ -f "${setup}" ]] || { c_err "找不到 ${setup}"; return 1; }

    python3 - "${setup}" <<'PYEOF'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1])
s = p.read_text()

if 'compute_120' in s:
    print('[=] setup.py 已打过补丁，跳过')
    sys.exit(0)

bak = p.with_name(p.name + '.orig')
if not bak.exists():
    bak.write_text(s)
    print(f'[+] 备份原文件 -> {bak.name}')

# 把连续的 gencode 行整体替换成 sm_120
new, n = re.subn(
    r"(?:[ \t]*'-gencode=arch=compute_\d+,code=sm_\d+',?[ \t]*\n)+",
    "                    '-gencode=arch=compute_120,code=sm_120',\n",
    s)
assert n == 2, f'期望替换 2 处 gencode 块，实际 {n} 处'
p.write_text(new)
print(f'[+] 已替换 {n} 处 gencode 块 -> compute_120/sm_120')
PYEOF

    # 补丁 B: AT_DISPATCH_xxx(t.type(), ...) -> AT_DISPATCH_xxx(t.scalar_type(), ...)
    python3 - "${REPO}/modules/droid_slam/src" <<'PYEOF'
import re, sys, pathlib
src = pathlib.Path(sys.argv[1])
total = 0
for f in sorted(list(src.glob('*.cu')) + list(src.glob('*.cpp'))):
    s = f.read_text()
    new, n = re.subn(r'(AT_DISPATCH_[A-Z_]*\(\s*\w+)\.type\(\)', r'\1.scalar_type()', s)
    if n:
        bak = f.with_name(f.name + '.orig')
        if not bak.exists():
            bak.write_text(s)
        f.write_text(new)
        print(f'[+] {f.name}: {n} 处 .type() -> .scalar_type()')
        total += n
if total == 0:
    print('[=] AT_DISPATCH 已是 scalar_type()，跳过')
PYEOF

    # 补丁 C: lietorch 的 DISPATCH_GROUP_AND_FLOATING_TYPES 宏 + 38 处调用点
    python3 - "${REPO}/modules/droid_slam/thirdparty/lietorch/lietorch" <<'PYEOF'
import re, sys, pathlib
root = pathlib.Path(sys.argv[1])

def backup_write(f, old, new):
    bak = f.with_name(f.name + '.orig')
    if not bak.exists():
        bak.write_text(old)
    f.write_text(new)

# 宏体：::detail::scalar_type() 在新版 torch 已不存在，直接用传进来的 ScalarType
h = root / 'include' / 'dispatch.h'
s = h.read_text()
if '::detail::scalar_type' in s:
    new = s.replace('at::ScalarType _st = ::detail::scalar_type(the_type);',
                    'at::ScalarType _st = the_type;')
    assert new != s, 'dispatch.h 宏体替换失败'
    backup_write(h, s, new)
    print('[+] dispatch.h: 宏体去掉 ::detail::scalar_type()')
else:
    print('[=] dispatch.h 已打过补丁，跳过')

# 调用点：DISPATCH_GROUP_AND_FLOATING_TYPES(g, X.type(), ...) -> X.scalar_type()
total = 0
for f in sorted(root.glob('src/*.cu')) + sorted(root.glob('src/*.cpp')):
    s = f.read_text()
    new, n = re.subn(r'(DISPATCH_GROUP_AND_FLOATING_TYPES\(\s*\w+\s*,\s*\w+)\.type\(\)',
                     r'\1.scalar_type()', s)
    if n:
        backup_write(f, s, new)
        print(f'[+] {f.name}: {n} 处 .type() -> .scalar_type()')
        total += n
if total == 0:
    print('[=] lietorch 调用点已是 scalar_type()，跳过')
PYEOF

    # 补丁 D: Metric3D 的 comm.py 硬 import mmcv
    python3 - "${REPO}/modules/metric3d/mono/utils/comm.py" <<'PYEOF'
import sys, pathlib
f = pathlib.Path(sys.argv[1])
s = f.read_text()
old = '\nfrom mmcv.utils import collect_env as collect_base_env\n'   # 行首无缩进
# 注意：判据必须带行首换行。打完补丁后这句会缩进到 try 里，若用不带缩进的子串
# 判断会一直匹配成功，导致重复打补丁、把 try 嵌套坏掉。
if 'mmengine.utils.dl_utils import collect_env' in s or old not in s:
    print('[=] comm.py 已打过补丁，跳过')
    sys.exit(0)
# 唯一使用点（collect_env()）在源码里整段被注释掉了，是个死导入。
# 同文件里 get_git_hash 已经有 try/except 回退 mmengine，作者漏了这一个，照着补齐。
new = s.replace(old,
    '\ntry:\n'
    '    from mmcv.utils import collect_env as collect_base_env\n'
    'except ImportError:\n'
    '    from mmengine.utils.dl_utils import collect_env as collect_base_env\n')
bak = f.with_name(f.name + '.orig')
if not bak.exists():
    bak.write_text(s)
f.write_text(new)
print('[+] comm.py: mmcv 导入改为可选（回退 mmengine）')
PYEOF

    # 补丁 E: depth.py 的上游 bug —— argparse 定义的是 --images (args.images)，
    # 但调用时写成 args.rgb，导致 readme 里的分步用法直接 AttributeError。
    # （reconstruct.py 是直接调 depth.main()，所以一步到位的用法不受影响。）
    python3 - "${REPO}/depth.py" <<'PYEOF'
import sys, pathlib
f = pathlib.Path(sys.argv[1])
s = f.read_text()
if 'input_images=args.rgb' not in s:
    print('[=] depth.py 已打过补丁，跳过')
    sys.exit(0)
bak = f.with_name(f.name + '.orig')
if not bak.exists():
    bak.write_text(s)
f.write_text(s.replace('input_images=args.rgb', 'input_images=args.images'))
print('[+] depth.py: args.rgb -> args.images (上游 bug)')
PYEOF

    # 补丁 F: matplotlib 3.9 移除了 matplotlib.cm.get_cmap()。
    # Metric3D 存深度彩色图时用到它（depth.py --out-colormap / reconstruct.py），
    # 本机 matplotlib 3.11 会 AttributeError。换成 3.5+ 就有的 matplotlib.colormaps[]。
    python3 - "${REPO}/modules/metric3d/mono/utils/transform.py" <<'PYEOF'
import sys, pathlib
f = pathlib.Path(sys.argv[1])
s = f.read_text()
old = 'cmap_m = matplotlib.cm.get_cmap(cmap)'
if old not in s:
    print('[=] transform.py 已打过补丁，跳过')
    sys.exit(0)
bak = f.with_name(f.name + '.orig')
if not bak.exists():
    bak.write_text(s)
f.write_text(s.replace(old, 'cmap_m = matplotlib.colormaps[cmap]'))
print('[+] transform.py: matplotlib.cm.get_cmap -> matplotlib.colormaps[]')
PYEOF

    c_ok "补丁完成"
}

#------------------------------------------------------------------------------
# 4. conda 环境 + Python 依赖
#------------------------------------------------------------------------------
#  依赖相对上游 requirements.txt 的改动，全部是为了 5090：
#    torch 2.0.1  -> 2.8.0+cu128    上游版本没有 sm_120 kernel
#    torchvision  -> 0.23.0+cu128   跟 torch 配套
#    numpy 1.26.1 -> 1.26.4         保持 <2（DROID-SLAM 老代码对 numpy 2 不友好）
#    torch-scatter -> 源码编译       data.pyg.org 的 pt28cu128 预编译轮子链接的是
#                                   GLIBC 2.32，本机 Ubuntu 20.04 只有 2.31，装上
#                                   能装、一 import 就 OSError。必须本地编。
#    xformers 0.0.21 -> 不装        它死锁 torch 2.0.1；Metric3D 里三处 import 都在
#                                   try/except ImportError 内，XFORMERS_AVAILABLE=False
#                                   时自动回退标准 attention
#    mmcv         -> 不装           只在 ViT_DINO_reg.py 的 __main__ 分支里用到，
#                                   实际推理路径只需要 mmengine（纯 Python）
#------------------------------------------------------------------------------
do_env() {
    c_step "创建 conda 环境 ${ENV_DROID}"

    if [[ -x "${PY}" ]]; then
        c_ok "环境已存在: ${CONDA_ROOT}/envs/${ENV_DROID}"
    else
        "${CONDA_ROOT}/bin/conda" create -y -n "${ENV_DROID}" "python=${PY_VER}"
    fi
    c_info "python: $(${PY} --version 2>&1)"

    c_info "装 torch ${TORCH_VER}+${CU_TAG} / torchvision ${TV_VER}+${CU_TAG}"
    "${PY}" -m pip install -q --upgrade pip
    "${PY}" -m pip install "torch==${TORCH_VER}" "torchvision==${TV_VER}" \
        --index-url "https://download.pytorch.org/whl/${CU_TAG}"

    # gdown>=6 删掉了 download(fuzzy=...) 参数，而 download_models.py 用了它，
    # 装 6.x 的话下载权重会直接 TypeError。
    c_info "装 droid_metric / Metric3D 其余依赖"
    "${PY}" -m pip install \
        "numpy==1.26.4" opencv-python "gdown<6" py3_wget tqdm psutil \
        open3d tensorboard scipy matplotlib pyyaml evo \
        mmengine timm html4vision plyfile Pillow

    # setuptools >= 80 移除了 `setup.py install`，而 droid_slam 的 setup.py 里
    # 有两个 setup() 调用（droid_backends + lietorch）。pip install . 只会执行
    # 第一个，lietorch 会漏装，所以必须保留 setup.py install 的能力。
    c_info "钉 setuptools<80（droid_slam 需要 setup.py install 跑两个 setup()）"
    "${PY}" -m pip install -q "setuptools<80" wheel ninja

    # data.pyg.org 的预编译轮子链接 GLIBC 2.32，本机 Ubuntu 20.04 是 2.31，
    # 装得上但 import 就 OSError，所以本地编。
    if "${PY}" -c 'import torch_scatter' 2>/dev/null; then
        c_ok "torch_scatter 已可用"
    else
        c_info "从源码编译 torch_scatter (sm_120)"
        "${PY}" -m pip uninstall -y torch_scatter >/dev/null 2>&1 || true
        FORCE_CUDA=1 "${PY}" -m pip install --no-build-isolation \
            --no-binary=torch_scatter torch_scatter
    fi

    c_ok "Python 依赖装完"
}

#------------------------------------------------------------------------------
# 5. 编译 CUDA 扩展
#------------------------------------------------------------------------------
do_build() {
    c_step "编译 droid_backends / lietorch (sm_120)"
    [[ -x "${PY}" ]] || { c_err "环境 ${ENV_DROID} 不存在，先跑 'env'"; return 1; }

    grep -q 'compute_120' "${REPO}/modules/droid_slam/setup.py" \
        || { c_err "setup.py 没打补丁，先跑 'patch'"; return 1; }

    c_info "CUDA_HOME=${CUDA_HOME}  ARCH=${TORCH_CUDA_ARCH_LIST}  MAX_JOBS=${MAX_JOBS}"
    ( cd "${REPO}/modules/droid_slam" && "${PY}" setup.py install )

    c_ok "CUDA 扩展编译完成"
}

#------------------------------------------------------------------------------
# 6. 预训练权重
#------------------------------------------------------------------------------
do_models() {
    c_step "下载预训练权重 (~7GB -> ${REPO}/weights)"
    [[ -x "${PY}" ]] || { c_err "环境 ${ENV_DROID} 不存在，先跑 'env'"; return 1; }
    ( cd "${REPO}" && "${PY}" download_models.py )
    c_info "weights/:"; ls -lh "${REPO}/weights" 2>/dev/null || true
    c_ok "权重就绪"
}

#------------------------------------------------------------------------------
# 7. 验证
#------------------------------------------------------------------------------
do_verify() {
    c_step "验证 ${ENV_DROID} (droid_metric)"
    [[ -x "${PY}" ]] || { c_err "环境 ${ENV_DROID} 不存在"; return 1; }

    ( cd "${REPO}" && "${PY}" - <<'PYEOF'
import sys, torch
ok = True
def chk(name, fn):
    global ok
    try:
        r = fn(); print(f'  [\033[1;32m✓\033[0m] {name}: {r}')
    except Exception as e:
        ok = False; print(f'  [\033[1;31m✗\033[0m] {name}: {type(e).__name__}: {str(e)[:200]}')

d = torch.device('cuda')
print(f'  torch {torch.__version__} | cuda {torch.version.cuda}')
print(f'  arch_list: {torch.cuda.get_arch_list()}')
assert 'sm_120' in torch.cuda.get_arch_list(), 'torch 里没有 sm_120 kernel！'
print(f'  device: {torch.cuda.get_device_name(0)} {torch.cuda.get_device_capability(0)}')

def _matmul():
    a = torch.randn(2048, 2048, device=d); (a @ a).sum().item(); return 'sm_120 矩阵乘 OK'
chk('torch CUDA', _matmul)

def _lietorch():
    from lietorch import SO3
    p = SO3.Random(4, device=d)          # 触发 lietorch_backends 的 CUDA kernel
    return f'SO3.Random -> {tuple(p.shape)}'
chk('lietorch (CUDA)', _lietorch)

def _droid():
    import droid_backends
    return f'droid_backends @ {droid_backends.__file__.split("/")[-1]}'
chk('droid_backends', _droid)

def _scatter():
    from torch_scatter import scatter_mean
    src = torch.randn(16, 8, device=d); idx = torch.randint(0, 4, (16,), device=d)
    return f'scatter_mean -> {tuple(scatter_mean(src, idx, dim=0).shape)}'
chk('torch_scatter (CUDA)', _scatter)

chk('open3d',   lambda: __import__('open3d').__version__)
chk('mmengine', lambda: __import__('mmengine').__version__)

def _metric3d():
    from modules.metric3d.mono.model.monodepth_model import get_configured_monodepth_model
    return 'Metric3D 模型代码可导入 (xformers 缺失时自动回退)'
chk('Metric3D', _metric3d)

sys.exit(0 if ok else 1)
PYEOF
    ) || { c_err "${ENV_DROID} 验证未全部通过"; return 1; }
    c_ok "${ENV_DROID} 验证通过"

    c_step "验证 ${ENV_NERF} (nerfstudio，只读校验)"
    local pyn="${CONDA_ROOT}/envs/${ENV_NERF}/bin/python"
    if [[ ! -x "${pyn}" ]]; then
        c_warn "环境 ${ENV_NERF} 不存在，跳过"
        return 0
    fi
    "${pyn}" - <<'PYEOF'
import torch
from importlib.metadata import version as _v
print(f'  torch {torch.__version__} | arch_list 含 sm_120: {"sm_120" in torch.cuda.get_arch_list()}')

def _tcnn():
    # 实际跑一次 HashGrid + FullyFusedMLP，确认 sm_120 kernel 可用
    import tinycudann as tcnn
    d = torch.device('cuda')
    enc = tcnn.Encoding(3, {"otype": "HashGrid", "n_levels": 16, "n_features_per_level": 2,
                            "log2_hashmap_size": 19, "base_resolution": 16,
                            "per_level_scale": 1.4472}).to(d)
    net = tcnn.Network(enc.n_output_dims, 4, {"otype": "FullyFusedMLP", "activation": "ReLU",
                                              "output_activation": "None", "n_neurons": 64,
                                              "n_hidden_layers": 2}).to(d)
    y = net(enc(torch.rand(4096, 3, device=d))); torch.cuda.synchronize()
    return f'{_v("tinycudann")} | HashGrid+FullyFusedMLP -> {tuple(y.shape)}'

def _gsplat():
    # 首次调用会 JIT 编译 CUDA（约 6 分钟），之后走缓存
    from gsplat import rasterization
    d = torch.device('cuda'); N = 256
    img, _, _ = rasterization(
        torch.randn(N, 3, device=d), torch.randn(N, 4, device=d),
        torch.rand(N, 3, device=d) * 0.1, torch.rand(N, device=d),
        torch.rand(N, 3, device=d), torch.eye(4, device=d)[None],
        torch.tensor([[[300., 0, 150], [0, 300., 100], [0, 0, 1]]], device=d), 300, 200)
    torch.cuda.synchronize()
    return f'{_v("gsplat")} | rasterization -> {tuple(img.shape)}'

for name, fn in [
    ('tinycudann', _tcnn),
    ('nerfacc',    lambda: _v('nerfacc')),
    ('nerfstudio', lambda: _v('nerfstudio')),
    ('gsplat',     _gsplat),
]:
    try:    print(f'  [\033[1;32m✓\033[0m] {name}: {fn()}')
    except Exception as e: print(f'  [\033[1;31m✗\033[0m] {name}: {type(e).__name__}: {str(e)[:120]}')
PYEOF
    c_ok "${ENV_NERF} 校验完成（未做任何修改）"
}

#------------------------------------------------------------------------------
# 主流程
#------------------------------------------------------------------------------
main() {
    local target="${1:-all}"
    case "${target}" in
        check)  do_check ;;
        repo)   do_repo ;;
        patch)  do_patch ;;
        env)    do_env ;;
        build)  do_build ;;
        models) do_models ;;
        verify) do_verify ;;
        all)
            do_check
            do_repo
            do_patch
            do_env
            do_build
            do_models
            do_verify
            c_step "全部完成"
            cat <<EOF

  代码/权重/输出   ${ROOT}
  droid_metric     ${REPO}
  conda 环境       ${CONDA_ROOT}/envs/${ENV_DROID}   (droid_metric)
                   ${CONDA_ROOT}/envs/${ENV_NERF}    (nerfstudio，未改动)

  用法：
    source ${ROOT}/env_droid.sh        # 进 droid_metric 环境
    source ${ROOT}/env_nerfstudio.sh   # 进 nerfstudio 环境

EOF
            ;;
        *) c_err "未知参数: ${target}"
           grep -E '^#    bash install_nevstereo\.sh' "$0" | sed 's/^#/ /'
           exit 1 ;;
    esac
}

main "$@"
