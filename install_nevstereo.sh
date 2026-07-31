#!/usr/bin/env bash
#==============================================================================
#  Nevstereo installer - RTX 5090 / Blackwell (sm_120) support
#==============================================================================
#
#  Two conda environments with separate responsibilities:
#
#    nerfstudio_sm120  -> nerfstudio (nerfacto / splatfacto)   [existing; only validated]
#    nevstereo         -> droid_metric (DROID-SLAM + Metric3D) [created by this script]
#
#  Typical workflow:
#    conda activate nevstereo        # 1. Run droid_metric to generate poses, depth, and a mesh
#    conda activate nerfstudio_sm120 # 2. Run nerfstudio to train a radiance field
#
#  Usage:
#    bash install_nevstereo.sh            # Run the complete workflow
#    bash install_nevstereo.sh check      # Run environment checks only
#    bash install_nevstereo.sh repo       # Clone/update droid_metric only
#    bash install_nevstereo.sh patch      # Apply sm_120 patches only
#    bash install_nevstereo.sh env        # Create the conda environment and install Python dependencies
#    bash install_nevstereo.sh build      # Build CUDA extensions (droid_backends / lietorch)
#    bash install_nevstereo.sh models     # Download pretrained weights (~7 GB)
#    bash install_nevstereo.sh verify     # Run post-installation validation only
#
#  This script is idempotent: completed stages are skipped on subsequent runs.
#==============================================================================

set -euo pipefail

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------
ROOT="/mnt/g/algorithm_backup/Nevstereo"        # All code, weights, and outputs live here.
REPO="${ROOT}/droid_metric"
CONDA_ROOT="/home/pengcc/miniconda3"

ENV_DROID="nevstereo"                            # droid_metric environment
ENV_NERF="nerfstudio_sm120"                      # Existing nerfstudio environment; do not modify.

PY_VER="3.11"
TORCH_VER="2.8.0"
TV_VER="0.23.0"
CU_TAG="cu128"

# Important: /usr/local/cuda points to 11.8 on this host; explicitly use 12.8 for Blackwell.
export CUDA_HOME="/usr/local/cuda-12.8"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# 5090 = compute capability 12.0
export TORCH_CUDA_ARCH_LIST="12.0"

# nvcc CUDA template compilation is memory-intensive. This host has 30 GB of RAM;
# excessive MAX_JOBS triggers the OOM killer, and ninja reports "Killed".
export MAX_JOBS="${MAX_JOBS:-4}"

PY="${CONDA_ROOT}/envs/${ENV_DROID}/bin/python"

#------------------------------------------------------------------------------
# Helpers
#------------------------------------------------------------------------------
c_info()  { printf '\033[1;34m[*]\033[0m %s\n' "$*"; }
c_ok()    { printf '\033[1;32m[✓]\033[0m %s\n' "$*"; }
c_warn()  { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }
c_err()   { printf '\033[1;31m[✗]\033[0m %s\n' "$*" >&2; }
c_step()  { printf '\n\033[1;36m===== %s =====\033[0m\n' "$*"; }

#------------------------------------------------------------------------------
# 1. Prerequisite checks
#------------------------------------------------------------------------------
do_check() {
    c_step "Prerequisite checks"

    if ! command -v nvidia-smi >/dev/null; then
        c_err "nvidia-smi not found"; return 1
    fi
    local gpu cc
    gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
    c_info "GPU: ${gpu} (compute capability ${cc})"
    if [[ "${cc}" != "12.0" ]]; then
        c_warn "This script targets sm_120 (5090/Blackwell); current capability is ${cc}"
        c_warn "Change TORCH_CUDA_ARCH_LIST to use another GPU"
    fi

    if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
        c_err "${CUDA_HOME}/bin/nvcc not found; Blackwell requires CUDA >= 12.8"; return 1
    fi
    c_info "nvcc: $(${CUDA_HOME}/bin/nvcc --version | grep -oE 'release [0-9.]+' | head -1)"

    if [[ ! -x "${CONDA_ROOT}/bin/conda" ]]; then
        c_err "conda not found: ${CONDA_ROOT}/bin/conda"; return 1
    fi

    c_info "gcc: $(gcc -dumpversion)  (CUDA 12.8 supports gcc <= 14)"
    c_info "Memory: $(free -g | awk '/^Mem:/{print $2}') GB, cores: $(nproc), MAX_JOBS=${MAX_JOBS}"
    c_ok "Prerequisite checks passed"
}

#------------------------------------------------------------------------------
# 2. Clone repository
#------------------------------------------------------------------------------
do_repo() {
    c_step "Clone droid_metric"
    mkdir -p "${ROOT}"
    if [[ -d "${REPO}/.git" ]]; then
        c_ok "Already exists: ${REPO}"
        git -C "${REPO}" submodule update --init --recursive
    else
        git clone --recursive https://github.com/Jianxff/droid_metric.git "${REPO}"
    fi
    [[ -f "${REPO}/modules/droid_slam/setup.py" ]] || { c_err "droid_slam submodule is missing"; return 1; }
    [[ -d "${REPO}/modules/metric3d/mono"     ]] || { c_err "metric3d submodule is missing";   return 1; }
    c_ok "Repository ready"
}

#------------------------------------------------------------------------------
# 3. Patches
#------------------------------------------------------------------------------
#  Patch A - sm_120 (required for the RTX 5090)
#    DROID-SLAM's setup.py hard-codes nvcc gencode only through sm_86.
#    A .so built this way has no sm_120 kernel image and fails on a 5090:
#       CUDA error: no kernel image is available for execution on the device
#    Replace both gencode lists (droid_backends / lietorch_backends) with compute_120/sm_120.
#
#  Patch B - AT_DISPATCH for newer torch versions (required to compile)
#    This 2022 DROID-SLAM code uses AT_DISPATCH_xxx(tensor.type(), ...).
#    In newer torch versions tensor.type() returns at::DeprecatedTypeProperties, while
#    AT_DISPATCH requires c10::ScalarType, producing this compilation error:
#       error: no suitable conversion function from "const at::DeprecatedTypeProperties"
#              to "c10::ScalarType" exists
#    Change it to tensor.scalar_type() at three sites (correlation_kernels.cu x2, altcorr_kernel.cu x1).
#
#  Patch C - lietorch's custom dispatch macro (required to compile)
#    lietorch's DISPATCH_GROUP_AND_FLOATING_TYPES macro also receives X.type().
#    Its body calls ::detail::scalar_type(), an old ATen helper in the global detail
#    namespace that newer torch versions removed.
#    Make the macro use ScalarType directly and replace .type() with .scalar_type()
#    at 38 call sites (19 in lietorch_gpu.cu and 19 in lietorch_cpu.cpp).
#------------------------------------------------------------------------------
do_patch() {
    c_step "Apply patches (sm_120 + newer torch API)"
    local setup="${REPO}/modules/droid_slam/setup.py"
    [[ -f "${setup}" ]] || { c_err "${setup} not found"; return 1; }

    python3 - "${setup}" <<'PYEOF'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1])
s = p.read_text()

if 'compute_120' in s:
    print('[=] setup.py is already patched; skipping')
    sys.exit(0)

bak = p.with_name(p.name + '.orig')
if not bak.exists():
    bak.write_text(s)
    print(f'[+] Backed up original file -> {bak.name}')

# Replace each contiguous gencode block with sm_120.
new, n = re.subn(
    r"(?:[ \t]*'-gencode=arch=compute_\d+,code=sm_\d+',?[ \t]*\n)+",
    "                    '-gencode=arch=compute_120,code=sm_120',\n",
    s)
assert n == 2, f'Expected 2 gencode blocks, found {n}'
p.write_text(new)
print(f'[+] Replaced {n} gencode blocks -> compute_120/sm_120')
PYEOF

    # Patch B: AT_DISPATCH_xxx(t.type(), ...) -> AT_DISPATCH_xxx(t.scalar_type(), ...)
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
        print(f'[+] {f.name}: {n} occurrences of .type() -> .scalar_type()')
        total += n
if total == 0:
    print('[=] AT_DISPATCH already uses scalar_type(); skipping')
PYEOF

    # Patch C: lietorch's DISPATCH_GROUP_AND_FLOATING_TYPES macro and 38 call sites.
    python3 - "${REPO}/modules/droid_slam/thirdparty/lietorch/lietorch" <<'PYEOF'
import re, sys, pathlib
root = pathlib.Path(sys.argv[1])

def backup_write(f, old, new):
    bak = f.with_name(f.name + '.orig')
    if not bak.exists():
        bak.write_text(old)
    f.write_text(new)

# Macro body: newer torch removed ::detail::scalar_type(); use the supplied ScalarType.
h = root / 'include' / 'dispatch.h'
s = h.read_text()
if '::detail::scalar_type' in s:
    new = s.replace('at::ScalarType _st = ::detail::scalar_type(the_type);',
                    'at::ScalarType _st = the_type;')
    assert new != s, 'Failed to replace the dispatch.h macro body'
    backup_write(h, s, new)
    print('[+] dispatch.h: removed ::detail::scalar_type() from macro body')
else:
    print('[=] dispatch.h is already patched; skipping')

# Call sites: DISPATCH_GROUP_AND_FLOATING_TYPES(g, X.type(), ...) -> X.scalar_type()
total = 0
for f in sorted(root.glob('src/*.cu')) + sorted(root.glob('src/*.cpp')):
    s = f.read_text()
    new, n = re.subn(r'(DISPATCH_GROUP_AND_FLOATING_TYPES\(\s*\w+\s*,\s*\w+)\.type\(\)',
                     r'\1.scalar_type()', s)
    if n:
        backup_write(f, s, new)
        print(f'[+] {f.name}: {n} occurrences of .type() -> .scalar_type()')
        total += n
if total == 0:
    print('[=] lietorch call sites already use scalar_type(); skipping')
PYEOF

    # Patch D: Metric3D's comm.py unconditionally imports mmcv.
    python3 - "${REPO}/modules/metric3d/mono/utils/comm.py" <<'PYEOF'
import sys, pathlib
f = pathlib.Path(sys.argv[1])
s = f.read_text()
old = '\nfrom mmcv.utils import collect_env as collect_base_env\n'   # No leading indentation.
# The test must include the leading newline. After patching, this statement is
# indented inside try; an unindented substring test would keep nesting try blocks.
if 'mmengine.utils.dl_utils import collect_env' in s or old not in s:
    print('[=] comm.py is already patched; skipping')
    sys.exit(0)
# Its only use in collect_env() is commented out, so this is a dead import.
# get_git_hash already falls back to mmengine; apply the same pattern here.
new = s.replace(old,
    '\ntry:\n'
    '    from mmcv.utils import collect_env as collect_base_env\n'
    'except ImportError:\n'
    '    from mmengine.utils.dl_utils import collect_env as collect_base_env\n')
bak = f.with_name(f.name + '.orig')
if not bak.exists():
    bak.write_text(s)
f.write_text(new)
print('[+] comm.py: made mmcv optional with an mmengine fallback')
PYEOF

    # Patch E: argparse defines --images (args.images), but the upstream call
    # uses args.rgb, so the documented staged command raises AttributeError.
    # reconstruct.py calls depth.main() directly and is unaffected.
    python3 - "${REPO}/depth.py" <<'PYEOF'
import sys, pathlib
f = pathlib.Path(sys.argv[1])
s = f.read_text()
if 'input_images=args.rgb' not in s:
    print('[=] depth.py is already patched; skipping')
    sys.exit(0)
bak = f.with_name(f.name + '.orig')
if not bak.exists():
    bak.write_text(s)
f.write_text(s.replace('input_images=args.rgb', 'input_images=args.images'))
print('[+] depth.py: args.rgb -> args.images (upstream bug)')
PYEOF

    # Patch F: matplotlib 3.9 removed matplotlib.cm.get_cmap().
    # Metric3D uses it to save colorized depth maps; matplotlib 3.11 raises
    # AttributeError. Use matplotlib.colormaps[], available since 3.5.
    python3 - "${REPO}/modules/metric3d/mono/utils/transform.py" <<'PYEOF'
import sys, pathlib
f = pathlib.Path(sys.argv[1])
s = f.read_text()
old = 'cmap_m = matplotlib.cm.get_cmap(cmap)'
if old not in s:
    print('[=] transform.py is already patched; skipping')
    sys.exit(0)
bak = f.with_name(f.name + '.orig')
if not bak.exists():
    bak.write_text(s)
f.write_text(s.replace(old, 'cmap_m = matplotlib.colormaps[cmap]'))
print('[+] transform.py: matplotlib.cm.get_cmap -> matplotlib.colormaps[]')
PYEOF

    c_ok "Patches applied"
}

#------------------------------------------------------------------------------
# 4. Conda environment and Python dependencies
#------------------------------------------------------------------------------
#  Changes from upstream requirements.txt, all needed for the RTX 5090:
#    torch 2.0.1  -> 2.8.0+cu128    The upstream version has no sm_120 kernel.
#    torchvision  -> 0.23.0+cu128   Matches torch.
#    numpy 1.26.1 -> 1.26.4         Stay below 2; old DROID-SLAM dislikes NumPy 2.
#    torch-scatter -> source build   The pt28cu128 wheel links against GLIBC 2.32,
#                                   but this Ubuntu 20.04 host has 2.31. It installs
#                                   but raises OSError on import, so build locally.
#    xformers 0.0.21 -> omitted      It pins torch 2.0.1. Metric3D's imports are
#                                   guarded and fall back to standard attention.
#    mmcv         -> omitted         Only used in ViT_DINO_reg.py's __main__ branch;
#                                   inference only needs pure-Python mmengine.
#------------------------------------------------------------------------------
do_env() {
    c_step "Create conda environment ${ENV_DROID}"

    if [[ -x "${PY}" ]]; then
        c_ok "Environment already exists: ${CONDA_ROOT}/envs/${ENV_DROID}"
    else
        "${CONDA_ROOT}/bin/conda" create -y -n "${ENV_DROID}" "python=${PY_VER}"
    fi
    c_info "python: $(${PY} --version 2>&1)"

    c_info "Install torch ${TORCH_VER}+${CU_TAG} / torchvision ${TV_VER}+${CU_TAG}"
    "${PY}" -m pip install -q --upgrade pip
    "${PY}" -m pip install "torch==${TORCH_VER}" "torchvision==${TV_VER}" \
        --index-url "https://download.pytorch.org/whl/${CU_TAG}"

    # gdown>=6 removed download(fuzzy=...), which download_models.py uses.
    # Version 6.x would make weight downloads fail with TypeError.
    c_info "Install remaining droid_metric / Metric3D dependencies"
    "${PY}" -m pip install \
        "numpy==1.26.4" opencv-python "gdown<6" py3_wget tqdm psutil \
        open3d tensorboard scipy matplotlib pyyaml evo \
        mmengine timm html4vision plyfile Pillow

    # setuptools >= 80 removed `setup.py install`, while droid_slam's setup.py has
    # two setup() calls. `pip install .` only runs the first and omits lietorch.
    c_info "Pin setuptools<80 (droid_slam needs setup.py install for two setup() calls)"
    "${PY}" -m pip install -q "setuptools<80" wheel ninja

    # The data.pyg.org wheel links against GLIBC 2.32, while this host has 2.31.
    # It installs but raises OSError on import, so build it locally.
    if "${PY}" -c 'import torch_scatter' 2>/dev/null; then
        c_ok "torch_scatter is available"
    else
        c_info "Build torch_scatter from source (sm_120)"
        "${PY}" -m pip uninstall -y torch_scatter >/dev/null 2>&1 || true
        FORCE_CUDA=1 "${PY}" -m pip install --no-build-isolation \
            --no-binary=torch_scatter torch_scatter
    fi

    c_ok "Python dependencies installed"
}

#------------------------------------------------------------------------------
# 5. Build CUDA extensions
#------------------------------------------------------------------------------
do_build() {
    c_step "Build droid_backends / lietorch (sm_120)"
    [[ -x "${PY}" ]] || { c_err "Environment ${ENV_DROID} does not exist; run 'env' first"; return 1; }

    grep -q 'compute_120' "${REPO}/modules/droid_slam/setup.py" \
        || { c_err "setup.py is not patched; run 'patch' first"; return 1; }

    c_info "CUDA_HOME=${CUDA_HOME}  ARCH=${TORCH_CUDA_ARCH_LIST}  MAX_JOBS=${MAX_JOBS}"
    ( cd "${REPO}/modules/droid_slam" && "${PY}" setup.py install )

    c_ok "CUDA extensions built"
}

#------------------------------------------------------------------------------
# 6. Pretrained weights
#------------------------------------------------------------------------------
do_models() {
    c_step "Download pretrained weights (~7 GB -> ${REPO}/weights)"
    [[ -x "${PY}" ]] || { c_err "Environment ${ENV_DROID} does not exist; run 'env' first"; return 1; }
    ( cd "${REPO}" && "${PY}" download_models.py )
    c_info "weights/:"; ls -lh "${REPO}/weights" 2>/dev/null || true
    c_ok "Weights ready"
}

#------------------------------------------------------------------------------
# 7. Validation
#------------------------------------------------------------------------------
do_verify() {
    c_step "Validate ${ENV_DROID} (droid_metric)"
    [[ -x "${PY}" ]] || { c_err "Environment ${ENV_DROID} does not exist"; return 1; }

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
assert 'sm_120' in torch.cuda.get_arch_list(), 'torch has no sm_120 kernel!'
print(f'  device: {torch.cuda.get_device_name(0)} {torch.cuda.get_device_capability(0)}')

def _matmul():
    a = torch.randn(2048, 2048, device=d); (a @ a).sum().item(); return 'sm_120 matrix multiplication OK'
chk('torch CUDA', _matmul)

def _lietorch():
    from lietorch import SO3
    p = SO3.Random(4, device=d)          # Trigger a lietorch_backends CUDA kernel.
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
    return 'Metric3D model code imports successfully (automatic fallback without xformers)'
chk('Metric3D', _metric3d)

sys.exit(0 if ok else 1)
PYEOF
    ) || { c_err "${ENV_DROID} did not pass every validation"; return 1; }
    c_ok "${ENV_DROID} validation passed"

    c_step "Validate ${ENV_NERF} (nerfstudio, read-only)"
    local pyn="${CONDA_ROOT}/envs/${ENV_NERF}/bin/python"
    if [[ ! -x "${pyn}" ]]; then
        c_warn "Environment ${ENV_NERF} does not exist; skipping"
        return 0
    fi
    "${pyn}" - <<'PYEOF'
import torch
from importlib.metadata import version as _v
print(f'  torch {torch.__version__} | arch_list includes sm_120: {"sm_120" in torch.cuda.get_arch_list()}')

def _tcnn():
    # Run HashGrid + FullyFusedMLP once to verify that the sm_120 kernel works.
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
    # The first call JIT-compiles CUDA (about six minutes); later calls use the cache.
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
    c_ok "${ENV_NERF} validation complete (no changes made)"
}

#------------------------------------------------------------------------------
# Main workflow
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
            c_step "All stages complete"
            cat <<EOF

  Code/weights/output ${ROOT}
  droid_metric     ${REPO}
  Conda environments  ${CONDA_ROOT}/envs/${ENV_DROID}   (droid_metric)
                      ${CONDA_ROOT}/envs/${ENV_NERF}    (nerfstudio, unchanged)

  Usage:
    source ${ROOT}/env_droid.sh        # Enter the droid_metric environment
    source ${ROOT}/env_nerfstudio.sh   # Enter the nerfstudio environment

EOF
            ;;
        *) c_err "Unknown argument: ${target}"
           grep -E '^#    bash install_nevstereo\.sh' "$0" | sed 's/^#/ /'
           exit 1 ;;
    esac
}

main "$@"
