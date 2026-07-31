#!/usr/bin/env bash
#==============================================================================
#  进入 nerfstudio 环境 (nerfstudio_sm120)  ——  RTX 5090 / sm_120
#
#  用法：  source /mnt/g/algorithm_backup/Nevstereo/env_nerfstudio.sh
#
#  注意：这个环境是既有的，安装脚本不会修改它。
#        torch 2.12.0.dev+cu128 / tinycudann 2.0 / nerfacc 0.5.2 / gsplat 1.4.0
#==============================================================================

export NEVSTEREO_ROOT="/mnt/g/algorithm_backup/Nevstereo"

# 关键：本机 /usr/local/cuda 指向 11.8，Blackwell 必须用 12.8
export CUDA_HOME="/usr/local/cuda-12.8"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# gsplat 首次调用会 JIT 编译 CUDA。这两个变量必须每次都一致，
# 否则 ninja 会认为编译配置变了而全量重编（而且 MAX_JOBS 高了会被 OOM killer 杀）。
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS="${MAX_JOBS:-4}"

source /home/pengcc/miniconda3/etc/profile.d/conda.sh
conda activate nerfstudio_sm120

cd "${NEVSTEREO_ROOT}"

cat <<EOF
─────────────────────────────────────────────────────────────
 环境: nerfstudio_sm120
 CUDA: ${CUDA_HOME}   ARCH: ${TORCH_CUDA_ARCH_LIST}

 数据预处理:  ns-process-data images --data <图片目录> --output-dir <输出目录>
 训练:        ns-train nerfacto   --data <处理后目录>      # tinycudann 路径
              ns-train splatfacto --data <处理后目录>      # gsplat 路径
 查看:        ns-viewer --load-config <outputs/.../config.yml>
 导出:        ns-export pointcloud --load-config <config.yml> --output-dir <目录>
─────────────────────────────────────────────────────────────
EOF
