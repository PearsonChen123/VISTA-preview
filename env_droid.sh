#!/usr/bin/env bash
#==============================================================================
#  进入 droid_metric 环境 (nevstereo)  ——  RTX 5090 / sm_120
#
#  用法：  source /mnt/g/algorithm_backup/Nevstereo/env_droid.sh
#==============================================================================

export NEVSTEREO_ROOT="/mnt/g/algorithm_backup/Nevstereo"
export DROID_METRIC="${NEVSTEREO_ROOT}/droid_metric"

# 关键：本机 /usr/local/cuda 指向 11.8，Blackwell 必须用 12.8
export CUDA_HOME="/usr/local/cuda-12.8"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export TORCH_CUDA_ARCH_LIST="12.0"     # 5090
export MAX_JOBS="${MAX_JOBS:-4}"       # nvcc 很吃内存，并发高会被 OOM killer 杀

source /home/pengcc/miniconda3/etc/profile.d/conda.sh
conda activate nevstereo

cd "${DROID_METRIC}"

cat <<EOF
─────────────────────────────────────────────────────────────
 环境: nevstereo   (droid_metric = DROID-SLAM + Metric3D)
 目录: ${DROID_METRIC}
 CUDA: ${CUDA_HOME}   ARCH: ${TORCH_CUDA_ARCH_LIST}

 一步到位重建:
   python reconstruct.py --input <图片目录或视频> --output <输出目录> --viz

 分步:
   python depth.py  --images <图片目录> --out <深度目录>
   python slam.py   --image <图片目录> --depth <深度目录> --poses <位姿文件>
   python mesh.py   --image <图片目录> --depth <深度目录> --poses <位姿文件> --mesh <mesh.ply>

 不给 --intr 时内参按 COLMAP 惯例估计: fx=fy=max(W,H)*1.2, cx=W/2, cy=H/2
─────────────────────────────────────────────────────────────
EOF
