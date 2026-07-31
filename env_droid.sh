#!/usr/bin/env bash
#==============================================================================
#  Enter the droid_metric environment (nevstereo) - RTX 5090 / sm_120
#
#  Usage: source /mnt/g/algorithm_backup/Nevstereo/env_droid.sh
#==============================================================================

export NEVSTEREO_ROOT="/mnt/g/algorithm_backup/Nevstereo"
export DROID_METRIC="${NEVSTEREO_ROOT}/droid_metric"

# Important: /usr/local/cuda points to 11.8 on this host; Blackwell requires 12.8.
export CUDA_HOME="/usr/local/cuda-12.8"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export TORCH_CUDA_ARCH_LIST="12.0"     # 5090
export MAX_JOBS="${MAX_JOBS:-4}"       # nvcc is memory-intensive; high concurrency triggers the OOM killer.

source /home/pengcc/miniconda3/etc/profile.d/conda.sh
conda activate nevstereo

cd "${DROID_METRIC}"

cat <<EOF
─────────────────────────────────────────────────────────────
 Environment: nevstereo   (droid_metric = DROID-SLAM + Metric3D)
 Directory: ${DROID_METRIC}
 CUDA: ${CUDA_HOME}   ARCH: ${TORCH_CUDA_ARCH_LIST}

 Complete reconstruction:
   python reconstruct.py --input <image-directory-or-video> --output <output-directory> --viz

 Individual stages:
   python depth.py  --images <image-directory> --out <depth-directory>
   python slam.py   --image <image-directory> --depth <depth-directory> --poses <pose-file>
   python mesh.py   --image <image-directory> --depth <depth-directory> --poses <pose-file> --mesh <mesh.ply>

 Without --intr, intrinsics follow the COLMAP convention: fx=fy=max(W,H)*1.2, cx=W/2, cy=H/2
─────────────────────────────────────────────────────────────
EOF
