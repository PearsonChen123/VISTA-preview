#!/usr/bin/env bash
#==============================================================================
#  Enter the nerfstudio environment (nerfstudio_sm120) - RTX 5090 / sm_120
#
#  Usage: source /mnt/g/algorithm_backup/Nevstereo/env_nerfstudio.sh
#
#  Note: This environment already exists; the installation script does not modify it.
#        torch 2.12.0.dev+cu128 / tinycudann 2.0 / nerfacc 0.5.2 / gsplat 1.4.0
#==============================================================================

export NEVSTEREO_ROOT="/mnt/g/algorithm_backup/Nevstereo"

# Important: /usr/local/cuda points to 11.8 on this host; Blackwell requires 12.8.
export CUDA_HOME="/usr/local/cuda-12.8"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# The first gsplat call JIT-compiles CUDA. These variables must stay consistent,
# or ninja performs a full rebuild (and a high MAX_JOBS value triggers the OOM killer).
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS="${MAX_JOBS:-4}"

source /home/pengcc/miniconda3/etc/profile.d/conda.sh
conda activate nerfstudio_sm120

cd "${NEVSTEREO_ROOT}"

cat <<EOF
─────────────────────────────────────────────────────────────
 Environment: nerfstudio_sm120
 CUDA: ${CUDA_HOME}   ARCH: ${TORCH_CUDA_ARCH_LIST}

 Preprocess data: ns-process-data images --data <image-directory> --output-dir <output-directory>
 Train:           ns-train nerfacto   --data <processed-directory>      # tinycudann path
                  ns-train splatfacto --data <processed-directory>      # gsplat path
 View:            ns-viewer --load-config <outputs/.../config.yml>
 Export:          ns-export pointcloud --load-config <config.yml> --output-dir <directory>
─────────────────────────────────────────────────────────────
EOF
