#!/usr/bin/env bash
#==============================================================================
#  stereo-depth pipeline
#
#  Run from raw images to filtered depth maps with one command:
#
#      raw images
#          ├─ colmap     SfM + undistortion
#          ├─ transforms undistorted model -> transforms.json
#          ├─ train      train a nerfstudio model
#          ├─ export     export training-view camera poses
#          ├─ shift      create right-camera poses
#          ├─ campath    convert to nerfstudio camera_path
#          ├─ render     render left and right views
#          ├─ rotate     rotate for stereo matching
#          ├─ intrinsic  generate K.txt with real-scale baseline
#          ├─ stereo     FoundationStereo inference
#          ├─ depth      restore depth orientation and save 16-bit PNG
#          ├─ filter     multi-view consistency filtering
#          └─ slam       DROID RGB-D SLAM using filtered depth
#      depth maps
#
#  Usage:
#      ./run_pipeline.sh -c config.json                    # Complete workflow
#      ./run_pipeline.sh -c config.json --from render      # Resume from a stage
#      ./run_pipeline.sh -c config.json --only stereo,depth,filter
#      ./run_pipeline.sh -c config.json --show             # Show resolved configuration
#
#  Configure parameters in config.json; the CLI controls flow and common overrides.
#==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="${SCRIPT_DIR}/lib"

ALL_STEPS=(colmap transforms train export shift campath render rotate
           intrinsic stereo depth filter slam)

#------------------------------------------------------------------ Output ----
c_step()  { printf '\n\033[1;36m━━━ %s ━━━\033[0m\n' "$*"; }
c_info()  { printf '\033[1;34m[*]\033[0m %s\n' "$*"; }
c_ok()    { printf '\033[1;32m[✓]\033[0m %s\n' "$*"; }
c_warn()  { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }
c_err()   { printf '\033[1;31m[✗]\033[0m %s\n' "$*" >&2; }
die()     { c_err "$*"; exit 1; }

show_usage() {
    sed -n '2,33p' "$0" | sed 's/^#//'
    cat <<'EOF'

Options:
  -c, --config FILE     Configuration file (default: config.json)
  --only  a,b,c         Run only these stages
  --from  STEP          Run from this stage to the end
  --list                List stages
  --show                Print resolved configuration and exit
  -h, --help            Show this help

Common overrides (edit config.json for other values):
  --shift VALUE            Set normalized baseline directly
  --shift-pixels FRAC      Target disparity as a fraction of image width
  --no-auto-direction      Use direction from config.json
  --shift-direction DIR    up / down / left / right
  --vis                    Save filter visualizations
  --clean                  Delete intermediate products afterward
EOF
    printf '\nStages: %s\n' "${ALL_STEPS[*]}"
}

#------------------------------------------------------------------ Arguments ----
CONFIG_FILE="${SCRIPT_DIR}/config.json"
ONLY=""; FROM=""; SHOW=0
declare -a OVERRIDES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)         show_usage; exit 0 ;;
        --list)            printf '%s\n' "${ALL_STEPS[@]}"; exit 0 ;;
        -c|--config)       CONFIG_FILE="$2"; shift 2 ;;
        --only)            ONLY="$2"; shift 2 ;;
        --from)            FROM="$2"; shift 2 ;;
        --show)            SHOW=1; shift ;;
        --shift)           OVERRIDES+=("SHIFT=$2" "SHIFT_MODE=baseline"); shift 2 ;;
        --shift-pixels)    OVERRIDES+=("SHIFT_PIXELS=$2" "SHIFT_MODE=pixels"); shift 2 ;;
        --shift-direction) OVERRIDES+=("SHIFT_DIRECTION=$2" "AUTO_DIRECTION=0"); shift 2 ;;
        --no-auto-direction) OVERRIDES+=("AUTO_DIRECTION=0"); shift ;;
        --vis)             OVERRIDES+=("VIS=1"); shift ;;
        --clean)           OVERRIDES+=("CLEAN_INTERMEDIATE=1"); shift ;;
        *)                 die "Unknown argument: $1 (use --help)" ;;
    esac
done

[[ -f "${CONFIG_FILE}" ]] || die "Configuration file not found: ${CONFIG_FILE} (use -c)"

if [[ "${SHOW}" == "1" ]]; then
    exec python3 "${LIB_DIR}/load_config.py" "${CONFIG_FILE}" --dump
fi

# config.json is the sole configuration source; CLI overrides apply afterward.
eval "$(python3 "${LIB_DIR}/load_config.py" "${CONFIG_FILE}")"
for kv in "${OVERRIDES[@]+"${OVERRIDES[@]}"}"; do eval "${kv}"; done

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS

#---------------------------------------------------------------- Derived paths ----
UNDIST_DIR="${COLMAP_WORK_DIR}/undistorted"
UNDIST_IMAGES="${UNDIST_DIR}/images"
TRANSFORMS_JSON="${DATA_DIR}/transforms.json"

CAM_DIR="${WORK_DIR}/cam_path"
RENDER_LEFT="${WORK_DIR}/render/left"
RENDER_RIGHT="${WORK_DIR}/render/right"
ROT_LEFT="${WORK_DIR}/rotated/left"
ROT_RIGHT="${WORK_DIR}/rotated/right"
RAW_DEPTH_DIR="${WORK_DIR}/raw_depth"
DEPTH_DIR="${WORK_DIR}/depth"
DEPTH_FILTERED_DIR="${WORK_DIR}/depth_filtered"
VIS_DIR="${WORK_DIR}/vis_filter"
SLAM_DIR="${WORK_DIR}/slam"

POSES_LEFT="${CAM_DIR}/transforms_train.json"
POSES_RIGHT="${CAM_DIR}/transforms_train_right.json"
CAMPATH_LEFT="${CAM_DIR}/camera_path_left.json"
CAMPATH_RIGHT="${CAM_DIR}/camera_path_right.json"
INTRINSIC_FILE="${CAM_DIR}/K.txt"
INTRINSIC_ORIGINAL="${CAM_DIR}/K_origin.txt"

#---------------------------------------------------------------- Stage selection ----
should_run() {
    local step="$1"
    # Disabled stages run only when explicitly named by --only.
    case "${step}" in
        colmap) [[ "${COLMAP_ENABLED}" == "1" || ",${ONLY}," == *",colmap,"* ]] || return 1 ;;
        filter) [[ "${FILTER_ENABLED}" == "1" || ",${ONLY}," == *",filter,"* ]] || return 1 ;;
        slam)   [[ "${SLAM_ENABLED}"   == "1" || ",${ONLY}," == *",slam,"*   ]] || return 1 ;;
    esac
    if [[ -n "${ONLY}" ]]; then
        [[ ",${ONLY}," == *",${step},"* ]]; return
    fi
    if [[ -n "${FROM}" ]]; then
        local started=0
        for s in "${ALL_STEPS[@]}"; do
            [[ "${s}" == "${FROM}" ]] && started=1
            [[ "${s}" == "${step}" ]] && { [[ ${started} -eq 1 ]]; return; }
        done
        return 1
    fi
    return 0
}

for s in ${ONLY//,/ } ${FROM}; do
    [[ " ${ALL_STEPS[*]} " == *" ${s} "* ]] || die "Unknown stage: ${s}"
done
case "${SHIFT_DIRECTION}" in
    up|down|left|right|x|-x|y|-y) ;;
    *) die "stereo.direction must be up/down/left/right (x/-x/y/-y also accepted)" ;;
esac

#------------------------------------------------------------------ Summary ----
cat <<EOF

  Config        ${CONFIG_FILE}
  Scene         ${DATA_DIR}
  Products      ${WORK_DIR}
  Baseline      $([[ "${SHIFT_MODE}" == "pixels" ]] \
                && echo "disparity ${SHIFT_PIXELS} x image width (direction ${SHIFT_DIRECTION})" \
                || echo "${SHIFT} normalized (direction ${SHIFT_DIRECTION})")
  Direction     $([[ "${AUTO_DIRECTION}" == "1" ]] && echo "automatic from trajectory" || echo "fixed ${SHIFT_DIRECTION}")
  SLAM          $([[ "${SLAM_ENABLED}" == "1" ]] && echo "on (depth prior)" || echo "off")
  Filter        $([[ "${FILTER_ENABLED}" == "1" ]] && echo "on (NCC $([[ "${FILTER_USE_NCC}" == "1" ]] && echo on || echo off))" || echo "off")
  Depth PNG     $([[ "${DEPTH_PNG}" == "1" ]] && echo yes || echo no)
  Visualization $([[ "${VIS}" == "1" ]] && echo yes || echo no)
  Intermediates $([[ "${CLEAN_INTERMEDIATE}" == "1" ]] && echo delete || echo keep)
EOF

#================================================================== Stages ====

# 1. COLMAP: features -> matching -> sparse reconstruction -> undistortion
if should_run colmap; then
    c_step "colmap - SfM + undistortion"
    "${STEREO_PY}" "${LIB_DIR}/run_colmap.py" \
        --image-dir "${COLMAP_IMAGE_DIR}" \
        --work-dir "${COLMAP_WORK_DIR}" \
        --matcher "${COLMAP_MATCHER}" \
        --camera-model "${COLMAP_CAMERA_MODEL}" \
        --single-camera "${COLMAP_SINGLE_CAMERA}" \
        --undistort "${COLMAP_UNDISTORT}" \
        --use-gpu "${COLMAP_USE_GPU}" \
        --colmap-bin "${COLMAP_BIN}"
    c_ok "${UNDIST_DIR}"
fi

# 2. Undistorted model -> transforms.json
if should_run transforms; then
    c_step "transforms — COLMAP -> transforms.json"
    src_dir="${UNDIST_DIR}"; img_dir="${UNDIST_IMAGES}"
    if [[ ! -d "${src_dir}" ]]; then
        # Read sparse directly when undistortion was disabled or SfM used PINHOLE.
        c_info "No undistorted product; using ${COLMAP_WORK_DIR}/sparse"
        src_dir="${COLMAP_WORK_DIR}"; img_dir="${COLMAP_IMAGE_DIR}"
    fi
    "${STEREO_PY}" "${LIB_DIR}/colmap_to_transforms.py" \
        --colmap-dir "${src_dir}" --image-dir "${img_dir}" \
        --output "${TRANSFORMS_JSON}"
    c_ok "${TRANSFORMS_JSON}"
fi

# 3. Train nerfstudio
if should_run train; then
    c_step "train - ${NS_METHOD}"
    [[ -f "${TRANSFORMS_JSON}" ]] || die "${TRANSFORMS_JSON} not found; run transforms first"
    # shellcheck disable=SC2086
    "${NS_TRAIN}" "${NS_METHOD}" \
        --data "${DATA_DIR}" \
        --output-dir "${NS_OUTPUT_DIR}" \
        --max-num-iterations "${NS_ITERS}" \
        --vis tensorboard \
        --viewer.quit-on-train-completion True ${NS_EXTRA_ARGS}
    c_ok "${NS_OUTPUT_DIR}"
fi

#--------------------------------------------- Later stages require a trained model ----
need_model=0
for s in export campath render intrinsic; do should_run "$s" && need_model=1; done
# pixels mode needs dataparser scale to convert baseline into normalized space.
[[ "${SHIFT_MODE}" == "pixels" ]] && should_run shift && need_model=1
if [[ ${need_model} -eq 1 ]]; then
    if [[ -z "${CONFIG_PATH}" ]]; then
        CONFIG_PATH="$(find "${NS_OUTPUT_DIR}" -name config.yml -printf '%T@ %p\n' 2>/dev/null \
                       | sort -rn | head -1 | cut -d' ' -f2-)"
        [[ -n "${CONFIG_PATH}" ]] || die "No config.yml under ${NS_OUTPUT_DIR}; set nerfstudio.config_path"
        c_info "Automatically selected latest model: ${CONFIG_PATH}"
    fi
    [[ -f "${CONFIG_PATH}" ]] || die "Model configuration not found: ${CONFIG_PATH}"
    DATAPARSER_TRANSFORMS="$(dirname "${CONFIG_PATH}")/dataparser_transforms.json"
    [[ -f "${TRANSFORMS_JSON}" ]] || die "${TRANSFORMS_JSON} not found"
fi

mkdir -p "${CAM_DIR}" "${RENDER_LEFT}" "${RENDER_RIGHT}" \
         "${ROT_LEFT}" "${ROT_RIGHT}" "${RAW_DEPTH_DIR}" "${DEPTH_DIR}"

# Recover saved direction/baseline when shift is skipped to keep later stages aligned.
if ! should_run shift && [[ -f "${CAM_DIR}/stereo_params.txt" ]]; then
    SHIFT_DIRECTION="$(sed -n 1p "${CAM_DIR}/stereo_params.txt")"
    SHIFT="$(sed -n 2p "${CAM_DIR}/stereo_params.txt")"
    c_info "Reusing stereo parameters: direction ${SHIFT_DIRECTION}, shift ${SHIFT}"
fi

# 4. Export camera poses
if should_run export; then
    c_step "export - camera poses"
    "${NERF_PY}" "${LIB_DIR}/export_poses.py" \
        --load-config "${CONFIG_PATH}" --output-dir "${CAM_DIR}" \
        --combine-train-eval --reference-transforms "${TRANSFORMS_JSON}"
    [[ -f "${POSES_LEFT}" ]] || die "Pose export failed"
    c_ok "$(basename "${POSES_LEFT}")"
fi

# 5. Generate right-camera poses
if should_run shift; then
    c_step "shift - right-camera poses"
    # Follow the trajectory to stay near observed NeRF viewpoints. One global
    # direction is required because downstream uses a single K.txt.
    if [[ "${AUTO_DIRECTION}" == "1" ]]; then
        SHIFT_DIRECTION="$("${STEREO_PY}" "${LIB_DIR}/auto_direction.py" \
            --poses "${POSES_LEFT}" --fallback "${SHIFT_DIRECTION}" \
            --min-dominance "${AUTO_DIR_MIN_DOM}")"
        c_info "Automatic direction -> ${SHIFT_DIRECTION}"
    fi
    # In pixels mode, derive baseline from target disparity and COLMAP reference depth.
    if [[ "${SHIFT_MODE}" == "pixels" ]]; then
        rd_flag=(); [[ -n "${REFERENCE_DEPTH}" ]] && rd_flag+=(--reference-depth "${REFERENCE_DEPTH}")
        SHIFT="$("${STEREO_PY}" "${LIB_DIR}/resolve_shift.py" \
            --transforms-json "${TRANSFORMS_JSON}" \
            --dataparser-transforms "${DATAPARSER_TRANSFORMS}" \
            --mode pixels --shift-pixels "${SHIFT_PIXELS}" \
            --colmap-dir "${COLMAP_WORK_DIR}" \
            --percentile "${REFERENCE_DEPTH_PCT}" "${rd_flag[@]+"${rd_flag[@]}"}")"
        c_info "Disparity ${SHIFT_PIXELS} x width -> normalized shift = ${SHIFT}"
    fi
    "${STEREO_PY}" "${LIB_DIR}/stereo_shift.py" \
        --input "${POSES_LEFT}" --output "${POSES_RIGHT}" \
        --shift "${SHIFT}" --shift-direction "${SHIFT_DIRECTION}"
    # Save parameters for standalone rotate/intrinsic/depth runs.
    printf '%s\n%s\n' "${SHIFT_DIRECTION}" "${SHIFT}" > "${CAM_DIR}/stereo_params.txt"
    c_ok "$(basename "${POSES_RIGHT}")"
fi

# 6. Convert to camera_path format
if should_run campath; then
    c_step "campath - camera_path format"
    for side in left right; do
        [[ "${side}" == "left" ]] && { src="${POSES_LEFT}";  dst="${CAMPATH_LEFT}"; } \
                                  || { src="${POSES_RIGHT}"; dst="${CAMPATH_RIGHT}"; }
        "${STEREO_PY}" "${LIB_DIR}/make_camera_path.py" \
            "${src}" "${TRANSFORMS_JSON}" "${dst}" --config "${CONFIG_PATH}" --verbose
    done
    c_ok "camera_path_{left,right}.json"
fi

# 7. Render left and right views
if should_run render; then
    c_step "render - stereo views"
    for side in left right; do
        [[ "${side}" == "left" ]] && { cp_json="${CAMPATH_LEFT}";  out="${RENDER_LEFT}"; } \
                                  || { cp_json="${CAMPATH_RIGHT}"; out="${RENDER_RIGHT}"; }
        c_info "Render ${side} -> ${out}"
        "${NS_RENDER}" camera-path --load-config "${CONFIG_PATH}" \
            --camera-path-filename "${cp_json}" --output-path "${out}" \
            --output-format images --image-format png
    done
    n_l=$(find "${RENDER_LEFT}" -name '*.png' | wc -l)
    n_r=$(find "${RENDER_RIGHT}" -name '*.png' | wc -l)
    [[ "${n_l}" -gt 0 && "${n_l}" -eq "${n_r}" ]] || die "Render mismatch: left ${n_l}, right ${n_r}"
    c_ok "${n_l} images per side"
fi

# 8. Rotate to stereo-matching orientation
if should_run rotate; then
    c_step "rotate - images"
    "${STEREO_PY}" "${LIB_DIR}/rotate_images.py" \
        --left-dir "${RENDER_LEFT}" --right-dir "${RENDER_RIGHT}" \
        --output-left-dir "${ROT_LEFT}" --output-right-dir "${ROT_RIGHT}" \
        --shift-direction "${SHIFT_DIRECTION}"
    c_ok "rotated/{left,right}"
fi

# 9. Generate K.txt
if should_run intrinsic; then
    c_step "intrinsic - generate K.txt"
    "${STEREO_PY}" "${LIB_DIR}/make_intrinsics.py" \
        --transforms-json "${TRANSFORMS_JSON}" \
        --dataparser-transforms "${DATAPARSER_TRANSFORMS}" \
        --shift "${SHIFT}" --shift-direction "${SHIFT_DIRECTION}" \
        --output "${INTRINSIC_FILE}" --output-original "${INTRINSIC_ORIGINAL}"
    c_ok "$(basename "${INTRINSIC_FILE}")"
fi

# 10. FoundationStereo
if should_run stereo; then
    c_step "stereo - FoundationStereo inference"
    [[ -f "${FOUNDATION_MODEL}" ]] || die "Weights not found: ${FOUNDATION_MODEL}"
    # batch_process.py uses sys.path.append('..'); run from the repository root.
    ( cd "${FOUNDATION_DIR}" && "${STEREO_PY}" scripts/batch_process.py \
        --left_dir "${ROT_LEFT}" --right_dir "${ROT_RIGHT}" \
        --intrinsic_file "${INTRINSIC_FILE}" --ckpt_dir "${FOUNDATION_MODEL}" \
        --out_dir "${RAW_DEPTH_DIR}" --valid_iters "${VALID_ITERS}" )
    n=$(find "${RAW_DEPTH_DIR}" -name '*.npy' | wc -l)
    [[ "${n}" -gt 0 ]] || die "No depth maps were produced"
    c_ok "${n} maps -> ${RAW_DEPTH_DIR}"
fi

# 11. Depth postprocessing
if should_run depth; then
    c_step "depth - postprocessing"
    png_flag=(); [[ "${DEPTH_PNG}" == "1" ]] || png_flag+=(--no-png)
    "${STEREO_PY}" "${LIB_DIR}/depth_postprocess.py" \
        --depth-in "${RAW_DEPTH_DIR}" --depth-out "${DEPTH_DIR}" \
        --shift-direction "${SHIFT_DIRECTION}" \
        --transforms-json "${TRANSFORMS_JSON}" "${png_flag[@]+"${png_flag[@]}"}"
    c_ok "${DEPTH_DIR}"
fi

# 12. Multi-view geometric consistency filtering
if should_run filter; then
    c_step "filter - multi-view geometric consistency"
    vis_flag=(); [[ "${VIS}" == "1" ]] && vis_flag+=(--vis-dir "${VIS_DIR}")
    conf_flag=(); [[ "${FILTER_SAVE_CONF}" == "1" ]] && conf_flag+=(--save-confidence)
    ncc_flag=()
    if [[ "${FILTER_USE_NCC}" == "1" && -d "${RENDER_LEFT}" ]]; then
        ncc_flag+=(--image-dir "${RENDER_LEFT}" --ncc-window "${FILTER_NCC_WINDOW}"
                   --min-ncc "${FILTER_MIN_NCC}" --min-texture-std "${FILTER_MIN_TEXTURE_STD}")
    elif [[ "${FILTER_USE_NCC}" == "1" ]]; then
        c_warn "Rendered images were cleaned; skipping NCC and using geometry only"
    fi
    "${STEREO_PY}" "${LIB_DIR}/filter_depth.py" \
        --transforms-json "${TRANSFORMS_JSON}" \
        --depth-dir "${DEPTH_DIR}" --output-dir "${DEPTH_FILTERED_DIR}" \
        --num-src "${FILTER_NUM_SRC}" \
        --max-reproj-error "${FILTER_MAX_REPROJ}" \
        --max-depth-error "${FILTER_MAX_DEPTH_ERR}" \
        --min-triangulation-angle "${FILTER_MIN_TRI_ANGLE}" \
        --min-num-consistent "${FILTER_MIN_CONSISTENT}" \
        "${ncc_flag[@]+"${ncc_flag[@]}"}" "${conf_flag[@]+"${conf_flag[@]}"}" "${vis_flag[@]+"${vis_flag[@]}"}"
    c_ok "${DEPTH_FILTERED_DIR}"
fi

# 13. Run RGB-D SLAM with filtered depth
if should_run slam; then
    c_step "slam - DROID RGB-D SLAM with depth priors"
    slam_depth="${DEPTH_FILTERED_DIR}"
    if [[ "${SLAM_USE_FILTERED}" != "1" || ! -d "${slam_depth}" ]]; then
        [[ "${SLAM_USE_FILTERED}" == "1" ]] && c_warn "No depth_filtered; using unfiltered depth"
        slam_depth="${DEPTH_DIR}"
    fi
    eval_flag=(); [[ "${SLAM_EVALUATE}" == "1" ]] || eval_flag+=(--no-eval)
    "${STEREO_PY}" "${LIB_DIR}/run_rgbd_slam.py" \
        --droid-metric-dir "${DROID_METRIC_DIR}" \
        --image-dir "${RENDER_LEFT}" \
        --depth-dir "${slam_depth}" \
        --transforms-json "${TRANSFORMS_JSON}" \
        --output-dir "${SLAM_DIR}" \
        --global-ba-frontend "${SLAM_GLOBAL_BA}" \
        "${eval_flag[@]+"${eval_flag[@]}"}"
    c_ok "${SLAM_DIR}"
fi

# Clean intermediates last so failed runs can resume from stereo.
if [[ "${CLEAN_INTERMEDIATE}" == "1" ]] && should_run depth; then
    freed=$(du -sh "${WORK_DIR}/rotated" "${RAW_DEPTH_DIR}" 2>/dev/null | awk '{print $1}' | tr '\n' ' ')
    rm -rf "${WORK_DIR}/rotated" "${RAW_DEPTH_DIR}"
    c_ok "Deleted rotated/ and raw_depth/ intermediates (${freed})"
fi

c_step "Complete"
cat <<EOF

  Rendered training views ${RENDER_LEFT}
  Rendered stereo views   ${RENDER_RIGHT}
  Depth maps              ${DEPTH_DIR}
$([[ "${FILTER_ENABLED}" == "1" ]] && echo "  Filtered depth          ${DEPTH_FILTERED_DIR}")
$([[ "${SLAM_ENABLED}" == "1" ]] && echo "  SLAM poses              ${SLAM_DIR}/poses")

  Disk usage              $(du -sh "${WORK_DIR}" 2>/dev/null | cut -f1)
EOF
