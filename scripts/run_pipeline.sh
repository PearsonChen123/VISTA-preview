#!/usr/bin/env bash
#==============================================================================
#  stereo-depth pipeline
#
#  一条命令从原始图像跑到过滤后的深度图：
#
#      原始图像
#          ├─ colmap     SfM + 去畸变（一步跑完 4 条 colmap 命令）
#          ├─ transforms 去畸变模型 -> transforms.json
#          ├─ train      训练 nerfstudio 模型
#          ├─ export     从模型导出训练视角的相机位姿
#          ├─ shift      在相机自身坐标系下平移出右目位姿
#          ├─ campath    转成 nerfstudio 的 camera_path 格式
#          ├─ render     渲染左目（训练视角）+ 右目
#          ├─ rotate     旋转到立体匹配要求的朝向
#          ├─ intrinsic  生成 K.txt（内参 + 换算成真实尺度的基线）
#          ├─ stereo     FoundationStereo 推理
#          ├─ depth      深度图转回原始朝向 + 存 16 位 PNG
#          ├─ filter     多视图几何一致性过滤，剔除不可信深度
#          └─ slam       把过滤后的深度当可信深度，跑 DROID RGBD SLAM 出位姿
#      深度图
#
#  用法:
#      ./run_pipeline.sh -c config.json                    # 全流程
#      ./run_pipeline.sh -c config.json --from render      # 从某步开始
#      ./run_pipeline.sh -c config.json --only stereo,depth,filter
#      ./run_pipeline.sh -c config.json --show             # 只看解析后的配置
#
#  所有参数都在 config.json 里改。命令行只有流程控制和少数几个常用覆盖项。
#==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="${SCRIPT_DIR}/lib"

ALL_STEPS=(colmap transforms train export shift campath render rotate
           intrinsic stereo depth filter slam)

#------------------------------------------------------------------ 输出 ----
c_step()  { printf '\n\033[1;36m━━━ %s ━━━\033[0m\n' "$*"; }
c_info()  { printf '\033[1;34m[*]\033[0m %s\n' "$*"; }
c_ok()    { printf '\033[1;32m[✓]\033[0m %s\n' "$*"; }
c_warn()  { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }
c_err()   { printf '\033[1;31m[✗]\033[0m %s\n' "$*" >&2; }
die()     { c_err "$*"; exit 1; }

show_usage() {
    sed -n '2,33p' "$0" | sed 's/^#//'
    cat <<'EOF'

选项:
  -c, --config FILE     配置文件（默认 config.json）
  --only  a,b,c         只跑这几步
  --from  STEP          从这步开始跑到最后
  --list                列出所有步骤
  --show                打印解析后的完整配置后退出
  -h, --help            显示本帮助

常用覆盖（其余请改 config.json）:
  --shift VALUE            直接给归一化基线（切到 baseline 模式）
  --shift-pixels FRAC      目标视差占图像宽度的比例（切到 pixels 模式）
  --no-auto-direction      关掉自动方向，用 config.json 里的 direction
  --shift-direction DIR    up / down / left / right
  --vis                    输出过滤结果的可视化三联图
  --clean                  跑完删掉中间产物
EOF
    printf '\n步骤: %s\n' "${ALL_STEPS[*]}"
}

#------------------------------------------------------------------ 参数 ----
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
        *)                 die "未知参数: $1（用 --help 看用法）" ;;
    esac
done

[[ -f "${CONFIG_FILE}" ]] || die "找不到配置文件: ${CONFIG_FILE}（用 -c 指定）"

if [[ "${SHOW}" == "1" ]]; then
    exec python3 "${LIB_DIR}/load_config.py" "${CONFIG_FILE}" --dump
fi

# config.json 是唯一配置来源；命令行覆盖在它之后生效
eval "$(python3 "${LIB_DIR}/load_config.py" "${CONFIG_FILE}")"
for kv in "${OVERRIDES[@]+"${OVERRIDES[@]}"}"; do eval "${kv}"; done

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS

#---------------------------------------------------------------- 派生路径 ----
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

#---------------------------------------------------------------- 步骤选择 ----
should_run() {
    local step="$1"
    # 配置里关掉的步骤，除非 --only 显式点名，否则不跑
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
    [[ " ${ALL_STEPS[*]} " == *" ${s} "* ]] || die "未知步骤: ${s}"
done
case "${SHIFT_DIRECTION}" in
    up|down|left|right|x|-x|y|-y) ;;
    *) die "stereo.direction 只能是 up/down/left/right（也接受 x/-x/y/-y）" ;;
esac

#------------------------------------------------------------------ 概要 ----
cat <<EOF

  配置        ${CONFIG_FILE}
  场景        ${DATA_DIR}
  产物        ${WORK_DIR}
  基线        $([[ "${SHIFT_MODE}" == "pixels" ]] \
                && echo "视差 ${SHIFT_PIXELS} x 图像宽度 (方向 ${SHIFT_DIRECTION})" \
                || echo "${SHIFT} 归一化 (方向 ${SHIFT_DIRECTION})")
  方向        $([[ "${AUTO_DIRECTION}" == "1" ]] && echo "跟随轨迹自动判定" || echo "固定 ${SHIFT_DIRECTION}")
  SLAM        $([[ "${SLAM_ENABLED}" == "1" ]] && echo "开（深度当先验）" || echo "关")
  过滤        $([[ "${FILTER_ENABLED}" == "1" ]] && echo "开 (NCC $([[ "${FILTER_USE_NCC}" == "1" ]] && echo 开 || echo 关))" || echo "关")
  深度 PNG    $([[ "${DEPTH_PNG}" == "1" ]] && echo 生成 || echo 不生成)
  可视化      $([[ "${VIS}" == "1" ]] && echo 生成 || echo 不生成)
  中间产物    $([[ "${CLEAN_INTERMEDIATE}" == "1" ]] && echo 跑完删除 || echo 保留)
EOF

#================================================================== 步骤 ====

# 1. COLMAP：特征 -> 匹配 -> 稀疏重建 -> 去畸变
if should_run colmap; then
    c_step "colmap — SfM + 去畸变"
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

# 2. 去畸变模型 -> transforms.json
if should_run transforms; then
    c_step "transforms — COLMAP -> transforms.json"
    src_dir="${UNDIST_DIR}"; img_dir="${UNDIST_IMAGES}"
    if [[ ! -d "${src_dir}" ]]; then
        # 没跑去畸变（colmap.undistort=false，或 SfM 就用的 PINHOLE）时直接读 sparse。
        # 模型带畸变的话 colmap_to_transforms.py 会报错，不会静默放过。
        c_info "没有去畸变产物，直接用 ${COLMAP_WORK_DIR}/sparse"
        src_dir="${COLMAP_WORK_DIR}"; img_dir="${COLMAP_IMAGE_DIR}"
    fi
    "${STEREO_PY}" "${LIB_DIR}/colmap_to_transforms.py" \
        --colmap-dir "${src_dir}" --image-dir "${img_dir}" \
        --output "${TRANSFORMS_JSON}"
    c_ok "${TRANSFORMS_JSON}"
fi

# 3. 训练 nerfstudio
if should_run train; then
    c_step "train — 训练 ${NS_METHOD}"
    [[ -f "${TRANSFORMS_JSON}" ]] || die "找不到 ${TRANSFORMS_JSON}，先跑 transforms"
    # shellcheck disable=SC2086
    "${NS_TRAIN}" "${NS_METHOD}" \
        --data "${DATA_DIR}" \
        --output-dir "${NS_OUTPUT_DIR}" \
        --max-num-iterations "${NS_ITERS}" \
        --vis tensorboard \
        --viewer.quit-on-train-completion True ${NS_EXTRA_ARGS}
    c_ok "${NS_OUTPUT_DIR}"
fi

#--------------------------------------------- 后续步骤都要用训练好的模型 ----
need_model=0
for s in export campath render intrinsic; do should_run "$s" && need_model=1; done
# shift_mode=pixels 要从 dataparser_transforms.json 取 scale 才能把基线换算到归一化空间
[[ "${SHIFT_MODE}" == "pixels" ]] && should_run shift && need_model=1
if [[ ${need_model} -eq 1 ]]; then
    if [[ -z "${CONFIG_PATH}" ]]; then
        CONFIG_PATH="$(find "${NS_OUTPUT_DIR}" -name config.yml -printf '%T@ %p\n' 2>/dev/null \
                       | sort -rn | head -1 | cut -d' ' -f2-)"
        [[ -n "${CONFIG_PATH}" ]] || die "在 ${NS_OUTPUT_DIR} 下找不到 config.yml，请在 config.json 里指定 nerfstudio.config_path"
        c_info "自动选用最新模型: ${CONFIG_PATH}"
    fi
    [[ -f "${CONFIG_PATH}" ]] || die "找不到模型配置: ${CONFIG_PATH}"
    DATAPARSER_TRANSFORMS="$(dirname "${CONFIG_PATH}")/dataparser_transforms.json"
    [[ -f "${TRANSFORMS_JSON}" ]] || die "找不到 ${TRANSFORMS_JSON}"
fi

mkdir -p "${CAM_DIR}" "${RENDER_LEFT}" "${RENDER_RIGHT}" \
         "${ROT_LEFT}" "${ROT_RIGHT}" "${RAW_DEPTH_DIR}" "${DEPTH_DIR}"

# shift 那步没跑（比如 --only rotate）时，从上次的记录里取回方向和基线，
# 否则旋转/内参/深度三处口径会和渲染时不一致
if ! should_run shift && [[ -f "${CAM_DIR}/stereo_params.txt" ]]; then
    SHIFT_DIRECTION="$(sed -n 1p "${CAM_DIR}/stereo_params.txt")"
    SHIFT="$(sed -n 2p "${CAM_DIR}/stereo_params.txt")"
    c_info "沿用上次的双目参数: 方向 ${SHIFT_DIRECTION}, shift ${SHIFT}"
fi

# 4. 导出相机位姿
if should_run export; then
    c_step "export — 导出相机位姿"
    "${NERF_PY}" "${LIB_DIR}/export_poses.py" \
        --load-config "${CONFIG_PATH}" --output-dir "${CAM_DIR}" \
        --combine-train-eval --reference-transforms "${TRANSFORMS_JSON}"
    [[ -f "${POSES_LEFT}" ]] || die "位姿导出失败"
    c_ok "$(basename "${POSES_LEFT}")"
fi

# 5. 平移出右目位姿
if should_run shift; then
    c_step "shift — 生成右目位姿"
    # 方向跟着相机轨迹走：沿轨迹平移落在 NeRF 观测过的视角流形内，
    # 垂直于轨迹是外推、渲染会糊。方向必须全局统一（下游 K.txt 只有一份）。
    if [[ "${AUTO_DIRECTION}" == "1" ]]; then
        SHIFT_DIRECTION="$("${STEREO_PY}" "${LIB_DIR}/auto_direction.py" \
            --poses "${POSES_LEFT}" --fallback "${SHIFT_DIRECTION}" \
            --min-dominance "${AUTO_DIR_MIN_DOM}")"
        c_info "自动方向 -> ${SHIFT_DIRECTION}"
    fi
    # shift_mode=pixels 时，先按"目标视差占宽度的百分比"换算出基线。
    # 参考深度来自 COLMAP 稀疏点云，不需要渲染。
    if [[ "${SHIFT_MODE}" == "pixels" ]]; then
        rd_flag=(); [[ -n "${REFERENCE_DEPTH}" ]] && rd_flag+=(--reference-depth "${REFERENCE_DEPTH}")
        SHIFT="$("${STEREO_PY}" "${LIB_DIR}/resolve_shift.py" \
            --transforms-json "${TRANSFORMS_JSON}" \
            --dataparser-transforms "${DATAPARSER_TRANSFORMS}" \
            --mode pixels --shift-pixels "${SHIFT_PIXELS}" \
            --colmap-dir "${COLMAP_WORK_DIR}" \
            --percentile "${REFERENCE_DEPTH_PCT}" "${rd_flag[@]+"${rd_flag[@]}"}")"
        c_info "视差 ${SHIFT_PIXELS} x 宽度  ->  归一化 shift = ${SHIFT}"
    fi
    "${STEREO_PY}" "${LIB_DIR}/stereo_shift.py" \
        --input "${POSES_LEFT}" --output "${POSES_RIGHT}" \
        --shift "${SHIFT}" --shift-direction "${SHIFT_DIRECTION}"
    # 存下来，这样后面 --only rotate/intrinsic/depth 单跑时能拿到同一个方向
    printf '%s\n%s\n' "${SHIFT_DIRECTION}" "${SHIFT}" > "${CAM_DIR}/stereo_params.txt"
    c_ok "$(basename "${POSES_RIGHT}")"
fi

# 6. 转 camera_path 格式
if should_run campath; then
    c_step "campath — 转 camera_path 格式"
    for side in left right; do
        [[ "${side}" == "left" ]] && { src="${POSES_LEFT}";  dst="${CAMPATH_LEFT}"; } \
                                  || { src="${POSES_RIGHT}"; dst="${CAMPATH_RIGHT}"; }
        "${STEREO_PY}" "${LIB_DIR}/make_camera_path.py" \
            "${src}" "${TRANSFORMS_JSON}" "${dst}" --config "${CONFIG_PATH}" --verbose
    done
    c_ok "camera_path_{left,right}.json"
fi

# 7. 渲染左右目
if should_run render; then
    c_step "render — 渲染左右目"
    for side in left right; do
        [[ "${side}" == "left" ]] && { cp_json="${CAMPATH_LEFT}";  out="${RENDER_LEFT}"; } \
                                  || { cp_json="${CAMPATH_RIGHT}"; out="${RENDER_RIGHT}"; }
        c_info "渲染 ${side} -> ${out}"
        "${NS_RENDER}" camera-path --load-config "${CONFIG_PATH}" \
            --camera-path-filename "${cp_json}" --output-path "${out}" \
            --output-format images --image-format png
    done
    n_l=$(find "${RENDER_LEFT}" -name '*.png' | wc -l)
    n_r=$(find "${RENDER_RIGHT}" -name '*.png' | wc -l)
    [[ "${n_l}" -gt 0 && "${n_l}" -eq "${n_r}" ]] || die "渲染异常: 左 ${n_l} 右 ${n_r}"
    c_ok "左右各 ${n_l} 张"
fi

# 8. 旋转到立体匹配朝向
if should_run rotate; then
    c_step "rotate — 旋转图像"
    "${STEREO_PY}" "${LIB_DIR}/rotate_images.py" \
        --left-dir "${RENDER_LEFT}" --right-dir "${RENDER_RIGHT}" \
        --output-left-dir "${ROT_LEFT}" --output-right-dir "${ROT_RIGHT}" \
        --shift-direction "${SHIFT_DIRECTION}"
    c_ok "rotated/{left,right}"
fi

# 9. 生成 K.txt
if should_run intrinsic; then
    c_step "intrinsic — 生成 K.txt"
    "${STEREO_PY}" "${LIB_DIR}/make_intrinsics.py" \
        --transforms-json "${TRANSFORMS_JSON}" \
        --dataparser-transforms "${DATAPARSER_TRANSFORMS}" \
        --shift "${SHIFT}" --shift-direction "${SHIFT_DIRECTION}" \
        --output "${INTRINSIC_FILE}" --output-original "${INTRINSIC_ORIGINAL}"
    c_ok "$(basename "${INTRINSIC_FILE}")"
fi

# 10. FoundationStereo
if should_run stereo; then
    c_step "stereo — FoundationStereo 推理"
    [[ -f "${FOUNDATION_MODEL}" ]] || die "找不到权重: ${FOUNDATION_MODEL}"
    # batch_process.py 用 sys.path.append('..') 找 core/Utils，要在仓库根目录跑
    ( cd "${FOUNDATION_DIR}" && "${STEREO_PY}" scripts/batch_process.py \
        --left_dir "${ROT_LEFT}" --right_dir "${ROT_RIGHT}" \
        --intrinsic_file "${INTRINSIC_FILE}" --ckpt_dir "${FOUNDATION_MODEL}" \
        --out_dir "${RAW_DEPTH_DIR}" --valid_iters "${VALID_ITERS}" )
    n=$(find "${RAW_DEPTH_DIR}" -name '*.npy' | wc -l)
    [[ "${n}" -gt 0 ]] || die "没有输出任何深度图"
    c_ok "${n} 张 -> ${RAW_DEPTH_DIR}"
fi

# 11. 深度后处理
if should_run depth; then
    c_step "depth — 深度后处理"
    png_flag=(); [[ "${DEPTH_PNG}" == "1" ]] || png_flag+=(--no-png)
    "${STEREO_PY}" "${LIB_DIR}/depth_postprocess.py" \
        --depth-in "${RAW_DEPTH_DIR}" --depth-out "${DEPTH_DIR}" \
        --shift-direction "${SHIFT_DIRECTION}" \
        --transforms-json "${TRANSFORMS_JSON}" "${png_flag[@]+"${png_flag[@]}"}"
    c_ok "${DEPTH_DIR}"
fi

# 12. 多视图几何一致性过滤
if should_run filter; then
    c_step "filter — 多视图几何一致性过滤"
    vis_flag=(); [[ "${VIS}" == "1" ]] && vis_flag+=(--vis-dir "${VIS_DIR}")
    conf_flag=(); [[ "${FILTER_SAVE_CONF}" == "1" ]] && conf_flag+=(--save-confidence)
    ncc_flag=()
    if [[ "${FILTER_USE_NCC}" == "1" && -d "${RENDER_LEFT}" ]]; then
        ncc_flag+=(--image-dir "${RENDER_LEFT}" --ncc-window "${FILTER_NCC_WINDOW}"
                   --min-ncc "${FILTER_MIN_NCC}" --min-texture-std "${FILTER_MIN_TEXTURE_STD}")
    elif [[ "${FILTER_USE_NCC}" == "1" ]]; then
        c_warn "渲染图已被清理，NCC 跳过，只做几何三项"
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

# 13. 用过滤后的深度跑 RGBD SLAM
if should_run slam; then
    c_step "slam — DROID RGBD SLAM（深度当可信先验）"
    slam_depth="${DEPTH_FILTERED_DIR}"
    if [[ "${SLAM_USE_FILTERED}" != "1" || ! -d "${slam_depth}" ]]; then
        [[ "${SLAM_USE_FILTERED}" == "1" ]] && c_warn "没有 depth_filtered，改用未过滤的 depth"
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

# 中间产物清理放最后，中途失败时还能从 stereo 续跑
if [[ "${CLEAN_INTERMEDIATE}" == "1" ]] && should_run depth; then
    freed=$(du -sh "${WORK_DIR}/rotated" "${RAW_DEPTH_DIR}" 2>/dev/null | awk '{print $1}' | tr '\n' ' ')
    rm -rf "${WORK_DIR}/rotated" "${RAW_DEPTH_DIR}"
    c_ok "已删除中间产物 rotated/ raw_depth/ (${freed})"
fi

c_step "完成"
cat <<EOF

  渲染的训练视角   ${RENDER_LEFT}
  渲染的双目视角   ${RENDER_RIGHT}
  深度图           ${DEPTH_DIR}
$([[ "${FILTER_ENABLED}" == "1" ]] && echo "  过滤后深度       ${DEPTH_FILTERED_DIR}")
$([[ "${SLAM_ENABLED}" == "1" ]] && echo "  SLAM 位姿        ${SLAM_DIR}/poses")

  占用             $(du -sh "${WORK_DIR}" 2>/dev/null | cut -f1)
EOF
