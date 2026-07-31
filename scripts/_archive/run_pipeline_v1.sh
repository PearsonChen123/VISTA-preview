#!/usr/bin/env bash
#==============================================================================
#  stereo-depth pipeline
#
#  从训练好的 nerfstudio 模型出发，渲染出训练视角和对应的双目视角，
#  跑 FoundationStereo 得到深度图。
#
#      训练好的模型
#          │
#          ├─ export    从模型里导出训练视角的相机位姿
#          ├─ shift     平移出右目位姿
#          ├─ campath   转成 nerfstudio 的 camera_path 格式
#          ├─ render    渲染左目（训练视角）和右目
#          ├─ rotate    旋转到立体匹配要求的朝向
#          ├─ intrinsic 生成 K.txt（内参 + 换算后的真实基线）
#          ├─ stereo    FoundationStereo 推理
#          ├─ depth     深度图转回原始朝向 + 存 16 位 PNG
#          └─ filter    多视图几何一致性过滤，剔除不可信深度
#
#  用法:
#      ./run_pipeline.sh --data-dir DIR [选项]
#      ./run_pipeline.sh --data-dir DIR --only stereo,depth     # 只跑某几步
#      ./run_pipeline.sh --data-dir DIR --from render           # 从某步开始
#
#  参数见 --help，默认值在 config.sh。
#==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="${SCRIPT_DIR}/lib"

# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

ALL_STEPS=(export shift campath render rotate intrinsic stereo depth filter)

#------------------------------------------------------------------ 输出 ----
c_step()  { printf '\n\033[1;36m━━━ %s ━━━\033[0m\n' "$*"; }
c_info()  { printf '\033[1;34m[*]\033[0m %s\n' "$*"; }
c_ok()    { printf '\033[1;32m[✓]\033[0m %s\n' "$*"; }
c_err()   { printf '\033[1;31m[✗]\033[0m %s\n' "$*" >&2; }

die() { c_err "$*"; exit 1; }

show_usage() {
    sed -n '2,26p' "$0" | sed 's/^#//'
    cat <<EOF

选项:
  --data-dir DIR          数据集根目录（含 transforms.json），必填
  --config FILE           训练好的模型 config.yml（默认在 DATA_DIR/outputs 下找最新的）
  --work-dir DIR          产物目录（默认 DATA_DIR/stereo_depth）
  --shift VALUE           基线长度，归一化坐标系（默认 ${SHIFT}）
  --shift-direction DIR   右目画面相对左目往哪边移: up/down/left/right
                          （默认 ${SHIFT_DIRECTION}；也接受旧写法 x/-x/y/-y）
  --valid-iters N         FoundationStereo 迭代次数（默认 ${VALID_ITERS}）
  --foundation-model FILE FoundationStereo 权重（默认项目内 models/ 下的）
  --only  a,b,c           只跑这几步
  --from  STEP            从这步开始跑到最后
  --list                  列出所有步骤
  -h, --help              显示本帮助

省磁盘（30 帧 640x480 约 108M，一半以上是中间产物；500 帧 1080p 一次约 10G）:
  --no-vis                深度只存 .npy，不存 16 位 PNG
  --clean                 跑完删掉 rotated/ 和 raw_depth/
                          （删了就没法单独重跑 depth，得连 stereo 一起重跑）
  --lean                  等于 --no-vis --clean

深度过滤（filter 步骤，用位姿做多视图一致性检验）:
  --vis                     输出过滤结果的可视化三联图（默认不输出）
  --filter-max-depth-error  相对深度误差上限（默认 ${FILTER_MAX_DEPTH_ERR}，最严的一关）
  --filter-max-reproj       前后向重投影误差上限，像素（默认 ${FILTER_MAX_REPROJ}）
  --filter-min-consistent   至少几个源视角一致才保留（默认 ${FILTER_MIN_CONSISTENT}）
  --filter-num-src          每个参考视角用几个源视角（默认 ${FILTER_NUM_SRC}）
  --filter-min-ncc          NCC 下限（默认 ${FILTER_MIN_NCC}；无纹理区自动弃权）
  --filter-no-ncc           关掉 NCC，只做几何三项

步骤: ${ALL_STEPS[*]}
EOF
}

#------------------------------------------------------------------ 参数 ----
ONLY=""; FROM=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)          show_usage; exit 0 ;;
        --list)             printf '%s\n' "${ALL_STEPS[@]}"; exit 0 ;;
        --data-dir)         DATA_DIR="$2"; shift 2 ;;
        --config)           CONFIG_PATH="$2"; shift 2 ;;
        --work-dir)         WORK_DIR="$2"; shift 2 ;;
        --shift)            SHIFT="$2"; shift 2 ;;
        --shift-direction)  SHIFT_DIRECTION="$2"; shift 2 ;;
        --valid-iters)      VALID_ITERS="$2"; shift 2 ;;
        --foundation-model) FOUNDATION_MODEL="$2"; shift 2 ;;
        --only)             ONLY="$2"; shift 2 ;;
        --from)             FROM="$2"; shift 2 ;;
        --no-vis)           NO_VIS=1; shift ;;
        --clean)            CLEAN_INTERMEDIATE=1; shift ;;
        --lean)             NO_VIS=1; CLEAN_INTERMEDIATE=1; shift ;;
        --vis)              VIS=1; shift ;;
        --filter-max-depth-error) FILTER_MAX_DEPTH_ERR="$2"; shift 2 ;;
        --filter-max-reproj)      FILTER_MAX_REPROJ="$2"; shift 2 ;;
        --filter-min-consistent)  FILTER_MIN_CONSISTENT="$2"; shift 2 ;;
        --filter-num-src)         FILTER_NUM_SRC="$2"; shift 2 ;;
        --filter-min-ncc)         FILTER_MIN_NCC="$2"; shift 2 ;;
        --filter-no-ncc)          FILTER_USE_NCC=0; shift ;;
        *)                  die "未知参数: $1（用 --help 看用法）" ;;
    esac
done

[[ -n "${DATA_DIR}" ]] || die "必须指定 --data-dir"
DATA_DIR="$(cd "${DATA_DIR}" && pwd)" || die "数据集目录不存在: ${DATA_DIR}"
WORK_DIR="${WORK_DIR:-${DATA_DIR}/stereo_depth}"
derive_paths

#---------------------------------------------------------- 自动找 config ----
if [[ -z "${CONFIG_PATH}" ]]; then
    CONFIG_PATH="$(find "${DATA_DIR}/outputs" -name config.yml -printf '%T@ %p\n' 2>/dev/null \
                   | sort -rn | head -1 | cut -d' ' -f2-)"
    [[ -n "${CONFIG_PATH}" ]] || die "在 ${DATA_DIR}/outputs 下找不到 config.yml，请用 --config 指定"
    c_info "自动选用最新的模型: ${CONFIG_PATH}"
fi
DATAPARSER_TRANSFORMS="$(dirname "${CONFIG_PATH}")/dataparser_transforms.json"

#------------------------------------------------------------------ 预检 ----
[[ -f "${TRANSFORMS_JSON}" ]]       || die "找不到 ${TRANSFORMS_JSON}"
[[ -f "${CONFIG_PATH}" ]]           || die "找不到 ${CONFIG_PATH}"
[[ -f "${DATAPARSER_TRANSFORMS}" ]] || die "找不到 ${DATAPARSER_TRANSFORMS}（基线换算需要它的 scale）"
[[ -x "${NERF_PY}" ]]               || die "找不到 nerfstudio 环境的 python: ${NERF_PY}"
[[ -x "${STEREO_PY}" ]]             || die "找不到 nevstereo 环境的 python: ${STEREO_PY}"
[[ -f "${FOUNDATION_MODEL}" ]]      || die "找不到 FoundationStereo 权重: ${FOUNDATION_MODEL}"
[[ -f "${FOUNDATION_DIR}/scripts/batch_process.py" ]] \
                                    || die "找不到 FoundationStereo 代码: ${FOUNDATION_DIR}"

case "${SHIFT_DIRECTION}" in
    up|down|left|right|x|-x|y|-y) ;;
    *) die "--shift-direction 只能是 up/down/left/right（也接受 x/-x/y/-y）" ;;
esac

#---------------------------------------------------------------- 步骤选择 ----
should_run() {
    local step="$1"
    if [[ -n "${ONLY}" ]]; then
        [[ ",${ONLY}," == *",${step},"* ]]
        return
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

if [[ -n "${ONLY}" ]]; then
    IFS=',' read -ra _req <<< "${ONLY}"
    for s in "${_req[@]}"; do
        [[ " ${ALL_STEPS[*]} " == *" ${s} "* ]] || die "--only 里有未知步骤: ${s}"
    done
fi
if [[ -n "${FROM}" ]]; then
    [[ " ${ALL_STEPS[*]} " == *" ${FROM} "* ]] || die "--from 里有未知步骤: ${FROM}"
fi

#------------------------------------------------------------------ 概要 ----
cat <<EOF

  数据集      ${DATA_DIR}
  模型        ${CONFIG_PATH}
  产物        ${WORK_DIR}
  基线        ${SHIFT} (方向 ${SHIFT_DIRECTION})
  FS 权重     ${FOUNDATION_MODEL}
  迭代        ${VALID_ITERS}
  深度 PNG    $([[ "${NO_VIS}" == "1" ]] && echo "不生成 (--no-vis)" || echo "生成")
  中间产物    $([[ "${CLEAN_INTERMEDIATE}" == "1" ]] && echo "跑完删除 (--clean)" || echo "保留")
  过滤可视化  $([[ "${VIS}" == "1" && "${NO_VIS}" != "1" ]] && echo "生成 (--vis)" || echo "不生成")
EOF

mkdir -p "${CAM_DIR}" "${RENDER_LEFT}" "${RENDER_RIGHT}" \
         "${ROT_LEFT}" "${ROT_RIGHT}" "${RAW_DEPTH_DIR}" "${DEPTH_DIR}"

#================================================================== 步骤 ====

# 1. 从训练好的模型里导出训练视角的相机位姿
if should_run export; then
    c_step "export — 导出相机位姿"
    "${NERF_PY}" "${LIB_DIR}/export_poses.py" \
        --load-config "${CONFIG_PATH}" \
        --output-dir "${CAM_DIR}" \
        --combine-train-eval \
        --reference-transforms "${TRANSFORMS_JSON}"
    [[ -f "${POSES_LEFT}" ]] || die "位姿导出失败，没有生成 ${POSES_LEFT}"
    c_ok "$(basename "${POSES_LEFT}")"
fi

# 2. 平移出右目位姿
if should_run shift; then
    c_step "shift — 生成右目位姿"
    "${STEREO_PY}" "${LIB_DIR}/stereo_shift.py" \
        --input "${POSES_LEFT}" \
        --output "${POSES_RIGHT}" \
        --shift "${SHIFT}" \
        --shift-direction "${SHIFT_DIRECTION}"
    c_ok "$(basename "${POSES_RIGHT}")"
fi

# 3. 转成 nerfstudio 的 camera_path 格式
if should_run campath; then
    c_step "campath — 转 camera_path 格式"
    for side in left right; do
        [[ "${side}" == "left" ]] && { src="${POSES_LEFT}";  dst="${CAMPATH_LEFT}"; } \
                                  || { src="${POSES_RIGHT}"; dst="${CAMPATH_RIGHT}"; }
        "${STEREO_PY}" "${LIB_DIR}/make_camera_path.py" \
            "${src}" "${TRANSFORMS_JSON}" "${dst}" \
            --config "${CONFIG_PATH}" --verbose
    done
    c_ok "camera_path_{left,right}.json"
fi

# 4. 渲染左目（训练视角）和右目
if should_run render; then
    c_step "render — 渲染左右目"
    for side in left right; do
        [[ "${side}" == "left" ]] && { cp_json="${CAMPATH_LEFT}";  out="${RENDER_LEFT}"; } \
                                  || { cp_json="${CAMPATH_RIGHT}"; out="${RENDER_RIGHT}"; }
        c_info "渲染 ${side} -> ${out}"
        "${NS_RENDER}" camera-path \
            --load-config "${CONFIG_PATH}" \
            --camera-path-filename "${cp_json}" \
            --output-path "${out}" \
            --output-format images \
            --image-format png
    done
    n_l=$(find "${RENDER_LEFT}"  -name '*.png' | wc -l)
    n_r=$(find "${RENDER_RIGHT}" -name '*.png' | wc -l)
    [[ "${n_l}" -gt 0 && "${n_l}" -eq "${n_r}" ]] \
        || die "渲染结果异常: 左 ${n_l} 张, 右 ${n_r} 张"
    c_ok "左右各 ${n_l} 张"
fi

# 5. 旋转到立体匹配要求的朝向
if should_run rotate; then
    c_step "rotate — 旋转图像"
    "${STEREO_PY}" "${LIB_DIR}/rotate_images.py" \
        --left-dir "${RENDER_LEFT}"   --right-dir "${RENDER_RIGHT}" \
        --output-left-dir "${ROT_LEFT}" --output-right-dir "${ROT_RIGHT}" \
        --shift-direction "${SHIFT_DIRECTION}"
    c_ok "rotated/{left,right}"
fi

# 6. 生成 K.txt（内参 + 换算成真实尺度的基线）
if should_run intrinsic; then
    c_step "intrinsic — 生成 K.txt"
    "${STEREO_PY}" "${LIB_DIR}/make_intrinsics.py" \
        --transforms-json "${TRANSFORMS_JSON}" \
        --dataparser-transforms "${DATAPARSER_TRANSFORMS}" \
        --shift "${SHIFT}" \
        --shift-direction "${SHIFT_DIRECTION}" \
        --output "${INTRINSIC_FILE}" \
        --output-original "${INTRINSIC_ORIGINAL}"
    c_ok "$(basename "${INTRINSIC_FILE}")"
fi

# 7. FoundationStereo 推理
if should_run stereo; then
    c_step "stereo — FoundationStereo 推理"
    # batch_process.py 用 sys.path.append('..') 找 core/Utils，所以要在仓库根目录跑
    ( cd "${FOUNDATION_DIR}" && "${STEREO_PY}" scripts/batch_process.py \
        --left_dir "${ROT_LEFT}" \
        --right_dir "${ROT_RIGHT}" \
        --intrinsic_file "${INTRINSIC_FILE}" \
        --ckpt_dir "${FOUNDATION_MODEL}" \
        --out_dir "${RAW_DEPTH_DIR}" \
        --valid_iters "${VALID_ITERS}" )
    n=$(find "${RAW_DEPTH_DIR}" -name '*.npy' | wc -l)
    [[ "${n}" -gt 0 ]] || die "FoundationStereo 没有输出任何深度图"
    c_ok "${n} 张深度图 -> ${RAW_DEPTH_DIR}"
fi

# 8. 深度图转回原始朝向
if should_run depth; then
    c_step "depth — 深度后处理"
    png_flag=()
    [[ "${NO_VIS}" == "1" ]] && png_flag+=(--no-png)
    "${STEREO_PY}" "${LIB_DIR}/depth_postprocess.py" \
        --depth-in "${RAW_DEPTH_DIR}" \
        --depth-out "${DEPTH_DIR}" \
        --shift-direction "${SHIFT_DIRECTION}" \
        --transforms-json "${TRANSFORMS_JSON}" \
        "${png_flag[@]}"
    c_ok "${DEPTH_DIR}"

    # 中间产物清理只在 depth 成功之后做，否则删了还得从 stereo 重跑
    if [[ "${CLEAN_INTERMEDIATE}" == "1" ]]; then
        freed=$(du -sh "${WORK_DIR}/rotated" "${RAW_DEPTH_DIR}" 2>/dev/null | awk '{print $1}' | tr '\n' ' ')
        rm -rf "${WORK_DIR}/rotated" "${RAW_DEPTH_DIR}"
        c_ok "已删除中间产物 rotated/ raw_depth/ (${freed})"
    fi
fi

# 9. 多视图几何一致性过滤
if should_run filter; then
    c_step "filter — 多视图几何一致性过滤"
    vis_flag=()
    [[ "${VIS}" == "1" && "${NO_VIS}" != "1" ]] && vis_flag+=(--vis-dir "${VIS_DIR}")
    # NCC 要用渲染图；render/left 被 --clean 删掉的话就只能退回几何三项
    ncc_flag=()
    if [[ "${FILTER_USE_NCC}" == "1" && -d "${RENDER_LEFT}" ]]; then
        ncc_flag+=(--image-dir "${RENDER_LEFT}"
                   --ncc-window "${FILTER_NCC_WINDOW}"
                   --min-ncc "${FILTER_MIN_NCC}"
                   --min-texture-std "${FILTER_MIN_TEXTURE_STD}")
    fi
    "${STEREO_PY}" "${LIB_DIR}/filter_depth.py" \
        --transforms-json "${TRANSFORMS_JSON}" \
        --depth-dir "${DEPTH_DIR}" \
        --output-dir "${DEPTH_FILTERED_DIR}" \
        --num-src "${FILTER_NUM_SRC}" \
        --max-reproj-error "${FILTER_MAX_REPROJ}" \
        --max-depth-error "${FILTER_MAX_DEPTH_ERR}" \
        --min-triangulation-angle "${FILTER_MIN_TRI_ANGLE}" \
        --min-num-consistent "${FILTER_MIN_CONSISTENT}" \
        "${ncc_flag[@]}" "${vis_flag[@]}"
    c_ok "${DEPTH_FILTERED_DIR}"
fi

c_step "完成"
cat <<EOF

  渲染的训练视角   ${RENDER_LEFT}
  渲染的双目视角   ${RENDER_RIGHT}
  深度图           ${DEPTH_DIR}$([[ "${NO_VIS}" == "1" ]] && echo "  (.npy)" || echo "  (.npy + 16 位 .png)")
  过滤后深度       ${DEPTH_FILTERED_DIR}

  占用             $(du -sh "${WORK_DIR}" 2>/dev/null | cut -f1)
EOF
