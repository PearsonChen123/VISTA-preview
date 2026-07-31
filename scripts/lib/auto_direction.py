#!/usr/bin/env python3
"""按相机轨迹自动选双目平移方向。

**为什么要跟着轨迹走**

NeRF 只在相机真正去过的视角附近训练充分。造双目时把虚拟相机沿**轨迹方向**平移，
落点仍在观测过的视角流形内，渲染质量接近训练视角；垂直于轨迹平移就是外推，
渲染会明显变糊、出伪影。

极端情况：轨迹全是上下运动，却选了左右平移——右目渲染的是 NeRF 从没见过的视角。

**怎么判定**

对每一帧取轨迹切向（相机中心的中心差分），投影到该帧自己的相机坐标系，
看 x 分量还是 y 分量占优，再按多数投票定一个全局方向。

**为什么是全局而不是逐帧**

下游 FoundationStereo 的 batch_process.py 整批共用一个 K.txt，而不同方向对应
不同的图像旋转和内参旋转，逐帧变方向就得逐帧一个 K 文件。所以这里只出一个
全局方向。轨迹方向变化剧烈的序列（比如绕圈）会给出低主导性警告，
那种情况建议分段跑。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# 相机在自身坐标系下的移动方向 -> 本项目的方向命名。
#
# 命名含义是"右目画面相对左目往哪边移"，而相机移动方向与画面移动方向相反：
#   相机往上移 -> 画面内容往下跑 -> 记作 down
# 这层反向关系已在 stereo_shift.py 里验证过（见 lib/common.py 的对应表）。
CAMERA_MOTION_TO_DIRECTION = {
    ("y", +1): "down",     # 相机上移
    ("y", -1): "up",       # 相机下移
    ("x", +1): "left",     # 相机右移
    ("x", -1): "right",    # 相机左移
}


def load_poses(path: Path):
    """读 transforms.json 或 export_poses.py 导出的 frame 列表，返回按名字排序的 c2w。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data["frames"] if isinstance(data, dict) else data
    key = lambda f: str(f.get("file_path", ""))            # noqa: E731
    mats = []
    for f in sorted(frames, key=key):
        m = np.asarray(f.get("transform_matrix", f.get("transform")), dtype=float)
        if m.shape == (3, 4):
            m = np.vstack([m, [0, 0, 0, 1.0]])
        mats.append(m)
    return np.stack(mats)


def analyze(c2w: np.ndarray):
    """返回 (方向, 统计信息)。"""
    if len(c2w) < 2:
        raise SystemExit("[auto_direction] 至少需要 2 帧才能算轨迹方向")

    centers = c2w[:, :3, 3]
    tangent = np.gradient(centers, axis=0)                  # 中心差分
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    if (norms < 1e-9).any():
        raise SystemExit("[auto_direction] 轨迹里有重合的相机位置，无法求切向")
    tangent /= norms

    # world -> 各帧自己的相机系
    local = np.einsum("nij,nj->ni", c2w[:, :3, :3].transpose(0, 2, 1), tangent)
    tx, ty = local[:, 0], local[:, 1]

    sum_x, sum_y = np.abs(tx).sum(), np.abs(ty).sum()
    axis = "x" if sum_x >= sum_y else "y"
    dominance = max(sum_x, sum_y) / (sum_x + sum_y)

    comp = tx if axis == "x" else ty
    sign = 1 if (comp > 0).sum() >= (comp <= 0).sum() else -1
    consistency = max((comp > 0).mean(), (comp <= 0).mean())

    direction = CAMERA_MOTION_TO_DIRECTION[(axis, sign)]
    return direction, {
        "axis": axis,
        "sign": sign,
        "dominance": float(dominance),
        "sign_consistency": float(consistency),
        "mean_abs_x": float(np.abs(tx).mean()),
        "mean_abs_y": float(np.abs(ty).mean()),
        "n_frames": len(c2w),
    }


def main():
    ap = argparse.ArgumentParser(description="按相机轨迹自动选双目平移方向")
    ap.add_argument("--poses", required=True, type=Path,
                    help="transforms.json，或 export_poses.py 导出的 frame 列表")
    ap.add_argument("--fallback", default="up",
                    help="主导性不足时退回的方向（默认 up）")
    ap.add_argument("--min-dominance", type=float, default=0.6,
                    help="轨迹方向主导性低于此值就退回 fallback（默认 0.6）")
    ap.add_argument("--quiet", action="store_true", help="只输出最终方向")
    args = ap.parse_args()

    log = (lambda *a: None) if args.quiet else (
        lambda *a: print("[auto_direction]", *a, file=sys.stderr))

    direction, st = analyze(load_poses(args.poses))

    log(f"{st['n_frames']} 帧，轨迹切向在相机系下 "
        f"|x| 均值 {st['mean_abs_x']:.3f}  |y| 均值 {st['mean_abs_y']:.3f}")
    log(f"主导轴 {st['axis']}（主导性 {st['dominance']:.1%}），"
        f"符号一致性 {st['sign_consistency']:.0%}")

    if st["dominance"] < args.min_dominance:
        log(f"主导性低于 {args.min_dominance:.0%}——轨迹方向不明确"
            f"（绕圈/自由移动的序列会这样）")
        log(f"退回配置里的方向 {args.fallback}；"
            f"这种序列建议分段跑，每段单独定方向")
        print(args.fallback)
        return

    log(f"相机沿自身 {'+' if st['sign'] > 0 else '-'}{st['axis']} 移动"
        f"  ->  方向 {direction}")
    if st["sign_consistency"] < 0.8:
        log(f"注意: 符号一致性只有 {st['sign_consistency']:.0%}，"
            f"轨迹中途折返过。方向轴是对的，正负取了多数")

    print(direction)


if __name__ == "__main__":
    main()
