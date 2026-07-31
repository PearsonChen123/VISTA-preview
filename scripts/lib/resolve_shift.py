#!/usr/bin/env python3
"""把"目标视差占图像宽度的百分比"换算成基线长度。

**为什么需要参考深度**

视差和深度是反比关系：

    disparity_px = fx * baseline / Z

同一个基线，近处物体视差大、远处小。所以"平移 10% 宽度"这种说法必须锚定一个
深度才有意义——本脚本把它定义为**在场景参考深度处**达到该视差。

参考深度从 COLMAP 稀疏点云统计出来，不用渲染，读文件 + numpy 投影，毫秒级。
（也可以在配置里直接给 stereo.reference_depth 写死。）

**输出的是归一化坐标系下的 shift**

位姿平移发生在 nerfstudio 的归一化空间，而深度是真实尺度，所以：

    baseline_metric = frac * W * Z_ref / fx
    shift_norm      = baseline_metric * scale

下游 make_intrinsics.py 再用 baseline = shift_norm / scale 换回去，闭合。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colmap_model import find_sparse_dir, read_images_binary, read_points3D_binary


def scene_depth_from_colmap(colmap_dir: Path, percentile: float):
    """从稀疏点云统计场景深度。

    对每张注册图像，把它实际看到的 3D 点投到相机系取 z 深度——
    用可见性而不是全部点，这样统计的是"相机真正拍到的东西有多远"。
    """
    sparse = find_sparse_dir(colmap_dir)
    if not (sparse / "points3D.bin").is_file():
        raise FileNotFoundError(f"{sparse} 下没有 points3D.bin")

    images = read_images_binary(sparse / "images.bin")
    points = read_points3D_binary(sparse / "points3D.bin")
    if not points:
        raise ValueError(f"{sparse}/points3D.bin 是空的（稀疏重建没出点）")

    xyz = {pid: p.xyz for pid, p in points.items()}
    depths = []
    for im in images.values():
        ids = im.point3D_ids
        ids = ids[ids >= 0]
        if ids.size == 0:
            continue
        pts = np.array([xyz[i] for i in ids if i in xyz])
        if pts.size == 0:
            continue
        w2c = im.world_to_camera()
        z = (pts @ w2c[:3, :3].T + w2c[:3, 3])[:, 2]      # OpenCV: z 就是深度
        depths.append(z[z > 0])

    if not depths:
        raise ValueError("没有任何图像看到有效的 3D 点")
    allz = np.concatenate(depths)
    return float(np.percentile(allz, percentile)), allz


def main():
    ap = argparse.ArgumentParser(
        description="目标视差百分比 -> 基线（归一化坐标系下的 shift）")
    ap.add_argument("--transforms-json", required=True, type=Path)
    ap.add_argument("--dataparser-transforms", required=True, type=Path)
    ap.add_argument("--mode", required=True, choices=["pixels", "baseline"])
    ap.add_argument("--shift", type=float, default=0.2,
                    help="mode=baseline 时直接用这个值（归一化坐标系）")
    ap.add_argument("--shift-pixels", type=float, default=0.1,
                    help="mode=pixels 时的目标视差，占图像宽度的比例（0.1 = 10%%）")
    ap.add_argument("--colmap-dir", type=Path, default=None,
                    help="用于统计场景深度的 COLMAP 目录")
    ap.add_argument("--reference-depth", type=float, default=None,
                    help="直接指定参考深度（米），给了就不去读点云")
    ap.add_argument("--percentile", type=float, default=25.0,
                    help="从点云深度分布取哪个分位作参考（默认 25）。"
                         "取近处而非中位数，是为了把近平面视差压在网络好处理的范围内")
    ap.add_argument("--quiet", action="store_true", help="只输出最终数值")
    args = ap.parse_args()

    log = (lambda *a: None) if args.quiet else (
        lambda *a: print("[resolve_shift]", *a, file=sys.stderr))

    if args.mode == "baseline":
        log(f"mode=baseline，直接用 shift={args.shift}")
        print(args.shift)
        return

    tf = json.loads(args.transforms_json.read_text(encoding="utf-8"))
    fx, W = float(tf["fl_x"]), int(tf["w"])
    scale = float(json.loads(
        args.dataparser_transforms.read_text(encoding="utf-8"))["scale"])

    if args.reference_depth is not None:
        Z = args.reference_depth
        log(f"参考深度取配置指定值 {Z:.4f} m")
    else:
        if args.colmap_dir is None:
            raise SystemExit(
                "[resolve_shift] mode=pixels 需要参考深度：\n"
                "  要么让 colmap 步骤跑过（从稀疏点云自动统计），\n"
                "  要么在 config.json 里写 stereo.reference_depth")
        try:
            Z, allz = scene_depth_from_colmap(args.colmap_dir, args.percentile)
        except (FileNotFoundError, ValueError) as e:
            raise SystemExit(
                f"[resolve_shift] 无法从 {args.colmap_dir} 统计场景深度: {e}\n"
                f"  请在 config.json 里写 stereo.reference_depth")
        log(f"稀疏点云 {len(allz):,} 个观测，深度分布 "
            f"5%={np.percentile(allz,5):.3f}  50%={np.percentile(allz,50):.3f}  "
            f"95%={np.percentile(allz,95):.3f} m")
        log(f"参考深度取 {args.percentile:g}% 分位 = {Z:.4f} m")

    target_px = args.shift_pixels * W
    baseline = target_px * Z / fx
    shift_norm = baseline * scale

    log(f"目标视差 {args.shift_pixels:.1%} x {W}px = {target_px:.1f} px @ {Z:.3f} m")
    log(f"基线 = {target_px:.1f} * {Z:.3f} / {fx:.1f} = {baseline:.6f} m")
    log(f"归一化 shift = {baseline:.6f} * {scale:.6f} = {shift_norm:.6f}")
    if args.reference_depth is None:
        log("该基线下各深度的实际视差:")
        for p in (5, 25, 50, 95):
            zp = float(np.percentile(allz, p))
            d = fx * baseline / zp
            mark = "  <- 锚点" if abs(p - args.percentile) < 1e-6 else ""
            log(f"    {p:>2}% 分位 Z={zp:7.3f} m  ->  {d:6.1f} px ({d/W:5.1%} 宽度){mark}")
    else:
        log("（近处视差更大、远处更小，这是立体几何本身的性质）")

    print(f"{shift_norm:.9f}")


if __name__ == "__main__":
    main()
