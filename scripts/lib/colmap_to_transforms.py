#!/usr/bin/env python3
"""COLMAP 稀疏模型 -> nerfstudio 的 transforms.json。

**为什么用 undistorted**

直接拿 mapper 的输出（OPENCV / RADIAL 等带畸变的模型）也能写出 transforms.json，
但畸变参数要一路传到渲染和立体匹配，任何一环没处理就会错。
`colmap image_undistorter` 会把图像和内参一起转成无畸变的 PINHOLE 模型，
下游就只剩 fx/fy/cx/cy 四个数，干净得多。

所以默认要求输入是 undistorted 的模型；带畸变的模型会明确报错并提示怎么做，
除非显式加 --allow-distorted（这时畸变系数会写进 transforms.json，
由 nerfstudio 自己处理，但立体渲染那条链路不保证正确）。

**坐标约定**

COLMAP 存的是 world-to-camera，OpenCV 约定（x 右, y 下, z 前）。
nerfstudio 要 camera-to-world，OpenGL/Blender 约定（x 右, y 上, z 后）：

    transform_matrix = inv(w2c) @ diag(1, -1, -1, 1)
"""

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from colmap_model import find_sparse_dir, read_cameras_binary, read_images_binary

# OpenCV 相机系 -> OpenGL 相机系
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def build_transforms(cameras, images, image_dir: Path, out_path: Path,
                     allow_distorted: bool = False, downscale: float = 1.0):
    distorted = [c for c in cameras.values() if not c.is_undistorted]
    if distorted and not allow_distorted:
        models = sorted({c.model for c in distorted})
        raise SystemExit(
            f"[colmap_to_transforms] 模型带畸变（{', '.join(models)}），不是 undistorted。\n"
            f"  先跑:  colmap image_undistorter --image_path <原图> \\\n"
            f"             --input_path <sparse/0> --output_path <undistorted>\n"
            f"  然后用 <undistorted> 作为输入。\n"
            f"  确实要用带畸变的模型请加 --allow-distorted。")

    # 按文件名排序，保证 transforms.json 的顺序稳定——
    # 下游的位姿导出、渲染、深度回写都按这个顺序对齐
    ordered = sorted(images.values(), key=lambda im: im.name)

    frames = []
    for im in ordered:
        cam = cameras[im.camera_id]
        c2w_gl = im.camera_to_world() @ CV_TO_GL
        frame = {
            "file_path": str((image_dir / im.name).as_posix()),
            "transform_matrix": c2w_gl.tolist(),
            "colmap_im_id": int(im.id),
        }
        # 多相机时把内参写进每一帧
        if len(cameras) > 1:
            fx, fy, cx, cy = (v / downscale for v in cam.pinhole_intrinsics())
            frame.update({"fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy,
                          "w": round(cam.width / downscale),
                          "h": round(cam.height / downscale)})
        frames.append(frame)

    # 单相机时内参写在顶层
    ref = cameras[ordered[0].camera_id]
    # 用降采样过的图（images_4 等）时内参要同比缩放，否则投影全错
    fx, fy, cx, cy = (v / downscale for v in ref.pinhole_intrinsics())
    k1, k2, p1, p2 = ref.distortion()

    out = {
        "camera_model": "OPENCV",
        "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy,
        "w": round(ref.width / downscale), "h": round(ref.height / downscale),
        "k1": k1, "k2": k2, "p1": p1, "p2": p2,
        "frames": frames,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out, ref


def main():
    ap = argparse.ArgumentParser(description="COLMAP 稀疏模型 -> transforms.json")
    ap.add_argument("--colmap-dir", required=True, type=Path,
                    help="含 cameras.bin/images.bin 的目录，或它的上级"
                         "（会自动找 sparse/ 或 sparse/0/）")
    ap.add_argument("--image-dir", required=True, type=Path,
                    help="图像目录。undistorted 流程下是 <undistorted>/images")
    ap.add_argument("--output", required=True, type=Path, help="transforms.json 路径")
    ap.add_argument("--downscale", type=float, default=1.0,
                    help="图像相对 COLMAP 模型的降采样倍数。用 images_4 就填 4——"
                         "内参会同比缩放，否则投影全错")
    ap.add_argument("--allow-distorted", action="store_true",
                    help="允许带畸变的模型（不推荐，见模块开头说明）")
    args = ap.parse_args()

    sparse = find_sparse_dir(args.colmap_dir)
    cameras = read_cameras_binary(sparse / "cameras.bin")
    images = read_images_binary(sparse / "images.bin")
    if not images:
        raise SystemExit(f"[colmap_to_transforms] {sparse} 里没有注册成功的图像")

    # transforms.json 里的 file_path 用相对路径，整个目录搬走也不会失效
    try:
        rel = args.image_dir.resolve().relative_to(args.output.resolve().parent)
    except ValueError:
        rel = args.image_dir.resolve()

    out, ref = build_transforms(cameras, images, Path(rel), args.output,
                                args.allow_distorted, args.downscale)

    missing = [f["file_path"] for f in out["frames"]
               if not (args.output.parent / f["file_path"]).exists()][:3]

    print(f"[colmap_to_transforms] 稀疏模型  {sparse}")
    print(f"[colmap_to_transforms] 相机模型  {ref.model} ({len(cameras)} 个相机)")
    print(f"[colmap_to_transforms] 降采样    {args.downscale}x")
    print(f"[colmap_to_transforms] 内参      {out['w']}x{out['h']}  "
          f"fx={out['fl_x']:.3f} fy={out['fl_y']:.3f} "
          f"cx={out['cx']:.3f} cy={out['cy']:.3f}")
    print(f"[colmap_to_transforms] 注册图像  {len(out['frames'])} 张")
    print(f"[colmap_to_transforms] 写出      {args.output}")
    if missing:
        print(f"[colmap_to_transforms] 警告: 有图像文件找不到，例如 {missing}")


if __name__ == "__main__":
    main()
