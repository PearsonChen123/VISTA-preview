#!/usr/bin/env python3
"""一条命令跑完 COLMAP：特征提取 -> 匹配 -> 稀疏重建 -> 去畸变。

原来这四步要手敲四条命令，参数散在各处。这里合成一步，并且每步都能单独跳过
（产物已存在时自动跳过，用 --overwrite 强制重跑）。

输出布局：

    <work>/colmap/
    ├── database.db
    ├── sparse/0/          mapper 的输出（带畸变的原始模型）
    └── undistorted/
        ├── images/        去畸变后的图像  <- 训练和渲染都用这份
        └── sparse/        去畸变后的模型（PINHOLE，无畸变系数）

**两个相机模型别搞混**

  --camera-model 是 **SfM 期间**用来拟合镜头的模型：
      PINHOLE  断言镜头本身无畸变（图已去过畸变、或渲染出来的图）
      OPENCV   估计畸变系数，真实相机尤其广角/鱼眼该用这个

  而交给下游（transforms.json / nerfstudio）的模型**恒为 PINHOLE**：
  带畸变的模型经 image_undistorter 之后只剩 fx/fy/cx/cy。
  这是必须的——畸变系数要一路传到训练、渲染、立体匹配，任何一环漏了就错。

  所以 SfM 用 PINHOLE 时，image_undistorter 数值上是个空操作，只是复制一份图像，
  可以用 --undistort 0 跳过省磁盘。
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

MATCHERS = {
    "exhaustive": "exhaustive_matcher",       # 无序图集，最稳但 O(n^2)
    "sequential": "sequential_matcher",       # 视频抽帧，按时序匹配，快得多
    "spatial": "spatial_matcher",             # 有 GPS/位置先验时
    "vocab_tree": "vocab_tree_matcher",       # 大规模图集
}


def run(cmd, label):
    print(f"\n\033[1;34m[colmap]\033[0m {label}")
    print("  " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd])
    if r.returncode != 0:
        raise SystemExit(f"[run_colmap] {label} 失败 (exit {r.returncode})")


def main():
    ap = argparse.ArgumentParser(description="一步跑完 COLMAP SfM + 去畸变")
    ap.add_argument("--image-dir", required=True, type=Path, help="输入图像目录")
    ap.add_argument("--work-dir", required=True, type=Path, help="COLMAP 工作目录")
    ap.add_argument("--matcher", default="exhaustive", choices=list(MATCHERS),
                    help="匹配策略。视频抽帧用 sequential 快很多（默认 exhaustive）")
    ap.add_argument("--camera-model", default="PINHOLE",
                    help="SfM 期间用的相机模型。PINHOLE=断言镜头无畸变（默认）；"
                         "真实镜头有畸变请用 OPENCV / RADIAL")
    ap.add_argument("--undistort", type=int, default=1,
                    help="是否跑 image_undistorter。PINHOLE 输入时它数值上是空操作，"
                         "只是复制一份图像；设 0 可省这份磁盘")
    ap.add_argument("--single-camera", type=int, default=1,
                    help="所有图共用一个相机内参（默认 1）")
    ap.add_argument("--use-gpu", type=int, default=1)
    ap.add_argument("--colmap-bin", default="colmap")
    ap.add_argument("--overwrite", action="store_true", help="已有产物也强制重跑")
    args = ap.parse_args()

    if shutil.which(args.colmap_bin) is None:
        raise SystemExit(f"[run_colmap] 找不到 colmap 可执行文件: {args.colmap_bin}")
    if not args.image_dir.is_dir():
        raise SystemExit(f"[run_colmap] 图像目录不存在: {args.image_dir}")

    n_img = sum(1 for p in args.image_dir.iterdir()
                if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if n_img == 0:
        raise SystemExit(f"[run_colmap] {args.image_dir} 里没有图像")

    work = args.work_dir
    db = work / "database.db"
    sparse = work / "sparse"
    undist = work / "undistorted"
    work.mkdir(parents=True, exist_ok=True)

    print(f"[run_colmap] {n_img} 张图  ->  {work}")
    print(f"[run_colmap] 匹配策略 {args.matcher}, 相机模型 {args.camera_model}")
    if args.camera_model in ("PINHOLE", "SIMPLE_PINHOLE"):
        print(f"[run_colmap] 注意: {args.camera_model} 不估计畸变，等于断言镜头本身无畸变。")
        print(f"[run_colmap]       真实镜头（尤其广角/鱼眼）请改用 OPENCV，"
              f"否则重投影误差会偏高、甚至配准失败。")

    # 1. 特征提取
    if args.overwrite or not db.exists():
        run([args.colmap_bin, "feature_extractor",
             "--database_path", db,
             "--image_path", args.image_dir,
             "--ImageReader.camera_model", args.camera_model,
             "--ImageReader.single_camera", args.single_camera,
             "--SiftExtraction.use_gpu", args.use_gpu], "特征提取")
    else:
        print(f"\n[colmap] 特征提取 —— {db} 已存在，跳过")

    # 2. 匹配
    stamp = work / ".matched"
    if args.overwrite or not stamp.exists():
        run([args.colmap_bin, MATCHERS[args.matcher],
             "--database_path", db,
             "--SiftMatching.use_gpu", args.use_gpu], f"匹配 ({args.matcher})")
        stamp.touch()
    else:
        print(f"\n[colmap] 匹配 —— 已完成，跳过")

    # 3. 稀疏重建
    if args.overwrite or not (sparse / "0" / "cameras.bin").exists():
        sparse.mkdir(parents=True, exist_ok=True)
        run([args.colmap_bin, "mapper",
             "--database_path", db,
             "--image_path", args.image_dir,
             "--output_path", sparse], "稀疏重建")
    else:
        print(f"\n[colmap] 稀疏重建 —— {sparse}/0 已存在，跳过")

    models = sorted(p for p in sparse.glob("*") if (p / "cameras.bin").exists())
    if not models:
        raise SystemExit("[run_colmap] 稀疏重建没有产出任何模型")
    if len(models) > 1:
        print(f"[run_colmap] 警告: 重建出 {len(models)} 个模型（场景可能没连通），"
              f"用最大的那个 {models[0].name}")

    # 4. 去畸变
    if not args.undistort:
        print(f"\n[colmap] 去畸变 —— 已按配置跳过，下游直接用 {models[0]}")
        print(f"[run_colmap] 完成: 稀疏模型 {models[0]}")
        return
    if args.overwrite or not (undist / "sparse" / "cameras.bin").exists():
        run([args.colmap_bin, "image_undistorter",
             "--image_path", args.image_dir,
             "--input_path", models[0],
             "--output_path", undist,
             "--output_type", "COLMAP"], "去畸变")
    else:
        print(f"\n[colmap] 去畸变 —— {undist} 已存在，跳过")

    n_out = sum(1 for p in (undist / "images").iterdir()
                if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    print(f"\n[run_colmap] 完成: {n_out}/{n_img} 张图注册成功")
    print(f"[run_colmap] 去畸变图像  {undist/'images'}")
    print(f"[run_colmap] 去畸变模型  {undist/'sparse'}")
    if n_out < n_img:
        print(f"[run_colmap] 提示: {n_img-n_out} 张没注册上。"
              f"视频抽帧的话试试 --matcher sequential，或检查重叠度/纹理")


if __name__ == "__main__":
    main()
