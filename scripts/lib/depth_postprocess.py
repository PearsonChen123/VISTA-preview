#!/usr/bin/env python3
"""把 FoundationStereo 输出的深度图转回原始朝向，并存成 nerfstudio 能读的 16 位 PNG。

注意 batch_process.py 存出来的 .npy **已经是深度**（米），不是视差——
它内部做了 depth = fx * baseline / disp。旧版脚本参数名叫 --disp-dir 有误导，
这里改叫 --depth-in。

图像在送进网络前被旋转过，所以深度图要转回去（逆旋转，见 common.py）。

同时可选地把 depth_file_path 写回 transforms.json，供 depth-nerfacto 之类使用。
"""

import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DIRECTION_CHOICES, inverse_of, rotate_array, rotation_for

# 16 位 PNG 的量程：DEPTH_MAX 米映射到 65535
DEPTH_MAX_METERS = 400.0


def _process_one(job):
    src, out_dir, rotation, save_png = job
    src = Path(src)
    try:
        depth = np.load(src).astype(np.float32)
        depth = rotate_array(depth, rotation)

        npy_path = Path(out_dir) / src.name
        np.save(npy_path, depth)

        if save_png:
            scaled = np.clip(depth * (65535.0 / DEPTH_MAX_METERS), 0, 65535).astype(np.uint16)
            cv2.imwrite(str(npy_path.with_suffix(".png")), scaled)

        valid = depth[np.isfinite(depth) & (depth > 0)]
        stats = (float(np.min(valid)), float(np.median(valid)), float(np.max(valid))) if valid.size else None
        return {"stem": src.stem, "png": npy_path.with_suffix(".png").name, "stats": stats}
    except Exception as exc:                                   # noqa: BLE001
        return {"stem": src.stem, "error": str(exc)}


def update_transforms(transforms_json: Path, depth_dir: Path, results):
    """把 depth_file_path 写回 transforms.json 的对应 frame。

    两种对齐方式：

    1. 按 file_path 的 stem 匹配。名字对得上时最稳。
    2. 按顺序对齐。ns-render 输出的是 5 位数字（``00000.png``），而数据集里
       常常是 4 位（``0000.png``），stem 根本对不上。但整条 pipeline 的顺序是
       有保证的——camera_path 由 transforms_train.json 生成，后者又是用
       ``--reference-transforms transforms.json`` 按原顺序对齐过的，
       所以第 i 张渲染图就对应第 i 个 frame。

    先试 stem，不行再退回按顺序，并且要求数量完全一致，避免默默错位。
    """
    data = json.loads(transforms_json.read_text(encoding="utf-8"))
    frames = data.get("frames")
    if not frames:
        raise ValueError(f"{transforms_json} 里没有 frames")

    rel_dir = os.path.relpath(depth_dir.resolve(), transforms_json.resolve().parent)
    results = sorted(results, key=lambda r: r["stem"])

    def _assign(frame, item):
        frame["depth_file_path"] = os.path.join(rel_dir, item["png"]).replace("\\", "/")

    by_stem = {Path(f.get("file_path", "")).stem: f for f in frames}
    matched = [(by_stem[r["stem"]], r) for r in results if r["stem"] in by_stem]

    if matched:
        mode = "按文件名"
    elif len(results) == len(frames):
        mode = "按顺序"
        matched = list(zip(frames, results))
    else:
        print(f"[depth_postprocess] 警告: 深度图 {len(results)} 张与 frame {len(frames)} 个"
              f"数量不符，且文件名对不上，跳过写回 transforms.json")
        return

    for frame, item in matched:
        _assign(frame, item)

    transforms_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[depth_postprocess] transforms.json 已更新 {len(matched)}/{len(frames)} 帧"
          f"的 depth_file_path（{mode}对齐）")


def main():
    ap = argparse.ArgumentParser(description="深度图转回原始朝向 + 存 16 位 PNG")
    ap.add_argument("--depth-in", required=True, type=Path,
                    help="FoundationStereo 的输出目录（.npy，已是深度）")
    ap.add_argument("--depth-out", required=True, type=Path, help="输出目录")
    ap.add_argument("--shift-direction", required=True, choices=list(DIRECTION_CHOICES),
                    metavar="DIR", help="up/down/left/right（也接受 x/-x/y/-y）")
    ap.add_argument("--transforms-json", type=Path,
                    help="可选：把 depth_file_path 写回这个 transforms.json")
    ap.add_argument("--no-png", action="store_true", help="只存 .npy，不存 16 位 PNG")
    args = ap.parse_args()

    files = sorted(args.depth_in.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"{args.depth_in} 里没有 .npy")

    # 图像当初被正向旋转过，深度要转回去
    rotation = inverse_of(rotation_for(args.shift_direction))
    args.depth_out.mkdir(parents=True, exist_ok=True)

    jobs = [(str(p), str(args.depth_out), rotation, not args.no_png) for p in files]
    results = []
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as pool:
        for res in tqdm(pool.map(_process_one, jobs), total=len(jobs),
                        desc=f"深度后处理 ({rotation})"):
            results.append(res)

    failed = [r for r in results if "error" in r]
    if failed:
        raise RuntimeError(f"{len(failed)} 个深度图处理失败，例如 {failed[0]['stem']}: {failed[0]['error']}")

    stats = [r["stats"] for r in results if r.get("stats")]
    if stats:
        arr = np.array(stats)
        print(f"[depth_postprocess] {len(results)} 帧完成 | 深度(米) "
              f"最小 {arr[:,0].min():.3f}  中位 {np.median(arr[:,1]):.3f}  最大 {arr[:,2].max():.3f}")
    else:
        print(f"[depth_postprocess] 警告: {len(results)} 帧里没有任何有效深度值")

    if args.transforms_json:
        update_transforms(args.transforms_json, args.depth_out, results)


if __name__ == "__main__":
    main()
