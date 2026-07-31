import os
import sys
import glob
import cv2
import numpy as np
import torch


def main():
    # 路径设置
    input_dir = "/mnt/h/RGBD-500/images"
    repo_root = "/mnt/f/algorithm_F/Depth-Anything-V2"
    checkpoints_dir = os.path.join(repo_root, "checkpoints")

    # 模型配置
    encoder = "vitl"  # 可选: vits | vitb | vitl | vitg
    input_size = 518
    checkpoint_path = os.path.join(checkpoints_dir, f"depth_anything_v2_{encoder}.pth")

    # 将仓库加入路径，便于 import 包
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except Exception as e:
        raise RuntimeError(
            f"无法导入 Depth-Anything-V2 包，请确认路径是否正确: {repo_root}. 错误: {e}"
        )

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"未找到权重文件: {checkpoint_path}\n"
            f"请从 README 提供的链接下载并放到该目录下。"
        )

    device = (
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"使用设备: {device}")

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    print("加载 Depth Anything V2 模型…")
    model = DepthAnythingV2(**model_configs[encoder])
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()
    print("模型加载完成。")

    # 收集待处理的图像文件（递归）
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    all_paths = glob.glob(os.path.join(input_dir, "**", "*"), recursive=True)
    img_files = [p for p in all_paths if os.path.isfile(p) and os.path.splitext(p)[1].lower() in exts]
    img_files.sort()

    print(f"在 {input_dir} 找到 {len(img_files)} 张图像。")

    eps = 1e-8  # 防止除零

    # 统一输出目录
    out_dir = "/mnt/h/RGBD-500/stereo/depth_DAM"
    os.makedirs(out_dir, exist_ok=True)
    for idx, img_path in enumerate(img_files, start=1):
        print(f"进度 {idx}/{len(img_files)}: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            print(f"  警告: 无法读取图像，跳过: {img_path}")
            continue

        # 模型输出为视差(disparity)
        disparity = model.infer_image(img, input_size)

        # 转深度(depth = 1 / disparity)
        depth = 1.0 / (disparity + eps)

        # 保存为 .npy，输出到指定目录，文件名与输入同名
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(out_dir, base_name + ".npy")
        np.save(out_path, depth)

        # 简要统计
        try:
            print(
                f"  视差范围: [{float(disparity.min()):.6f}, {float(disparity.max()):.6f}] | "
                f"深度范围: [{float(depth.min()):.6f}, {float(depth.max()):.6f}] -> 保存: {out_path}"
            )
        except Exception:
            print(f"  已保存: {out_path}")


if __name__ == "__main__":
    main()


