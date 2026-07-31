"""立体几何的公共定义。

**为什么需要旋转**

nerfstudio 的相机坐标系是 x 右 / y 上 / z 后。我们通过平移相机来造双目，
但立体匹配网络（FoundationStereo 等）要求视差是**水平**的、且右图内容相对左图左移。
所以当基线不是沿 -x 时，必须把渲染出来的图旋转到那个约定下，
跑完再把深度图转回原始朝向。

旧版实现把这张对应表抄了三份（旋转图像的脚本靠 shift 正负判断、
生成 K.txt 的内联脚本一份、深度后处理又一份且注释成"与前者相反"），
三处口径必须手工保持一致，很容易改漏。这里统一成一张表。
"""

import cv2
import numpy as np

# 用户面的方向名，描述的是**右目图像相对左目、内容往哪边移**
# （在最终转回原始朝向之后看到的效果）。
#
# 这比原来的 x / -x / y / -y 直观：那套是相机在自身坐标系里往哪个轴平移，
# 而相机往下移，画面内容是往上跑的，方向正好相反，很容易搞混。
#
# 对应关系是投影几何算出来并且和实际渲染对过的：
#   right -> +x 轴 -> 内容向右      up   -> +y 轴 -> 内容向上
#   left  -> -x 轴 -> 内容向左      down -> -y 轴 -> 内容向下
DIRECTION_ALIASES = {
    "right": "x",  "x": "x",
    "left": "-x",  "-x": "-x",
    "up": "y",     "y": "y",
    "down": "-y",  "-y": "-y",
}

# 命令行推荐用的写法（x/-x/y/-y 仍然接受，只是不再宣传）
DIRECTION_CHOICES = ("up", "down", "left", "right", "x", "-x", "y", "-y")

# 轴方向 -> 渲染图/内参需要施加的旋转
ROTATION_FOR_SHIFT = {
    "x": "180",
    "-x": "none",
    "y": "90cc",
    "-y": "90c",
}


def normalize_direction(direction: str) -> str:
    """把 up/down/left/right 或 x/-x/y/-y 统一成轴形式。"""
    key = str(direction).strip().lower()
    if key not in DIRECTION_ALIASES:
        raise ValueError(
            f"未知的方向 {direction!r}，可选: {', '.join(DIRECTION_CHOICES)}")
    return DIRECTION_ALIASES[key]

# 逆旋转，用于把深度图转回原始朝向
INVERSE_ROTATION = {
    "none": "none",
    "180": "180",
    "90cc": "90c",
    "90c": "90cc",
}

def rotation_for(shift_direction: str) -> str:
    """方向 -> 图像/内参的旋转类型。接受 up/down/left/right 或 x/-x/y/-y。"""
    return ROTATION_FOR_SHIFT[normalize_direction(shift_direction)]


def inverse_of(rotation: str) -> str:
    """取逆旋转。"""
    if rotation not in INVERSE_ROTATION:
        raise ValueError(f"未知的旋转类型 {rotation!r}")
    return INVERSE_ROTATION[rotation]


def rotate_image(image: np.ndarray, rotation: str) -> np.ndarray:
    """旋转 HxW 或 HxWxC 的图像（走 cv2，用于 uint8 图片）。"""
    if rotation == "none":
        return image
    codes = {
        "90cc": cv2.ROTATE_90_COUNTERCLOCKWISE,
        "90c": cv2.ROTATE_90_CLOCKWISE,
        "180": cv2.ROTATE_180,
    }
    return cv2.rotate(image, codes[rotation])


def rotate_array(array: np.ndarray, rotation: str) -> np.ndarray:
    """旋转 2D 数组（走 numpy，用于 float 深度/视差图）。

    np.rot90 的 k>0 是逆时针，与 cv2 的 COUNTERCLOCKWISE 同向。
    """
    if rotation == "none":
        return array
    k = {"90cc": 1, "90c": -1, "180": 2}[rotation]
    return np.ascontiguousarray(np.rot90(array, k=k))


def rotate_intrinsics(K: np.ndarray, rotation: str, width: int, height: int) -> np.ndarray:
    """把 3x3 内参矩阵旋转到与旋转后图像一致的朝向。

    width/height 是**旋转前**的图像尺寸。
    90 度旋转会交换 fx/fy，并按新的图像边界重算主点。
    """
    if rotation == "none":
        return K.copy()

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    if rotation == "90cc":
        return np.array([[fy, 0, cy],
                         [0, fx, width - 1.0 - cx],
                         [0, 0, 1]])
    if rotation == "90c":
        return np.array([[fy, 0, height - 1.0 - cy],
                         [0, fx, cx],
                         [0, 0, 1]])
    # 180
    return np.array([[fx, 0, width - 1.0 - cx],
                     [0, fy, height - 1.0 - cy],
                     [0, 0, 1]])


def rotated_size(width: int, height: int, rotation: str):
    """旋转后的 (width, height)。"""
    if rotation in ("90cc", "90c"):
        return height, width
    return width, height


def read_intrinsic_file(path):
    """读 FoundationStereo 格式的 K 文件：第一行 9 个内参，第二行基线。"""
    with open(path, "r") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"{path} 格式不对：需要两行（K 的 9 个元素 + 基线）")
    K = np.array([float(v) for v in lines[0].split()], dtype=float).reshape(3, 3)
    return K, float(lines[1])


def write_intrinsic_file(path, K: np.ndarray, baseline: float):
    """写 FoundationStereo 格式的 K 文件。"""
    with open(path, "w") as f:
        f.write(" ".join(str(v) for v in K.flatten()) + "\n")
        f.write(f"{baseline}\n")
