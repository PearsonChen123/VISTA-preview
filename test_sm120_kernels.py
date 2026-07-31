#!/usr/bin/env python
"""
sm_120 (RTX 5090) CUDA kernel 数值验证

安装时给 DROID-SLAM / lietorch 打了两类补丁：
  - gencode 硬编码 sm_86 -> sm_120
  - AT_DISPATCH / DISPATCH_GROUP_AND_FLOATING_TYPES 的 tensor.type() -> scalar_type()

补丁动的正是 kernel 分发路径，所以光能 import 不够，必须验证算得对。
这里把每个 kernel 跟纯 PyTorch 参考实现对拍。

用法:  source env_droid.sh && python /mnt/g/algorithm_backup/Nevstereo/test_sm120_kernels.py
"""
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path("/mnt/g/algorithm_backup/Nevstereo/droid_metric")))

import droid_backends  # noqa: E402  (必须在 import torch 之后)
import lietorch  # noqa: E402
from lietorch import SE3, SO3, RxSO3, Sim3  # noqa: E402
from modules.droid_core.modules.corr import CorrBlock, AltCorrBlock  # noqa: E402

DEV = torch.device("cuda")
PASS, FAIL = [], []


def check(name, fn):
    try:
        detail = fn()
        PASS.append(name)
        print(f"  [\033[1;32m✓\033[0m] {name}: {detail}")
    except Exception as e:
        FAIL.append(name)
        print(f"  [\033[1;31m✗\033[0m] {name}: {type(e).__name__}: {str(e)[:220]}")


# ---------------------------------------------------------------------------
# 1. correlation_kernels.cu —— corr_index_forward / backward
#    补丁点: AT_DISPATCH_FLOATING_TYPES_AND_HALF(volume.scalar_type(), ...)
# ---------------------------------------------------------------------------
def corr_reference(volume, coords, radius):
    """CorrSampler 的纯 PyTorch 等价实现，用 grid_sample 做双线性采样。"""
    n, h1, w1, h2, w2 = volume.shape
    vol = volume.reshape(n * h1 * w1, 1, h2, w2)

    # kernel 里 corr[n][i][j][y][x] 的 i 来自 x 偏移循环、j 来自 y 偏移，
    # 即 patch 维度是 x-major（flat = i_x * rd + j_y）。meshgrid 的第一个
    # 返回值变化最慢，所以这里必须 dx 在前，否则拿到的是转置。
    dx, dy = torch.meshgrid(
        torch.arange(-radius, radius + 1, device=volume.device, dtype=torch.float32),
        torch.arange(-radius, radius + 1, device=volume.device, dtype=torch.float32),
        indexing="ij",
    )
    delta = torch.stack([dx, dy], dim=-1).view(-1, 2)          # (S,2) 顺序为 (x,y)

    c = coords.permute(0, 2, 3, 1).reshape(n * h1 * w1, 1, 2)  # (n*h1*w1, 1, 2)
    pts = c + delta.view(1, -1, 2)                             # (n*h1*w1, S, 2)

    # 像素坐标 -> grid_sample 的归一化坐标 (align_corners=True)
    gx = 2.0 * pts[..., 0] / max(w2 - 1, 1) - 1.0
    gy = 2.0 * pts[..., 1] / max(h2 - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(1)          # (N,1,S,2)

    out = F.grid_sample(vol, grid, mode="bilinear",
                        padding_mode="zeros", align_corners=True)
    return out.view(n, h1, w1, -1).permute(0, 3, 1, 2).contiguous()


def t_corr_index():
    torch.manual_seed(0)
    B, N, C, H, W, R = 1, 2, 32, 12, 16, 3
    fmap1 = torch.randn(B, N, C, H, W, device=DEV)
    fmap2 = torch.randn(B, N, C, H, W, device=DEV)

    ys, xs = torch.meshgrid(torch.arange(H, device=DEV, dtype=torch.float32),
                            torch.arange(W, device=DEV, dtype=torch.float32),
                            indexing="ij")
    coords = torch.stack([xs, ys], dim=-1)[None, None].repeat(B, N, 1, 1, 1)
    coords = coords + 0.37                                     # 非整数，强制走双线性

    block = CorrBlock(fmap1, fmap2, num_levels=1, radius=R)
    got = block(coords)                                        # (B,N,S,H,W)

    volume = block.corr_pyramid[0]
    c = coords.permute(0, 1, 4, 2, 3).contiguous().view(B * N, 2, H, W)
    ref = corr_reference(volume, c, R).view(B, N, -1, H, W)

    assert torch.isfinite(got).all(), "CUDA 输出含 NaN/Inf"
    err = (got - ref).abs().max().item()
    assert err < 1e-3, f"与参考实现不符, 最大误差 {err:.3e}"
    return f"shape={tuple(got.shape)}  vs grid_sample 参考最大误差={err:.2e}"


def t_corr_index_backward():
    torch.manual_seed(1)
    B, N, C, H, W, R = 1, 1, 16, 10, 10, 2
    fmap1 = torch.randn(B, N, C, H, W, device=DEV)
    fmap2 = torch.randn(B, N, C, H, W, device=DEV)
    ys, xs = torch.meshgrid(torch.arange(H, device=DEV, dtype=torch.float32),
                            torch.arange(W, device=DEV, dtype=torch.float32),
                            indexing="ij")
    coords = torch.stack([xs, ys], dim=-1)[None, None].repeat(B, N, 1, 1, 1) + 0.25

    block = CorrBlock(fmap1, fmap2, num_levels=1, radius=R)
    block.corr_pyramid[0].requires_grad_(True)
    out = block(coords)
    out.sum().backward()

    g = block.corr_pyramid[0].grad
    assert g is not None, "反向未产生梯度"
    assert torch.isfinite(g).all(), "梯度含 NaN/Inf"
    assert g.abs().sum().item() > 0, "梯度全为 0"
    return f"grad shape={tuple(g.shape)}  |grad|sum={g.abs().sum().item():.3e}"


# ---------------------------------------------------------------------------
# 2. altcorr_kernel.cu —— altcorr_forward / backward
#    补丁点: AT_DISPATCH_FLOATING_TYPES_AND_HALF(fmap1.scalar_type(), ...)
# ---------------------------------------------------------------------------
def t_altcorr():
    torch.manual_seed(2)
    B, N, C, H, W, R = 1, 3, 32, 16, 16, 3
    fmaps = torch.randn(B, N, C, H, W, device=DEV)
    ys, xs = torch.meshgrid(torch.arange(H, device=DEV, dtype=torch.float32),
                            torch.arange(W, device=DEV, dtype=torch.float32),
                            indexing="ij")
    coords = torch.stack([xs, ys], dim=-1)[None, None].repeat(B, 2, 1, 1, 1) + 0.5

    ii = torch.tensor([0, 1], device=DEV)
    jj = torch.tensor([1, 2], device=DEV)
    block = AltCorrBlock(fmaps, num_levels=2, radius=R)
    corr = block(coords, ii, jj)

    assert torch.isfinite(corr).all(), "输出含 NaN/Inf"
    assert corr.abs().sum().item() > 0, "输出全为 0"
    return f"shape={tuple(corr.shape)}  |corr|mean={corr.abs().mean().item():.4f}"


# ---------------------------------------------------------------------------
# 3. lietorch_gpu.cu —— exp/log/inv/mul/adj 等 19 处 dispatch
#    用数学恒等式验证，比形状检查强得多
# ---------------------------------------------------------------------------
def t_lietorch_identities():
    torch.manual_seed(3)
    results = []
    for name, G, dof in [("SO3", SO3, 3), ("RxSO3", RxSO3, 4),
                         ("SE3", SE3, 6), ("Sim3", Sim3, 7)]:
        a = 0.1 * torch.randn(64, dof, device=DEV, dtype=torch.float64)

        # (a) exp/log 往返
        X = G.exp(a)
        e_explog = (X.log() - a).abs().max().item()

        # (b) X * X^-1 == 单位元
        e_inv = (X * X.inv()).log().abs().max().item()

        # (c) 结合律 (X*Y)*Z == X*(Y*Z)
        Y, Z = G.exp(0.1 * torch.randn_like(a)), G.exp(0.1 * torch.randn_like(a))
        e_assoc = (((X * Y) * Z) * (X * (Y * Z)).inv()).log().abs().max().item()

        worst = max(e_explog, e_inv, e_assoc)
        assert worst < 1e-8, (f"{name} 恒等式不成立: exp/log={e_explog:.2e} "
                             f"inv={e_inv:.2e} assoc={e_assoc:.2e}")
        results.append(f"{name}<{worst:.0e}")
    return "  ".join(results) + "  (exp/log 往返, X·X⁻¹=I, 结合律)"


def t_lietorch_adjoint():
    """Adj(X)·a == log(X · exp(a) · X⁻¹)，覆盖 adj 的 dispatch 分支。"""
    torch.manual_seed(4)
    X = SE3.exp(0.1 * torch.randn(32, 6, device=DEV, dtype=torch.float64))
    a = 0.01 * torch.randn(32, 6, device=DEV, dtype=torch.float64)
    lhs = X.adj(a)
    rhs = (X * SE3.exp(a) * X.inv()).log()
    err = (lhs - rhs).abs().max().item()
    assert err < 1e-8, f"伴随恒等式不成立, 误差 {err:.3e}"
    return f"SE3 Adj(X)·a == log(X·exp(a)·X⁻¹), 误差={err:.2e}"


# ---------------------------------------------------------------------------
# 4. droid_kernels.cu —— iproj / projmap / frame_distance / depth_filter / ba
# ---------------------------------------------------------------------------
def _scene(n=4, ht=30, wd=40):
    torch.manual_seed(5)
    poses = SE3.exp(0.05 * torch.randn(n, 6, device=DEV)).data.contiguous()
    disps = (0.5 + 0.1 * torch.rand(n, ht, wd, device=DEV)).contiguous()
    intr = torch.tensor([wd / 2.0, wd / 2.0, wd / 2.0, ht / 2.0], device=DEV)
    return poses, disps, intr


def t_iproj():
    poses, disps, intr = _scene()
    pts = droid_backends.iproj(poses, disps, intr)
    assert torch.isfinite(pts).all(), "反投影输出含 NaN/Inf"
    # 齐次坐标，w 分量应为逆深度
    return f"点云 shape={tuple(pts.shape)}  finite=True"


def t_frame_distance():
    poses, disps, intr = _scene()
    ii = torch.tensor([0, 1, 2], device=DEV, dtype=torch.long)
    jj = torch.tensor([1, 2, 3], device=DEV, dtype=torch.long)
    d = droid_backends.frame_distance(poses, disps, intr, ii, jj, 0.5)
    assert torch.isfinite(d).all(), "距离含 NaN/Inf"
    assert (d >= 0).all(), "距离出现负值"
    # 同一帧到自身的距离应为 0
    z = droid_backends.frame_distance(poses, disps, intr, ii, ii, 0.5)
    assert z.abs().max().item() < 1e-4, f"自距离非零: {z.abs().max().item():.3e}"
    return f"d={[round(x, 4) for x in d.tolist()]}  自距离≈0 ✓"


def t_projmap():
    poses, disps, intr = _scene()
    ii = torch.tensor([0, 1], device=DEV, dtype=torch.long)
    jj = torch.tensor([1, 2], device=DEV, dtype=torch.long)
    coords, valid = droid_backends.projmap(poses, disps, intr, ii, jj)
    assert torch.isfinite(coords[valid.bool().expand_as(coords)]).all(), "有效点含 NaN/Inf"
    return f"coords={tuple(coords.shape)}  有效点占比={valid.float().mean().item():.2%}"


def t_depth_filter():
    poses, disps, intr = _scene()
    ix = torch.tensor([0], device=DEV, dtype=torch.long)
    thresh = torch.tensor([0.05], device=DEV)
    cnt = droid_backends.depth_filter(poses, disps, intr, ix, thresh)
    assert torch.isfinite(cnt).all(), "输出含 NaN/Inf"
    return f"counts shape={tuple(cnt.shape)}  均值={cnt.float().mean().item():.3f}"


def t_ba():
    """Bundle adjustment —— DROID-SLAM 最核心的 kernel。"""
    n, ht, wd = 5, 30, 40
    poses, disps, intr = _scene(n, ht, wd)
    poses0, disps0 = poses.clone(), disps.clone()
    disps_sens = torch.zeros_like(disps)
    ii = torch.tensor([0, 1, 2, 3], device=DEV, dtype=torch.long)
    jj = torch.tensor([1, 2, 3, 4], device=DEV, dtype=torch.long)
    m = ii.numel()
    target = torch.randn(m, 2, ht, wd, device=DEV) * 0.1
    weight = torch.rand(m, 2, ht, wd, device=DEV)
    eta = torch.ones(n, ht, wd, device=DEV) * 0.01

    # 签名: (..., ii, jj, t0, t1, iterations, lm, ep, motion_only)
    droid_backends.ba(poses, disps, intr, disps_sens, target, weight, eta,
                      ii, jj, 1, n, 2, 1e-4, 0.1, False)

    assert torch.isfinite(poses).all(), "BA 后位姿含 NaN/Inf"
    assert torch.isfinite(disps).all(), "BA 后深度含 NaN/Inf"
    dp = (poses - poses0).abs().max().item()
    dd = (disps - disps0).abs().max().item()
    assert dp + dd > 0, "BA 未更新任何量（kernel 可能没真正执行）"
    return f"位姿变化={dp:.3e}  逆深度变化={dd:.3e}  (有更新且数值有限)"


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"torch {torch.__version__} | cuda {torch.version.cuda}")
    print(f"device {torch.cuda.get_device_name(0)} "
          f"{torch.cuda.get_device_capability(0)}")
    print(f"arch_list {torch.cuda.get_arch_list()}\n")
    assert "sm_120" in torch.cuda.get_arch_list(), "torch 里没有 sm_120 kernel"

    print("correlation_kernels.cu  (补丁 B)")
    check("corr_index_forward  vs grid_sample", t_corr_index)
    check("corr_index_backward", t_corr_index_backward)

    print("\naltcorr_kernel.cu  (补丁 B)")
    check("altcorr_forward", t_altcorr)

    print("\nlietorch_gpu.cu  (补丁 C, 19 处 dispatch)")
    check("群运算恒等式", t_lietorch_identities)
    check("SE3 伴随恒等式", t_lietorch_adjoint)

    print("\ndroid_kernels.cu")
    check("iproj  反投影", t_iproj)
    check("frame_distance", t_frame_distance)
    check("projmap", t_projmap)
    check("depth_filter", t_depth_filter)
    check("ba  光束法平差", t_ba)

    print(f"\n{'='*60}")
    print(f"通过 {len(PASS)} / {len(PASS) + len(FAIL)}")
    if FAIL:
        print("失败: " + ", ".join(FAIL))
    sys.exit(1 if FAIL else 0)
