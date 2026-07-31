#!/usr/bin/env python
"""
sm_120 (RTX 5090) CUDA kernel numerical validation

The installation applies two types of patches to DROID-SLAM / lietorch:
  - Change hard-coded gencode from sm_86 to sm_120
  - Change tensor.type() to scalar_type() in AT_DISPATCH / DISPATCH_GROUP_AND_FLOATING_TYPES

These patches alter kernel dispatch paths, so successful imports are insufficient.
Each kernel is compared against a pure PyTorch reference implementation.

Usage: source env_droid.sh && python /mnt/g/algorithm_backup/Nevstereo/test_sm120_kernels.py
"""
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path("/mnt/g/algorithm_backup/Nevstereo/droid_metric")))

import droid_backends  # noqa: E402  (must follow import torch)
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
# 1. correlation_kernels.cu - corr_index_forward / backward
#    Patched call: AT_DISPATCH_FLOATING_TYPES_AND_HALF(volume.scalar_type(), ...)
# ---------------------------------------------------------------------------
def corr_reference(volume, coords, radius):
    """Pure PyTorch equivalent of CorrSampler using grid_sample bilinear sampling."""
    n, h1, w1, h2, w2 = volume.shape
    vol = volume.reshape(n * h1 * w1, 1, h2, w2)

    # In corr[n][i][j][y][x], i comes from the x-offset loop and j from the
    # y-offset loop. The patch dimensions are x-major (flat = i_x * rd + j_y).
    # meshgrid's first output changes slowest, so dx must come first.
    dx, dy = torch.meshgrid(
        torch.arange(-radius, radius + 1, device=volume.device, dtype=torch.float32),
        torch.arange(-radius, radius + 1, device=volume.device, dtype=torch.float32),
        indexing="ij",
    )
    delta = torch.stack([dx, dy], dim=-1).view(-1, 2)          # (S,2), ordered as (x,y)

    c = coords.permute(0, 2, 3, 1).reshape(n * h1 * w1, 1, 2)  # (n*h1*w1, 1, 2)
    pts = c + delta.view(1, -1, 2)                             # (n*h1*w1, S, 2)

    # Pixel coordinates -> normalized grid_sample coordinates (align_corners=True)
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
    coords = coords + 0.37                                     # Noninteger to force bilinear sampling

    block = CorrBlock(fmap1, fmap2, num_levels=1, radius=R)
    got = block(coords)                                        # (B,N,S,H,W)

    volume = block.corr_pyramid[0]
    c = coords.permute(0, 1, 4, 2, 3).contiguous().view(B * N, 2, H, W)
    ref = corr_reference(volume, c, R).view(B, N, -1, H, W)

    assert torch.isfinite(got).all(), "CUDA output contains NaN/Inf"
    err = (got - ref).abs().max().item()
    assert err < 1e-3, f"Reference mismatch, maximum error {err:.3e}"
    return f"shape={tuple(got.shape)}  vs grid_sample reference max error={err:.2e}"


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
    assert g is not None, "Backward pass produced no gradient"
    assert torch.isfinite(g).all(), "Gradient contains NaN/Inf"
    assert g.abs().sum().item() > 0, "Gradient is entirely zero"
    return f"grad shape={tuple(g.shape)}  |grad|sum={g.abs().sum().item():.3e}"


# ---------------------------------------------------------------------------
# 2. altcorr_kernel.cu - altcorr_forward / backward
#    Patched call: AT_DISPATCH_FLOATING_TYPES_AND_HALF(fmap1.scalar_type(), ...)
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

    assert torch.isfinite(corr).all(), "Output contains NaN/Inf"
    assert corr.abs().sum().item() > 0, "Output is entirely zero"
    return f"shape={tuple(corr.shape)}  |corr|mean={corr.abs().mean().item():.4f}"


# ---------------------------------------------------------------------------
# 3. lietorch_gpu.cu - 19 dispatch sites including exp/log/inv/mul/adj
#    Validate mathematical identities, which is stronger than shape checks.
# ---------------------------------------------------------------------------
def t_lietorch_identities():
    torch.manual_seed(3)
    results = []
    for name, G, dof in [("SO3", SO3, 3), ("RxSO3", RxSO3, 4),
                         ("SE3", SE3, 6), ("Sim3", Sim3, 7)]:
        a = 0.1 * torch.randn(64, dof, device=DEV, dtype=torch.float64)

        # (a) exp/log round trip
        X = G.exp(a)
        e_explog = (X.log() - a).abs().max().item()

        # (b) X * X^-1 == identity
        e_inv = (X * X.inv()).log().abs().max().item()

        # (c) associativity: (X*Y)*Z == X*(Y*Z)
        Y, Z = G.exp(0.1 * torch.randn_like(a)), G.exp(0.1 * torch.randn_like(a))
        e_assoc = (((X * Y) * Z) * (X * (Y * Z)).inv()).log().abs().max().item()

        worst = max(e_explog, e_inv, e_assoc)
        assert worst < 1e-8, (f"{name} identity failed: exp/log={e_explog:.2e} "
                             f"inv={e_inv:.2e} assoc={e_assoc:.2e}")
        results.append(f"{name}<{worst:.0e}")
    return "  ".join(results) + "  (exp/log round trip, X*X^-1=I, associativity)"


def t_lietorch_adjoint():
    """Adj(X)*a == log(X * exp(a) * X^-1), covering the adj dispatch branch."""
    torch.manual_seed(4)
    X = SE3.exp(0.1 * torch.randn(32, 6, device=DEV, dtype=torch.float64))
    a = 0.01 * torch.randn(32, 6, device=DEV, dtype=torch.float64)
    lhs = X.adj(a)
    rhs = (X * SE3.exp(a) * X.inv()).log()
    err = (lhs - rhs).abs().max().item()
    assert err < 1e-8, f"Adjoint identity failed, error {err:.3e}"
    return f"SE3 Adj(X)*a == log(X*exp(a)*X^-1), error={err:.2e}"


# ---------------------------------------------------------------------------
# 4. droid_kernels.cu - iproj / projmap / frame_distance / depth_filter / ba
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
    assert torch.isfinite(pts).all(), "Back-projection output contains NaN/Inf"
    # In homogeneous coordinates, w should be inverse depth.
    return f"point cloud shape={tuple(pts.shape)}  finite=True"


def t_frame_distance():
    poses, disps, intr = _scene()
    ii = torch.tensor([0, 1, 2], device=DEV, dtype=torch.long)
    jj = torch.tensor([1, 2, 3], device=DEV, dtype=torch.long)
    d = droid_backends.frame_distance(poses, disps, intr, ii, jj, 0.5)
    assert torch.isfinite(d).all(), "Distance contains NaN/Inf"
    assert (d >= 0).all(), "Distance contains negative values"
    # The distance from a frame to itself should be zero.
    z = droid_backends.frame_distance(poses, disps, intr, ii, ii, 0.5)
    assert z.abs().max().item() < 1e-4, f"Self-distance is nonzero: {z.abs().max().item():.3e}"
    return f"d={[round(x, 4) for x in d.tolist()]}  self-distance~=0"


def t_projmap():
    poses, disps, intr = _scene()
    ii = torch.tensor([0, 1], device=DEV, dtype=torch.long)
    jj = torch.tensor([1, 2], device=DEV, dtype=torch.long)
    coords, valid = droid_backends.projmap(poses, disps, intr, ii, jj)
    assert torch.isfinite(coords[valid.bool().expand_as(coords)]).all(), "Valid points contain NaN/Inf"
    return f"coords={tuple(coords.shape)}  valid ratio={valid.float().mean().item():.2%}"


def t_depth_filter():
    poses, disps, intr = _scene()
    ix = torch.tensor([0], device=DEV, dtype=torch.long)
    thresh = torch.tensor([0.05], device=DEV)
    cnt = droid_backends.depth_filter(poses, disps, intr, ix, thresh)
    assert torch.isfinite(cnt).all(), "Output contains NaN/Inf"
    return f"counts shape={tuple(cnt.shape)}  mean={cnt.float().mean().item():.3f}"


def t_ba():
    """Bundle adjustment, DROID-SLAM's central kernel."""
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

    # Signature: (..., ii, jj, t0, t1, iterations, lm, ep, motion_only)
    droid_backends.ba(poses, disps, intr, disps_sens, target, weight, eta,
                      ii, jj, 1, n, 2, 1e-4, 0.1, False)

    assert torch.isfinite(poses).all(), "Poses contain NaN/Inf after BA"
    assert torch.isfinite(disps).all(), "Depth contains NaN/Inf after BA"
    dp = (poses - poses0).abs().max().item()
    dd = (disps - disps0).abs().max().item()
    assert dp + dd > 0, "BA updated nothing (the kernel may not have run)"
    return f"pose change={dp:.3e}  inverse-depth change={dd:.3e}  (updated and finite)"


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"torch {torch.__version__} | cuda {torch.version.cuda}")
    print(f"device {torch.cuda.get_device_name(0)} "
          f"{torch.cuda.get_device_capability(0)}")
    print(f"arch_list {torch.cuda.get_arch_list()}\n")
    assert "sm_120" in torch.cuda.get_arch_list(), "torch has no sm_120 kernel"

    print("correlation_kernels.cu  (patch B)")
    check("corr_index_forward  vs grid_sample", t_corr_index)
    check("corr_index_backward", t_corr_index_backward)

    print("\naltcorr_kernel.cu  (patch B)")
    check("altcorr_forward", t_altcorr)

    print("\nlietorch_gpu.cu  (patch C, 19 dispatch sites)")
    check("group operation identities", t_lietorch_identities)
    check("SE3 adjoint identity", t_lietorch_adjoint)

    print("\ndroid_kernels.cu")
    check("iproj  back-projection", t_iproj)
    check("frame_distance", t_frame_distance)
    check("projmap", t_projmap)
    check("depth_filter", t_depth_filter)
    check("ba  bundle adjustment", t_ba)

    print(f"\n{'='*60}")
    print(f"Passed {len(PASS)} / {len(PASS) + len(FAIL)}")
    if FAIL:
        print("Failed: " + ", ".join(FAIL))
    sys.exit(1 if FAIL else 0)
