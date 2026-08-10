"""SO(3) exponential map, logarithm, and right Jacobian.

Batched, torch-only. Convention: a rotation is parameterized by an axis-angle
vector ``w`` (B, 3). Updates are *additive* on ``w`` (the solver does
``w <- w - d``), so the Jacobian of a rotated point w.r.t. ``w`` carries the
SO(3) right-Jacobian correction:

    d(R(w) p) / dw = -R(w) [p]_x J_r(w)

This mirrors the articulated Jacobian used in the POEM-v2 / UA-Fit solver
(``axes = Rg @ Jr`` followed by a cross product with the lever arm).
"""
from __future__ import annotations

import torch
from torch import Tensor

_EPS = 1e-8


def skew(w: Tensor) -> Tensor:
    """(..., 3) axis-angle -> (..., 3, 3) skew-symmetric matrix."""
    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    z = torch.zeros_like(wx)
    row0 = torch.stack([z, -wz, wy], dim=-1)
    row1 = torch.stack([wz, z, -wx], dim=-1)
    row2 = torch.stack([-wy, wx, z], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def so3_exp(w: Tensor) -> Tensor:
    """Rodrigues: (..., 3) axis-angle -> (..., 3, 3) rotation matrix."""
    theta = torch.linalg.norm(w, dim=-1, keepdim=True).unsqueeze(-1)  # (...,1,1)
    W = skew(w)
    W2 = W @ W
    small = theta < 1e-4
    # Series-safe coefficients.
    a = torch.where(small, 1.0 - theta**2 / 6.0, torch.sin(theta) / theta.clamp_min(_EPS))
    b = torch.where(small, 0.5 - theta**2 / 24.0, (1.0 - torch.cos(theta)) / theta.clamp_min(_EPS) ** 2)
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(W)
    return eye + a * W + b * W2


def so3_right_jacobian(w: Tensor) -> Tensor:
    """Right Jacobian of SO(3): (..., 3) -> (..., 3, 3)."""
    theta = torch.linalg.norm(w, dim=-1, keepdim=True).unsqueeze(-1)
    W = skew(w)
    W2 = W @ W
    small = theta < 1e-4
    c1 = torch.where(small, 0.5 - theta**2 / 24.0, (1.0 - torch.cos(theta)) / theta.clamp_min(_EPS) ** 2)
    c2 = torch.where(small, 1.0 / 6.0 - theta**2 / 120.0, (theta - torch.sin(theta)) / theta.clamp_min(_EPS) ** 3)
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(W)
    return eye - c1 * W + c2 * W2


def so3_log(R: Tensor) -> Tensor:
    """(..., 3, 3) rotation -> (..., 3) axis-angle. Numerically stable near 0/pi."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos)  # (...,)
    vee = torch.stack(
        [R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]],
        dim=-1,
    )  # 2 sin(theta) * axis
    small = theta < 1e-4
    coef = torch.where(small, 0.5 + theta**2 / 12.0, theta / (2.0 * torch.sin(theta).clamp_min(_EPS)))
    return coef.unsqueeze(-1) * vee


def point_rotation_jacobian(w: Tensor, p: Tensor) -> Tensor:
    """d(R(w) p) / dw with the right-Jacobian correction.

    ``w`` (B, 3), ``p`` (B, J, 3) canonical points -> (B, J, 3, 3).
    """
    R = so3_exp(w)                       # (B,3,3)
    Jr = so3_right_jacobian(w)           # (B,3,3)
    RJr = R @ Jr                         # (B,3,3)
    P = skew(p)                          # (B,J,3,3)
    # -R Jr [p]_x  == -(R Jr) @ [p]_x, but we need -R [p]_x Jr; expand carefully:
    # d(Rp)/dw = -R [p]_x Jr
    Rp = R.unsqueeze(1)                  # (B,1,3,3)
    return -(Rp @ P) @ Jr.unsqueeze(1)   # (B,J,3,3)
