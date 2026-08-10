"""Pinhole multi-view camera: projection and closed-form Jacobian.

Grounds the reprojection energy. Mirrors ``_project_and_jac`` in the POEM-v2 /
UA-Fit solver (``analytic_fitter.py``).
"""
from __future__ import annotations

import torch
from torch import Tensor


class PinholeCameras:
    """A batch of ``V`` calibrated views.

    ``K`` (B, V, 3, 3) intrinsics, ``w2c`` (B, V, 4, 4) world-to-camera.
    """

    def __init__(self, K: Tensor, w2c: Tensor):
        assert K.shape[-2:] == (3, 3) and w2c.shape[-2:] == (4, 4)
        self.K = K
        self.w2c = w2c

    @property
    def num_views(self) -> int:
        return self.K.shape[1]

    def project_and_jac(self, X_world: Tensor):
        """``X_world`` (B, J, 3) -> (uv (B, V, J, 2), duv_dXworld (B, V, J, 2, 3)).

        ``Z<=0`` points are clamped to avoid divide-by-zero; callers mask them
        via the energy weight if desired.
        """
        B, J, _ = X_world.shape
        V = self.num_views
        Rcw = self.w2c[..., :3, :3]                      # (B,V,3,3)
        tcw = self.w2c[..., :3, 3]                        # (B,V,3)
        Xw = X_world.unsqueeze(1)                         # (B,1,J,3)
        Xc = torch.einsum("bvij,bvkj->bvki", Rcw, Xw) + tcw.unsqueeze(2)  # (B,V,J,3)
        fx = self.K[..., 0, 0].unsqueeze(-1)             # (B,V,1)
        fy = self.K[..., 1, 1].unsqueeze(-1)
        cx = self.K[..., 0, 2].unsqueeze(-1)
        cy = self.K[..., 1, 2].unsqueeze(-1)
        Xx, Yy, Zz = Xc[..., 0], Xc[..., 1], Xc[..., 2]  # (B,V,J)
        Zc = Zz.clamp_min(1e-6)
        u = fx * Xx / Zc + cx
        v = fy * Yy / Zc + cy
        uv = torch.stack([u, v], dim=-1)                 # (B,V,J,2)
        # duv/dXcam (B,V,J,2,3)
        zero = torch.zeros_like(Xx)
        duv_dXcam = torch.stack(
            [
                torch.stack([fx / Zc, zero, -fx * Xx / Zc**2], dim=-1),
                torch.stack([zero, fy / Zc, -fy * Yy / Zc**2], dim=-1),
            ],
            dim=-2,
        )                                                # (B,V,J,2,3)
        # chain to world: duv/dXworld = duv/dXcam @ Rcw
        duv_dXworld = duv_dXcam @ Rcw.unsqueeze(2)       # (B,V,J,2,3)
        return uv, duv_dXworld
