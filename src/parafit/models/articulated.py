"""Two rigid parts joined by one revolute DOF, plus a global SE(3) pose.

Geometry (``moving`` part articulates, ``fixed`` part does not)::

    X_moving = R(w) @ R_a(alpha) @ k_moving + t
    X_fixed  = R(w) @             k_fixed  + t

with ``R_a(alpha) = exp(alpha * axis)`` a rotation about a *fixed* canonical
axis. Parameters are ``[rot(3) axis-angle, trans(3), art(1)]``, P = 7, laid out
in that order.

This is the ARCTIC object convention when constructed with ``axis = [0,0,-1]``
and the moving part first (verified against ARCTIC ground truth to 0.0004 mm),
but nothing here is ARCTIC-specific: any two-part revolute object (laptop lid,
scissors, box lid, microwave door, pliers) fits the same 7 DOF.

Setting a strong prior on the ``art`` DOF recovers a rigid 6-DOF body, so the
rigid and articulated cases are one model at two settings of a weight rather
than two code paths.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from parafit.core.lie import point_rotation_jacobian, skew, so3_exp
from parafit.core.types import ParamSpec, State
from parafit.models.base import Model


class TwoPartRevolute(Model):
    """Two rigid keypoint sets joined by a single revolute DOF.

    ``kp_moving`` (Km, 3) and ``kp_fixed`` (Kf, 3) are canonical keypoints
    shared across the batch. ``landmarks`` are returned as
    ``[moving; fixed]``, i.e. concatenated in that order.
    """

    def __init__(self, kp_moving: Tensor, kp_fixed: Tensor,
                 axis: Optional[Tensor] = None):
        self.kp_moving = kp_moving                  # (Km,3) canonical
        self.kp_fixed = kp_fixed                    # (Kf,3) canonical
        if axis is None:
            axis = torch.tensor([0.0, 0.0, -1.0], dtype=kp_moving.dtype)
        # Normalize so ``alpha`` is an angle in radians regardless of input scale.
        self.axis = (axis / axis.norm().clamp_min(1e-12)).to(kp_moving.dtype)
        self.param_spec = ParamSpec({"rot": (0, 3), "trans": (3, 3), "art": (6, 1)})

    @property
    def n_moving(self) -> int:
        return self.kp_moving.shape[0]

    # --- helpers ------------------------------------------------------------

    def _canonical_posed(self, params: Tensor) -> Tensor:
        """Canonical points after articulation, before the global pose: (B,J,3)."""
        B = params.shape[0]
        alpha = params[:, 6:7]                                   # (B,1)
        axis = self.axis.to(params.device, params.dtype)
        R_a = so3_exp(alpha * axis)                              # (B,3,3)
        moving = torch.einsum("bij,kj->bki", R_a, self.kp_moving.to(params))
        fixed = self.kp_fixed.to(params).expand(B, -1, -1)
        return torch.cat([moving, fixed], dim=1)                 # (B,J,3)

    # --- Model API ----------------------------------------------------------

    def forward(self, params: Tensor) -> State:
        B = params.shape[0]
        w, t = params[:, :3], params[:, 3:6]
        q = self._canonical_posed(params)                        # (B,J,3)
        X = torch.einsum("bij,bkj->bki", so3_exp(w), q) + t[:, None, :]
        return State(landmarks=X, batch_size=[B])

    def landmark_jacobian(self, params: Tensor, state: State) -> Tensor:
        """d(landmarks)/d(params) as (B, J, 3, 7).

        Three blocks, and the two rotation-like ones are *not* symmetric:

        * ``d/dt`` is the identity.
        * ``d/dw`` needs the SO(3) right-Jacobian correction, because the solver
          updates ``w`` additively (``theta <- theta - d``) rather than by a
          group retraction. That is ``-R [q]_x J_r(w)``.
        * ``d/dalpha`` needs *no* correction. ``alpha`` turns about a fixed axis,
          so ``R_a(alpha) = exp(alpha * axis)`` is a one-parameter subgroup and
          ``d/dalpha exp(alpha*K) = K exp(alpha*K)`` holds exactly. The
          derivative is therefore ``R(w) @ (axis x q)`` on the moving part and
          exactly zero on the fixed part.
        """
        B = params.shape[0]
        w = params[:, :3]
        q = self._canonical_posed(params)                        # (B,J,3)
        J_total = q.shape[1]
        R = so3_exp(w)                                           # (B,3,3)

        # d/dw: (B,J,3,3), right-Jacobian corrected.
        d_rot = point_rotation_jacobian(w, q)

        # d/dt: identity, (B,J,3,3).
        d_trans = torch.eye(3, dtype=params.dtype, device=params.device)
        d_trans = d_trans.view(1, 1, 3, 3).expand(B, J_total, 3, 3)

        # d/dalpha: R @ (axis x q) on the moving part, 0 on the fixed part.
        axis = self.axis.to(params.device, params.dtype)
        K = skew(axis)                                           # (3,3)
        dq = torch.einsum("ij,bkj->bki", K, q)                   # axis x q, (B,J,3)
        d_art = torch.einsum("bij,bkj->bki", R, dq)              # (B,J,3)
        mask = torch.zeros(J_total, dtype=params.dtype, device=params.device)
        mask[: self.n_moving] = 1.0
        d_art = (d_art * mask[None, :, None]).unsqueeze(-1)      # (B,J,3,1)

        return torch.cat([d_rot, d_trans, d_art], dim=-1)        # (B,J,3,7)
