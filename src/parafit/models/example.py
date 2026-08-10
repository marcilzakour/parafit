"""RigidKeypointModel: a fully-working core-only reference Model.

Parameters are a 6-DoF rigid pose ``[rot(3 axis-angle) | trans(3)]`` applied to
a fixed canonical set of 3D keypoints. This is the "wrist 6-DoF" sub-problem of
the hand solver and needs no MANO/UmeTrack assets, so the solver core is
testable end to end (see ``tests/test_core_overfit.py``). It also demonstrates a
correct *analytic* landmark Jacobian for the registry/API to model against.
"""
from __future__ import annotations

import torch
from torch import Tensor

from parafit.core.lie import point_rotation_jacobian, so3_exp
from parafit.core.types import ParamSpec, State
from parafit.models.base import Model


class RigidKeypointModel(Model):
    def __init__(self, canonical_points: Tensor):
        """``canonical_points`` (J, 3) fixed keypoints in the body frame."""
        self.canonical = canonical_points
        self.param_spec = ParamSpec({"rot": (0, 3), "trans": (3, 3)})

    def forward(self, params: Tensor) -> State:
        w = params[:, 0:3]                                   # (B,3)
        t = params[:, 3:6]                                   # (B,3)
        R = so3_exp(w)                                        # (B,3,3)
        p = self.canonical.to(params).unsqueeze(0)           # (1,J,3)
        X = torch.einsum("bij,bkj->bki", R, p) + t.unsqueeze(1)  # (B,J,3)
        return State(landmarks=X)

    def landmark_jacobian(self, params: Tensor, state: State) -> Tensor:
        B = params.shape[0]
        w = params[:, 0:3]
        p = self.canonical.to(params).expand(B, -1, -1)      # (B,J,3)
        J = p.shape[1]
        drot = point_rotation_jacobian(w, p)                 # (B,J,3,3)
        dtrans = torch.eye(3, dtype=params.dtype, device=params.device)
        dtrans = dtrans.view(1, 1, 3, 3).expand(B, J, 3, 3)  # (B,J,3,3)
        return torch.cat([drot, dtrans], dim=-1)             # (B,J,3,6)
