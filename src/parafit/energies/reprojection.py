"""Multi-view reprojection energy with anisotropic precision.

Generalizes both data terms in the UA-Fit solver: the DOVF field term and the
external-2D (``WILOR_2D``) term are the same energy with different targets and
precisions. Works with any ``Model`` that produces landmarks.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from parafit.core.types import GNBlock, State
from parafit.energies.base import Energy, gauss_newton_block
from parafit.io.camera import PinholeCameras


class ReprojectionEnergy(Energy):
    name = "reprojection"

    def __init__(
        self,
        cameras: PinholeCameras,
        target_uv: Tensor,               # (B, V, J, 2) target pixels
        precision: Optional[Tensor] = None,  # (B,V,J,2,2) block, (B,V,J) scalar, or None
        weight: float = 1.0,
    ):
        self.cameras = cameras
        self.target_uv = target_uv
        self.precision = precision
        self.weight = weight

    def linearize(self, model, params: Tensor, state: State) -> GNBlock:
        X = state.landmarks                                  # (B,J,3)
        B, J, _ = X.shape
        uv, duv_dXw = self.cameras.project_and_jac(X)        # (B,V,J,2), (B,V,J,2,3)
        V = uv.shape[1]
        dXw_dp = state.landmark_jac.unsqueeze(1)             # (B,1,J,3,P)
        Jobs = torch.einsum("bvjdc,bvjcp->bvjdp", duv_dXw, dXw_dp.expand(B, V, J, 3, -1))
        r = uv - self.target_uv                              # (B,V,J,2)
        # Flatten (V,J) observations.
        P = Jobs.shape[-1]
        r = r.reshape(B, V * J, 2)
        Jobs = Jobs.reshape(B, V * J, 2, P)
        if self.precision is None:
            Om = None
        elif self.precision.dim() == 3:                      # (B,V,J) scalar
            Om = self.precision.reshape(B, V * J)
        else:                                                # (B,V,J,2,2)
            Om = self.precision.reshape(B, V * J, 2, 2)
        block = gauss_newton_block(r, Jobs, Om)
        if self.weight != 1.0:
            block = GNBlock(self.weight * block.A, self.weight * block.g, self.weight * block.cost)
        return block
