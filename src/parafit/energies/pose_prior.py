"""Pose prior energy: quadratic pull of the parameter vector toward a reference.

Isotropic L2-to-reference (matches the default ``prior=None`` branch of the
UA-Fit ``_normal_eqs``); a Mahalanobis precision (PCA hand prior) can be passed
per-DOF. Operates directly on the parameter vector, so it needs no Jacobian.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from parafit.core.types import GNBlock, State
from parafit.energies.base import Energy, WeightLike, scale_block


class PosePriorEnergy(Energy):
    name = "pose_prior"

    def __init__(self, ref: Tensor, weight: WeightLike = 1.0, precision: Optional[Tensor] = None):
        self.ref = ref               # (B, P) or (P,)
        self.weight = weight         # float, or Tensor 0-dim / (B,)
        self.precision = precision   # (B,P,P) | (P,) diag | None (isotropic)

    def linearize(self, model, params: Tensor, state: State) -> GNBlock:
        B, P = params.shape
        r = params - self.ref                                    # (B,P)
        if self.precision is None:
            W = torch.eye(P, dtype=params.dtype, device=params.device).expand(B, P, P)
        elif self.precision.dim() == 1:
            W = torch.diag_embed(self.precision).expand(B, P, P)
        else:
            W = self.precision
        # Build unweighted, then scale via scale_block: folding the weight into W
        # here would mis-broadcast a (B,) tensor against W's (B,P,P).
        A = W
        g = torch.einsum("bpq,bq->bp", W, r)
        cost = 0.5 * torch.einsum("bp,bpq,bq->b", r, W, r)
        return scale_block(GNBlock(A=A, g=g, cost=cost, batch_size=[B]), self.weight)
