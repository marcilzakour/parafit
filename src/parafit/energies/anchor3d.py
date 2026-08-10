"""3D anchor energy: pull landmarks toward a target (temporal / external prior).

Generalizes the learned-3D-anchor and temporal-coherence blocks of the UA-Fit
solver (``mu3d``/``omega3`` and ``TEMPORAL_ANCHOR``). Core, torch-only.
"""
from __future__ import annotations

from typing import Optional

from torch import Tensor

from parafit.core.types import GNBlock, State
from parafit.energies.base import Energy, gauss_newton_block


class Anchor3DEnergy(Energy):
    name = "anchor3d"

    def __init__(self, target: Tensor, precision: Optional[Tensor] = None, weight: float = 1.0):
        self.target = target          # (B, J, 3)
        self.precision = precision    # (B,J,3,3) block | (B,J) scalar | None
        self.weight = weight

    def linearize(self, model, params: Tensor, state: State) -> GNBlock:
        r = state.landmarks - self.target            # (B,J,3)
        J = state.landmark_jac                        # (B,J,3,P)
        block = gauss_newton_block(r, J, self.precision)
        if self.weight != 1.0:
            block = block * self.weight
        return block
