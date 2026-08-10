"""The Energy ABC and Gauss-Newton block assembly.

Every energy contributes the same triple to the normal equations -- an
``(A, g, cost)`` -- exactly as the inline energy terms do today in the UA-Fit
solver (``_build_system_unc``). Making that triple an object removes the
module-global on/off switches and lets energies compose explicitly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor

from parafit.core.types import GNBlock, State


def gauss_newton_block(r: Tensor, J: Tensor, Omega: Optional[Tensor] = None) -> GNBlock:
    """Assemble one block from residuals and their Jacobian.

    ``r`` (B, M, d) residuals, ``J`` (B, M, d, P) Jacobian, ``Omega`` the
    precision: (B, M, d, d) anisotropic block, (B, M) per-residual scalar, or
    ``None`` (identity). ``A = sum J^T Omega J``, ``g = sum J^T Omega r``.
    """
    B, M, d, P = J.shape
    if Omega is None:
        Om = torch.eye(d, dtype=r.dtype, device=r.device).view(1, 1, d, d).expand(B, M, d, d)
    elif Omega.dim() == 2:  # per-residual scalar weight
        Om = Omega.view(B, M, 1, 1) * torch.eye(d, dtype=r.dtype, device=r.device)
    else:
        Om = Omega
    A = torch.einsum("bmdp,bmde,bmeq->bpq", J, Om, J)
    g = torch.einsum("bmdp,bmde,bme->bp", J, Om, r)
    cost = 0.5 * torch.einsum("bmd,bmde,bme->b", r, Om, r)
    return GNBlock(A=A, g=g, cost=cost, batch_size=[B])


class Energy(ABC):
    """A composable residual term of the fitting objective."""

    #: registry name (optional; set by @register).
    name: str = ""

    @abstractmethod
    def linearize(self, model, params: Tensor, state: State) -> GNBlock:
        """Return this term's ``(A, g, cost)`` at the current linearization point.

        ``state.landmark_jac`` (B, J, 3, P) has been populated by the solver
        (analytic or autograd), so energies that operate on landmarks can chain
        through it without recomputing the model Jacobian.
        """
