"""The Energy ABC and Gauss-Newton block assembly.

Every energy contributes the same triple to the normal equations -- an
``(A, g, cost)`` -- exactly as the inline energy terms do today in the UA-Fit
solver (``_build_system_unc``). Making that triple an object removes the
module-global on/off switches and lets energies compose explicitly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import torch
from torch import Tensor

from parafit.core.types import GNBlock, State

#: An energy weight. A Python ``float`` is a fixed hyperparameter; a ``Tensor``
#: makes the weight a *learnable, differentiable* quantity, optionally per-batch.
WeightLike = Union[float, Tensor]


def scale_block(block: GNBlock, weight: WeightLike) -> GNBlock:
    """Scale an energy's ``(A, g, cost)`` by ``weight``.

    Accepts either form:

    * ``float`` -- fixed hyperparameter. ``1.0`` short-circuits to a no-op, so
      the default path is bit-identical to an unweighted block.
    * ``Tensor`` -- 0-dim (shared) or ``(B,)`` (per-batch-element). Gradient
      flows to ``weight``, which is what lets a weight be *predicted* rather
      than tuned.

    Why this is not ``block * weight``: ``GNBlock``'s fields have different
    ranks -- ``A`` (B,P,P), ``g`` (B,P), ``cost`` (B,) -- so a ``(B,)`` tensor
    would broadcast against trailing dimensions and silently scale ``A`` along
    its last axis instead of its batch axis. Each field is therefore scaled
    explicitly. A ``float`` cannot hit that bug, which is why the old
    ``block * self.weight`` was correct until weights became tensors.

    Scaling all three fields by the same ``w`` is exactly ``w * ||r||^2_Omega``:
    ``A = J^T Omega J``, ``g = J^T Omega r`` and ``cost = 0.5 r^T Omega r`` are
    all linear in ``Omega``, so the weighted block stays a consistent
    Gauss-Newton system rather than a mismatched (A, g) pair.
    """
    if not isinstance(weight, Tensor):
        if weight == 1.0:
            return block
        return block * float(weight)

    w = weight.to(dtype=block.A.dtype, device=block.A.device)
    if w.ndim == 0:
        return GNBlock(A=block.A * w, g=block.g * w, cost=block.cost * w,
                       batch_size=block.batch_size)
    if w.ndim != 1 or w.shape[0] != block.A.shape[0]:
        raise ValueError(
            f"weight tensor must be 0-dim or (B,) with B={block.A.shape[0]}, "
            f"got shape {tuple(w.shape)}"
        )
    return GNBlock(
        A=block.A * w[:, None, None],
        g=block.g * w[:, None],
        cost=block.cost * w,
        batch_size=block.batch_size,
    )


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
