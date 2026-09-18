"""LMSolver: batched adaptive Levenberg-Marquardt over a Model + Energies.

Generalizes ``_gn_loop_unc`` from the UA-Fit solver: per iteration it assembles
the summed normal equations from all energies, solves the damped system, does a
per-sample accept/reject trust-region step, and adapts the damping. The model
and energies are fully pluggable; the core imports neither MANO nor cameras.

Backends: ``"torch"`` (this eager implementation) today. ``"cuda"`` /
``"tensorrt"`` are future execution backends -- the exportable path is the
fixed-iteration mode (``mode="fixed"``, no data-dependent accept/reject).
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from parafit.core.types import GNBlock, SolveResult, State
from parafit.energies.base import Energy
from parafit.models.base import Model


def _landmark_jac(model: Model, theta: Tensor, state: State) -> Tensor:
    J = model.landmark_jacobian(theta, state)
    if J is not None:
        return J
    # Autograd fallback (correct but slower; models are encouraged to override).
    from torch.func import jacrev, vmap

    def single(p: Tensor) -> Tensor:
        return model.forward(p.unsqueeze(0)).landmarks.squeeze(0)  # (J,3)

    try:
        return vmap(jacrev(single))(theta)                          # (B,J,3,P)
    except RuntimeError:                                            # forward not vmap-able (e.g. boolean-mask indexing): per-sample jacobian
        from torch.autograd.functional import jacobian
        return torch.stack([jacobian(single, theta[b], vectorize=False) for b in range(theta.shape[0])])


class LMSolver:
    def __init__(
        self,
        max_iters: int = 10,
        damping: float = 1e-3,
        mode: str = "lm",            # "lm" (adaptive) | "fixed" (exportable) | "pure" (GN)
        up: float = 3.0,
        down: float = 0.3,
        record: bool = False,
    ):
        self.max_iters = max_iters
        self.damping = damping
        self.mode = mode
        self.up = up
        self.down = down
        self.record = record

    def _assemble(self, model: Model, theta: Tensor, energies: Sequence[Energy], need_jac: bool):
        state = model.forward(theta)
        if need_jac:
            state.landmark_jac = _landmark_jac(model, theta, state)
        block: GNBlock | None = None
        for e in energies:
            b = e.linearize(model, theta, state)
            block = b if block is None else block + b
        return block, state

    def solve(self, model: Model, init_params: Tensor, energies: Sequence[Energy]) -> SolveResult:
        theta = init_params.clone()
        B, P = theta.shape
        eye = torch.eye(P, dtype=theta.dtype, device=theta.device).expand(B, P, P)
        lam = torch.full((B,), max(self.damping, 1e-6), dtype=theta.dtype, device=theta.device)
        history = []

        block, state = self._assemble(model, theta, energies, need_jac=True)
        cost = block.cost
        for it in range(self.max_iters):
            damp = (0.0 if self.mode == "pure" else lam).view(B, 1, 1) * eye
            d = torch.linalg.solve(block.A + damp, block.g.unsqueeze(-1)).squeeze(-1)  # (B,P)
            theta_try = theta - d

            if self.mode == "fixed" or self.mode == "pure":
                theta = theta_try
                block, state = self._assemble(model, theta, energies, need_jac=True)
                cost = block.cost
                if self.record:
                    history.append(cost.detach().mean().item())
                continue

            # Adaptive LM: accept per sample only if cost decreased.
            block_try, state_try = self._assemble(model, theta_try, energies, need_jac=True)
            acc = block_try.cost < cost                                  # (B,)
            accf = acc.view(B, 1)
            theta = torch.where(accf, theta_try, theta)
            cost = torch.where(acc, block_try.cost, cost)
            # Re-linearize only the accepted rows (rejected keep old block).
            A = torch.where(accf.unsqueeze(-1), block_try.A, block.A)
            g = torch.where(accf, block_try.g, block.g)
            block = GNBlock(A=A, g=g, cost=cost, batch_size=[B])
            state = state_try
            lam = torch.where(acc, (lam * self.down).clamp_min(1e-6), (lam * self.up).clamp_max(1e6))
            if self.record:
                history.append(cost.detach().mean().item())

        diagnostics = {"final_cost": cost.detach(), "damping": lam.detach()}
        if self.record:
            diagnostics["cost_history"] = history
        return SolveResult(params=theta, state=state, diagnostics=diagnostics)
