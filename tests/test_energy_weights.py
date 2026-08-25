"""Energy weights accept a float (fixed) or a Tensor (learned, per-batch).

The float path must stay bit-identical to the previous behaviour -- that is the
backward-compatibility contract. The tensor path is what lets a weight be
*predicted* instead of tuned, so it must (a) broadcast on the batch axis rather
than the trailing axis, and (b) carry gradient back to the weight itself.
"""
from __future__ import annotations

import pytest
import torch

from parafit.core.types import GNBlock
from parafit.energies.base import gauss_newton_block, scale_block

B, M, D, P = 3, 7, 2, 5


def _block(seed: int = 0) -> GNBlock:
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(B, M, D, generator=g, dtype=torch.float64)
    J = torch.randn(B, M, D, P, generator=g, dtype=torch.float64)
    return gauss_newton_block(r, J)


def _assert_close(a: GNBlock, b: GNBlock):
    torch.testing.assert_close(a.A, b.A)
    torch.testing.assert_close(a.g, b.g)
    torch.testing.assert_close(a.cost, b.cost)


# --- backward compatibility -------------------------------------------------

def test_float_one_is_exact_noop():
    blk = _block()
    _assert_close(scale_block(blk, 1.0), blk)


def test_float_scaling_matches_manual():
    blk = _block()
    out = scale_block(blk, 2.5)
    torch.testing.assert_close(out.A, blk.A * 2.5)
    torch.testing.assert_close(out.g, blk.g * 2.5)
    torch.testing.assert_close(out.cost, blk.cost * 2.5)


def test_scalar_tensor_matches_float():
    """A 0-dim tensor and the equivalent float must agree exactly."""
    blk = _block()
    _assert_close(scale_block(blk, torch.tensor(2.5, dtype=torch.float64)),
                  scale_block(blk, 2.5))


def test_uniform_tensor_matches_float():
    """Reduction check: a (B,) tensor of w equals the scalar float w."""
    blk = _block()
    w = torch.full((B,), 2.5, dtype=torch.float64)
    _assert_close(scale_block(blk, w), scale_block(blk, 2.5))


def test_ones_tensor_reproduces_unweighted():
    """Reduction check at the identity: w = 1 tensor == no weighting at all."""
    blk = _block()
    _assert_close(scale_block(blk, torch.ones(B, dtype=torch.float64)), blk)


# --- the broadcast trap -----------------------------------------------------

def test_per_batch_weight_scales_batch_axis():
    """Each batch element is scaled by its own w, on every field.

    GNBlock's fields have different ranks (A (B,P,P), g (B,P), cost (B,)), so a
    (B,) tensor must be reshaped per field. This is the failure a naive
    ``block * w`` would produce.
    """
    blk = _block()
    w = torch.tensor([0.5, 2.0, 3.0], dtype=torch.float64)
    out = scale_block(blk, w)
    for i in range(B):
        torch.testing.assert_close(out.A[i], blk.A[i] * w[i])
        torch.testing.assert_close(out.g[i], blk.g[i] * w[i])
        torch.testing.assert_close(out.cost[i], blk.cost[i] * w[i])


def test_naive_broadcast_would_have_been_wrong():
    """Guard the regression: trailing-axis broadcast != batch-axis scaling.

    P != B here, so ``A * w`` (with w of shape (B,)) is a shape error; when P
    happens to equal B it would instead be silently wrong. Either way it is not
    what we want, which is why scale_block exists.
    """
    blk = _block()
    w = torch.tensor([0.5, 2.0, 3.0], dtype=torch.float64)
    correct = scale_block(blk, w).A
    with pytest.raises(RuntimeError):
        _ = blk.A * w  # (B,P,P) * (B,) -> aligns w against the last axis, P != B
    assert not torch.allclose(correct, blk.A)


def test_rejects_wrong_shape():
    blk = _block()
    with pytest.raises(ValueError, match="0-dim or"):
        scale_block(blk, torch.ones(B + 1, dtype=torch.float64))
    with pytest.raises(ValueError, match="0-dim or"):
        scale_block(blk, torch.ones(B, 2, dtype=torch.float64))


# --- the point of the change: gradient reaches the weight -------------------

def test_gradient_flows_to_weight():
    blk = _block()
    w = torch.tensor([0.5, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    scale_block(blk, w).cost.sum().backward()
    assert w.grad is not None
    # d(sum cost)/d(w_i) = cost_i
    torch.testing.assert_close(w.grad, blk.cost)


def test_gradient_flows_at_zero_weight():
    """Nonzero gradient at w = 0: a gated-off energy can still be re-opened.

    This is what makes a weight usable as a *gate* -- the loss can prefer a term
    that is currently contributing nothing.
    """
    blk = _block()
    w = torch.zeros(B, dtype=torch.float64, requires_grad=True)
    out = scale_block(blk, w)
    assert torch.count_nonzero(out.A) == 0
    out.cost.sum().backward()
    torch.testing.assert_close(w.grad, blk.cost)
    assert (w.grad.abs() > 0).all()


# --- the energies themselves route through it -------------------------------

def test_energies_accept_tensor_weight():
    from parafit.energies.anchor3d import Anchor3DEnergy
    from parafit.energies.pose_prior import PosePriorEnergy
    from parafit.core.types import State

    J_, P_ = 4, 6
    torch.manual_seed(0)
    state = State(
        landmarks=torch.randn(B, J_, 3, dtype=torch.float64),
        landmark_jac=torch.randn(B, J_, 3, P_, dtype=torch.float64),
        batch_size=[B],
    )
    params = torch.randn(B, P_, dtype=torch.float64)
    w = torch.tensor([0.5, 2.0, 3.0], dtype=torch.float64)

    for energy in (
        Anchor3DEnergy(target=torch.randn(B, J_, 3, dtype=torch.float64)),
        PosePriorEnergy(ref=torch.zeros(B, P_, dtype=torch.float64)),
    ):
        base = energy.linearize(None, params, state)
        energy.weight = w
        out = energy.linearize(None, params, state)
        for i in range(B):
            torch.testing.assert_close(out.A[i], base.A[i] * w[i])
            torch.testing.assert_close(out.cost[i], base.cost[i] * w[i])
