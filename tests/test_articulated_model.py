"""TwoPartRevolute: forward convention, analytic Jacobian, and a fit.

The analytic Jacobian is checked against finite differences because the two
rotation-like blocks are asymmetric and easy to get wrong in opposite ways: the
global rotation needs the SO(3) right-Jacobian correction (additive updates),
while the articulation, being a one-parameter subgroup about a fixed axis, needs
none. A sign or a missing ``J_r`` in either place still produces a plausible
descent direction, so only finite differences catch it.
"""
from __future__ import annotations

import torch

from parafit.core.solver import LMSolver
from parafit.energies.anchor3d import Anchor3DEnergy
from parafit.models.articulated import TwoPartRevolute

DT = torch.float64
KM, KF, P = 6, 5, 7


def _model(seed: int = 0) -> TwoPartRevolute:
    g = torch.Generator().manual_seed(seed)
    return TwoPartRevolute(
        kp_moving=torch.randn(KM, 3, generator=g, dtype=DT) * 0.05,
        kp_fixed=torch.randn(KF, 3, generator=g, dtype=DT) * 0.05,
        axis=torch.tensor([0.0, 0.0, -1.0], dtype=DT),
    )


def _params(B: int = 3, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    rot = torch.randn(B, 3, generator=g, dtype=DT) * 0.6
    trans = torch.randn(B, 3, generator=g, dtype=DT) * 0.2
    art = torch.rand(B, 1, generator=g, dtype=DT) * 2.0   # ARCTIC _use_ spans ~1-2.3 rad
    return torch.cat([rot, trans, art], dim=1)


def _reference_forward(m: TwoPartRevolute, params: torch.Tensor) -> torch.Tensor:
    """Independent restatement of the ARCTIC convention, via matrix_exp.

    Deliberately does not share code with the model: it uses torch.matrix_exp
    instead of the series-based Rodrigues in lie.py, so agreement is evidence
    about the convention rather than about a shared helper.
    """
    rot, trans, art = params[:, :3], params[:, 3:6], params[:, 6:7]
    axis = m.axis.to(params)

    def hat(v):
        z = torch.zeros(v.shape[0], dtype=v.dtype, device=v.device)
        return torch.stack([
            torch.stack([z, -v[:, 2], v[:, 1]], -1),
            torch.stack([v[:, 2], z, -v[:, 0]], -1),
            torch.stack([-v[:, 1], v[:, 0], z], -1),
        ], dim=-2)

    R_a = torch.matrix_exp(hat(art * axis))
    R_g = torch.matrix_exp(hat(rot))
    moving = torch.einsum("bij,kj->bki", R_a, m.kp_moving.to(params))
    both = torch.cat([moving, m.kp_fixed.to(params).expand(params.shape[0], -1, -1)], 1)
    return torch.einsum("bij,bkj->bki", R_g, both) + trans[:, None, :]


# --- forward convention -----------------------------------------------------

def test_forward_matches_reference_convention():
    m, p = _model(), _params()
    torch.testing.assert_close(m.forward(p).landmarks, _reference_forward(m, p))


def test_zero_params_is_canonical():
    m = _model()
    p = torch.zeros(1, P, dtype=DT)
    expected = torch.cat([m.kp_moving, m.kp_fixed], 0)[None]
    torch.testing.assert_close(m.forward(p).landmarks, expected)


def test_articulation_moves_only_the_moving_part():
    """The fixed part must be exactly invariant to alpha -- the property that
    makes ``art`` identifiable at all."""
    m = _model()
    p0 = _params(B=1)
    p1 = p0.clone()
    p1[:, 6] += 0.7
    X0, X1 = m.forward(p0).landmarks, m.forward(p1).landmarks
    torch.testing.assert_close(X0[:, m.n_moving:], X1[:, m.n_moving:])
    assert (X0[:, : m.n_moving] - X1[:, : m.n_moving]).abs().max() > 1e-3


def test_rigid_limit_is_the_six_dof_body():
    """Freezing alpha leaves a pure SE(3) body: rigid is a setting, not a model."""
    m = _model()
    p = _params(B=2)
    p[:, 6] = 0.0
    X = m.forward(p).landmarks
    canon = torch.cat([m.kp_moving, m.kp_fixed], 0).to(DT)
    # Pairwise distances are preserved under a rigid transform.
    d_canon = torch.cdist(canon[None], canon[None])
    torch.testing.assert_close(torch.cdist(X, X), d_canon.expand_as(torch.cdist(X, X)))


# --- gradient integrity (CLAUDE.md gate 7) ----------------------------------

def test_analytic_jacobian_matches_finite_difference():
    m, p = _model(), _params(B=2)
    state = m.forward(p)
    Ja = m.landmark_jacobian(p, state)                     # (B,J,3,7)

    eps = 1e-6
    Jn = torch.zeros_like(Ja)
    for k in range(P):
        dp = torch.zeros_like(p)
        dp[:, k] = eps
        Xp = m.forward(p + dp).landmarks
        Xm = m.forward(p - dp).landmarks
        Jn[..., k] = (Xp - Xm) / (2 * eps)

    torch.testing.assert_close(Ja, Jn, rtol=1e-6, atol=1e-8)


def test_jacobian_blocks_are_individually_right():
    """Localize a failure: check each block separately so a regression names
    which of the three derivations broke."""
    m, p = _model(), _params(B=2)
    Ja = m.landmark_jacobian(p, m.forward(p))
    eps = 1e-6
    for name, sl in (("rot", slice(0, 3)), ("trans", slice(3, 6)), ("art", slice(6, 7))):
        for k in range(sl.start, sl.stop):
            dp = torch.zeros_like(p)
            dp[:, k] = eps
            num = (m.forward(p + dp).landmarks - m.forward(p - dp).landmarks) / (2 * eps)
            torch.testing.assert_close(
                Ja[..., k], num, rtol=1e-6, atol=1e-8,
                msg=lambda s, n=name, kk=k: f"{n} block, param {kk}: {s}",
            )


def test_fixed_part_has_exactly_zero_articulation_gradient():
    m, p = _model(), _params(B=2)
    Ja = m.landmark_jacobian(p, m.forward(p))
    assert torch.count_nonzero(Ja[:, m.n_moving:, :, 6]) == 0


def test_jacobian_at_zero_rotation_is_finite():
    """theta -> 0 is the series-expansion branch in lie.py; guard against NaN."""
    m = _model()
    p = torch.zeros(2, P, dtype=DT)
    Ja = m.landmark_jacobian(p, m.forward(p))
    assert torch.isfinite(Ja).all()


# --- single-batch overfit (CLAUDE.md gate 1) --------------------------------

def test_overfits_single_batch_to_zero():
    m = _model()
    target_params = _params(B=1, seed=7)
    target = m.forward(target_params).landmarks

    init = target_params.clone()
    init[:, :3] += 0.15
    init[:, 3:6] += 0.05
    init[:, 6:] += 0.3

    res = LMSolver(max_iters=60).solve(m, init, [Anchor3DEnergy(target=target)])
    err = (m.forward(res.params).landmarks - target).norm(dim=-1).max()
    assert err < 1e-6, f"did not overfit: max vertex error {err:.3e} m"
