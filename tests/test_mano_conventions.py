"""MANO convention gates: ``flat_hand_mean`` / ``center_idx`` are not preferences.

Every producer of MANO parameters commits to a convention, and mismatching it is
a **silent ~100 mm error** -- the fit still converges, it just converges to the
wrong hand. These tests pin both supported conventions and check that the
analytic Jacobian survives the non-default one, since ``center_idx=None`` changes
the forward and could in principle invalidate a Jacobian derived under
``center_idx=0``.

  * ``flat_hand_mean=True,  center_idx=0``    -- UA-Fit (parafit default)
  * ``flat_hand_mean=False, center_idx=None`` -- standard MANO: ARCTIC, and the
    HaMeR / WiLoR / HaWoR family
"""
import os

import pytest
import torch

_MANO_ROOT = os.environ.get("MANO_ASSETS_ROOT", "/code/ua-fit-code/assets/mano_v1_2")
_ARCTIC = "/data/arctic/arctic_data/data/processed_seqs/s01/box_use_01.npy"


def _model(**kw):
    try:
        from parafit.models.mano import ManoModel
    except Exception as e:  # pragma: no cover
        pytest.skip(f"manotorch unavailable: {e}")
    if not os.path.exists(os.path.join(_MANO_ROOT, "models", "MANO_RIGHT.pkl")):
        pytest.skip(f"MANO assets not found under {_MANO_ROOT}")
    return ManoModel(mano_assets_root=_MANO_ROOT, **kw)


def _arctic_frames(n=4):
    import numpy as np
    if not os.path.exists(_ARCTIC):
        pytest.skip("ARCTIC not available on this machine")
    d = np.load(_ARCTIC, allow_pickle=True).item()
    p, gt = d["params"], d["world_coord"]["verts.right"]
    i = np.linspace(0, len(gt) - 1, n).astype(int)
    pose = torch.tensor(np.concatenate([p["rot_r"][i], p["pose_r"][i]], 1), dtype=torch.float32)
    trans = torch.tensor(p["trans_r"][i], dtype=torch.float32)
    betas = torch.tensor(p["shape_r"][i], dtype=torch.float32)
    return torch.cat([pose, trans], 1), betas, torch.tensor(gt[i], dtype=torch.float32)


def test_standard_mano_reproduces_arctic_exactly():
    """The load-bearing check: ARCTIC's own params must round-trip to its own verts."""
    theta, betas, gt = _arctic_frames()
    m = _model(betas=betas, center_idx=None, flat_hand_mean=False)
    err = (m.forward(theta).verts - gt).norm(dim=-1)
    assert err.max() * 1000 < 0.5, f"max {err.max()*1000:.3f} mm under standard MANO"


def test_wrong_convention_is_catastrophic_not_subtle():
    """Guards the docstring's claim, and documents the failure magnitude.

    If this ever stops being large, the two conventions have converged and the
    warning in ManoModel.__init__ should be revisited.
    """
    theta, betas, gt = _arctic_frames()
    m = _model(betas=betas)                      # defaults: UA-Fit convention
    err = (m.forward(theta).verts - gt).norm(dim=-1)
    assert err.mean() * 1000 > 50, (
        "wrong convention should be a ~100 mm error; got "
        f"{err.mean()*1000:.1f} mm -- did the default change?")


def test_default_convention_is_unchanged():
    """Backward compatibility: the default must remain the UA-Fit convention."""
    m = _model()
    assert m.flat_hand_mean is True
    assert m.center_idx == 0


def _per_param_cosines(m):
    """Cosine between the analytic and finite-difference Jacobian, COLUMN BY COLUMN."""
    torch.manual_seed(0)
    theta = torch.zeros(2, m.n_params)
    theta[:, :48] = torch.randn(2, 48) * 0.05
    theta[:, 48:] = torch.randn(2, 3) * 0.02
    Ja = m.landmark_jacobian(theta, m.forward(theta))
    assert Ja is not None and torch.isfinite(Ja).all()
    eps, cs = 1e-4, []
    for k in range(m.n_params):
        dp = torch.zeros_like(theta)
        dp[:, k] = eps
        num = (m.forward(theta + dp).landmarks - m.forward(theta - dp).landmarks) / (2 * eps)
        a, n = Ja[..., k].flatten(), num.flatten()
        cs.append((torch.dot(a, n) / (a.norm() * n.norm()).clamp_min(1e-12)).item())
    return torch.tensor(cs)


@pytest.mark.parametrize("center_idx,flat", [(0, True), (None, False)])
def test_jacobian_is_a_good_aggregate_direction(center_idx, flat):
    """``center_idx=None`` changes the forward, so the Jacobian is re-checked there.

    Asserted on the MEDIAN column, not the mean over a flattened Jacobian: the
    flattened form is dominated by high-magnitude columns and hides weak ones
    entirely (see the next test).
    """
    cs = _per_param_cosines(_model(center_idx=center_idx, flat_hand_mean=flat))
    assert cs.median() > 0.95, f"median column cosine {cs.median():.3f}"


def test_weak_jacobian_columns_are_documented_not_hidden():
    """The analytic Jacobian has genuinely weak COLUMNS in both conventions.

    Measured: worst column is param 9 at ~0.28 (UA-Fit) / ~0.22 (standard MANO),
    with 4/51 and 9/51 columns below 0.9 respectively. This is the omitted
    pose-blendshape term, and it means those DOFs get a near-orthogonal descent
    direction -- slower convergence, not divergence, since LM damping absorbs it.

    This test exists so the fact stays visible. The existing aggregate test
    (``cos.mean() > 0.98`` over a flattened Jacobian) passes comfortably while
    a column sits at 0.28, which is exactly the kind of averaging that turns a
    real weakness into a green tick.
    """
    worst = {}
    for ci, fl in [(0, True), (None, False)]:
        cs = _per_param_cosines(_model(center_idx=ci, flat_hand_mean=fl))
        worst[(ci, fl)] = (cs.min().item(), int(cs.argmin()), int((cs < 0.9).sum()))
    for key, (mn, arg, nbad) in worst.items():
        assert mn > 0.15, f"{key}: worst column {mn:.3f} at param {arg} ({nbad}/51 below 0.9)"
    # The non-default convention must not be materially worse than the default.
    assert worst[(None, False)][0] > 0.5 * worst[(0, True)][0]


def test_translation_block_is_identity_in_both_conventions():
    """d(landmarks)/d(trans) is the identity regardless of centring."""
    for center_idx, flat in [(0, True), (None, False)]:
        m = _model(center_idx=center_idx, flat_hand_mean=flat)
        theta = torch.zeros(1, m.n_params)
        Ja = m.landmark_jacobian(theta, m.forward(theta))
        eye = torch.eye(3).view(1, 1, 3, 3).expand(1, m.num_joints, 3, 3)
        torch.testing.assert_close(Ja[..., 48:51], eye, rtol=1e-4, atol=1e-5)
