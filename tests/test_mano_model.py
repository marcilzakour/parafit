"""ManoModel gates (needs parafit[mano] + local MANO assets).

Skips cleanly if manotorch or the MANO pkls are unavailable. Validates:
  - forward produces 21 OpenPose joints in a sane scale;
  - the analytic landmark Jacobian is a good GN direction (high cosine vs
    autograd; loose finite-difference agreement -- the pose-blendshape term is
    deliberately omitted, so it is an approximation, not exact);
  - a multi-view reprojection overfit converges to a low landmark error.
"""
import os

import pytest
import torch

from parafit import LMSolver, PinholeCameras, ReprojectionEnergy

# Point at a local mano_v1_2 (models/MANO_*.pkl). Override with MANO_ASSETS_ROOT.
_MANO_ROOT = os.environ.get("MANO_ASSETS_ROOT", "/code/ua-fit-code/assets/mano_v1_2")


def _make_model():
    try:
        from parafit.models.mano import ManoModel
    except Exception as e:  # pragma: no cover
        pytest.skip(f"manotorch unavailable: {e}")
    if not os.path.exists(os.path.join(_MANO_ROOT, "models", "MANO_RIGHT.pkl")):
        pytest.skip(f"MANO assets not found under {_MANO_ROOT}")
    return ManoModel(mano_assets_root=_MANO_ROOT)


def _cams(B, V):
    K = torch.tensor([[600.0, 0, 128], [0, 600.0, 128], [0, 0, 1]]).view(1, 1, 3, 3).expand(B, V, 3, 3).contiguous()
    poss = [(0.0, 0.0, 0.5), (0.4, 0.0, 0.5), (0.0, 0.4, 0.5), (-0.3, 0.2, 0.5)][:V]
    w2c = []
    for c in poss:
        C = torch.tensor(c)
        z = -C / C.norm()
        up = torch.tensor([0.0, 1.0, 0.0])
        x = torch.cross(up, z, dim=0); x = x / x.norm()
        y = torch.cross(z, x, dim=0)
        R = torch.stack([x, y, z], 0)
        M = torch.eye(4); M[:3, :3] = R; M[:3, 3] = -R @ C
        w2c.append(M)
    w2c = torch.stack(w2c, 0).view(1, V, 4, 4).expand(B, V, 4, 4).contiguous()
    return PinholeCameras(K.float(), w2c.float())


def test_forward_shape_and_scale():
    m = _make_model()
    B = 2
    theta = torch.zeros(B, 51, dtype=torch.float32)
    st = m.forward(theta)
    assert st.landmarks.shape == (B, 21, 3)
    span = (st.landmarks.max(1).values - st.landmarks.min(1).values).norm(dim=-1)
    assert (span > 0.05).all() and (span < 0.4).all(), span  # a hand is ~10-20 cm


def test_analytic_jacobian_is_good_gn_direction():
    m = _make_model()
    B = 2
    torch.manual_seed(0)
    theta = torch.zeros(B, 51, dtype=torch.float32)
    theta[:, :48] = 0.1 * torch.randn(B, 48, dtype=torch.float32)
    st = m.forward(theta)
    Jan = m.landmark_jacobian(theta, st)                     # (B,J,3,51)
    # finite-difference reference (MANO uses boolean-mask ops that vmap/jacrev
    # cannot batch, so autograd-jac is unavailable; FD is the ground truth).
    eps = 1e-4
    Jfd = torch.zeros_like(Jan)
    for k in range(51):
        d = torch.zeros_like(theta); d[:, k] = eps
        lp = m.forward(theta + d).landmarks
        lm = m.forward(theta - d).landmarks
        Jfd[..., k] = (lp - lm) / (2 * eps)
    cos = torch.nn.functional.cosine_similarity(Jan.reshape(B, -1), Jfd.reshape(B, -1), dim=-1)
    rel = (Jan - Jfd).norm() / Jfd.norm()
    print(f"[mano-jac] cos(analytic, finite-diff)={cos.mean():.4f}  rel-err={rel:.3f}")
    # analytic omits the (small) pose-blendshape term by design, so it is an
    # approximate GN direction: high cosine, modest relative error.
    assert cos.mean() > 0.98, cos
    assert rel < 0.15, rel


def test_multiview_overfit():
    m = _make_model()
    B, V = 1, 4
    torch.manual_seed(1)
    gt = torch.zeros(B, 51, dtype=torch.float32)
    gt[:, :48] = 0.1 * torch.randn(B, 48, dtype=torch.float32)
    gt[:, 48:51] = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    cams = _cams(B, V)
    target_uv, _ = cams.project_and_jac(m.forward(gt).landmarks)
    energy = ReprojectionEnergy(cams, target_uv)
    init = torch.zeros(B, 51, dtype=torch.float32)
    res = LMSolver(max_iters=60, damping=1e-3).solve(m, init, [energy])
    err_mm = (m.forward(res.params).landmarks - m.forward(gt).landmarks).norm(dim=-1).mean().item() * 1000
    print(f"[mano-overfit] landmark-err={err_mm:.3f}mm")
    assert err_mm < 3.0, err_mm


if __name__ == "__main__":
    test_forward_shape_and_scale()
    test_analytic_jacobian_is_good_gn_direction()
    test_multiview_overfit()
    print("OK")
