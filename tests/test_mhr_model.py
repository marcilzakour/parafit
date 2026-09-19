"""MhrModel gates (skipped when MHR / pymomentum are not installed).

  * forward reproduces mhr.mhr.MHR (correctives on and off) to < 2e-3 cm
  * exact (autograd) Jacobian matches central finite differences (rel < 1e-4)
  * rigid-ride Jacobian is exact on joint landmarks (rel < 1e-5)
  * reduction: with correctives OFF, "lbs" == autograd exact (rel < 1e-8)
  * structured exact ("exact_mlp") == autograd exact with correctives ON (rel < 1e-5)
"""
import pytest
import torch

pym = pytest.importorskip("pymomentum.geometry")   # must be imported before torch does CUDA work
mhr = pytest.importorskip("mhr.mhr")

from parafit.models.mhr import MhrModel, N_IDENTITY, N_MODEL  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _params(model, B, gen, sigma=0.6):
    p = torch.zeros(B, N_MODEL + N_IDENTITY, device=model.device, dtype=model.dtype)
    p[:, :N_MODEL][:, model.rot_mask] = sigma * torch.randn(B, int(model.rot_mask.sum()), generator=gen).to(p)
    p[:, :N_MODEL][:, model.scale_mask] = 0.05 * torch.randn(B, int(model.scale_mask.sum()), generator=gen).to(p)
    p[:, N_MODEL:] = (0.7 * torch.randn(B, N_IDENTITY, generator=gen)).clamp(-2, 2).to(p)
    return p


def _rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


@pytest.fixture(scope="module")
def model64():
    return MhrModel(device=DEV, dtype=torch.float64, jac_mode="exact", correctives=True)


def test_forward_matches_mhr():
    gen = torch.Generator().manual_seed(0)
    m = MhrModel(device=DEV, jac_mode="rigid", correctives=True, full_mesh=True)
    ref = mhr.MHR.from_files(device=torch.device(DEV), lod=1)
    p = _params(m, 4, gen)
    for corr in (True, False):
        m.correctives = corr
        with torch.no_grad():
            v_ref, s_ref = ref(p[:, N_MODEL:], p[:, :N_MODEL], None, apply_correctives=corr)
            st = m.forward(p)
        assert (st.verts - v_ref).abs().max().item() < 2e-3
        assert (st.landmarks[:, :m.n_joints] - s_ref[..., :3]).abs().max().item() < 1e-3


def test_jacobians(model64):
    m = model64
    gen = torch.Generator().manual_seed(1)
    p = _params(m, 1, gen)
    f = lambda q: m.forward(q.unsqueeze(0)).landmarks.squeeze(0)
    P = p.shape[1]
    E = torch.eye(P, device=p.device, dtype=p.dtype)
    eps = 1e-3
    Jfd = torch.stack([(f(p[0] + eps * E[i]) - f(p[0] - eps * E[i])) / (2 * eps) for i in range(P)], -1)
    st = m.forward(p)
    m.jac_mode = "exact"; Je = m.landmark_jacobian(p, st)[0]
    assert _rel(Je, Jfd) < 1e-4
    J = m.n_joints
    m.jac_mode = "rigid"; Jr = m.landmark_jacobian(p, st)[0]
    assert _rel(Jr[:J], Jfd[:J]) < 1e-5                       # joints: rigid-ride is exact
    m.jac_mode = "exact_mlp"; Jm = m.landmark_jacobian(p, st)[0]
    assert _rel(Jm, Je) < 1e-5                                 # structured exact == autograd exact
    m.correctives = False
    st0 = m.forward(p)
    m.jac_mode = "exact"; Je0 = m.landmark_jacobian(p, st0)[0]
    m.jac_mode = "lbs"; Jl0 = m.landmark_jacobian(p, st0)[0]
    assert _rel(Jl0, Je0) < 1e-8                               # reduction check
    m.correctives = True
    m.jac_mode = "exact"
