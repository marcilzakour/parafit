"""tensordict batchable-API gates: the State/GNBlock/Observations tensorclasses
move, index, stack and add as one object, and an ``Observations``-driven solve
matches the plain-tensor path.
"""
import torch

from parafit import (GNBlock, LMSolver, Observations, PinholeCameras,
                     ReprojectionEnergy, State)
from parafit.models.example import RigidKeypointModel

DT = torch.float64


def _cams(B, V, DT=DT):
    K = torch.tensor([[500.0, 0, 256], [0, 500.0, 256], [0, 0, 1]], dtype=DT)
    K = K.view(1, 1, 3, 3).expand(B, V, 3, 3).contiguous()
    w2c = torch.eye(4, dtype=DT).view(1, 1, 4, 4).expand(B, V, 4, 4).contiguous().clone()
    w2c[..., 2, 3] = 0.6  # push the scene in front of every camera
    return K, w2c


def test_state_batchable():
    s = State(landmarks=torch.randn(4, 21, 3), batch_size=[4])
    assert s.to("cpu").batch_size[0] == 4
    assert s[torch.tensor([True, False, True, False])].landmarks.shape[0] == 2
    assert torch.stack([s, s], 0).batch_size == torch.Size([2, 4])
    s.landmark_jac = torch.randn(4, 21, 3, 6)  # lazily filled, like the solver does
    assert s.landmark_jac.shape == (4, 21, 3, 6)


def test_gnblock_add_and_scale():
    P = 6
    a = GNBlock(A=torch.ones(2, P, P), g=torch.ones(2, P), cost=torch.ones(2), batch_size=[2])
    b = a + a
    assert torch.allclose(b.A, 2 * a.A) and torch.allclose(b.cost, 2 * a.cost)
    c = a * 3.0  # weight scaling used by the energies
    assert torch.allclose(c.g, 3 * a.g)


def test_solve_from_observations():
    torch.manual_seed(0)
    B, V = 2, 4
    canonical = 0.08 * torch.randn(8, 3, dtype=DT)
    model = RigidKeypointModel(canonical)
    K, w2c = _cams(B, V)
    cams = PinholeCameras(K, w2c)
    gt = torch.zeros(B, 6, dtype=DT)
    gt[:, 0:3] = 0.2 * torch.randn(B, 3, dtype=DT)
    gt[:, 3:6] = 0.03 * torch.randn(B, 3, dtype=DT)
    target_uv, _ = cams.project_and_jac(model.forward(gt).landmarks)

    obs = Observations(K=K, w2c=w2c, target_uv=target_uv, batch_size=[B])
    energy = ReprojectionEnergy.from_observations(obs)
    res = LMSolver(max_iters=30, damping=1e-3).solve(model, torch.zeros(B, 6, dtype=DT), [energy])
    err_mm = (model.forward(res.params).landmarks - model.forward(gt).landmarks).norm(dim=-1).mean().item() * 1000
    print(f"[obs-solve] landmark-err={err_mm:.4f}mm")
    assert err_mm < 1e-3, err_mm


if __name__ == "__main__":
    test_state_batchable()
    test_gnblock_add_and_scale()
    test_solve_from_observations()
    print("OK")
