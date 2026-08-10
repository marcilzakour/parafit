"""End-to-end core validation on a multi-view rigid-fit problem (no MANO assets).

Covers the universal gates on the solver engine itself:
  - gate #7 gradient integrity: analytic landmark Jacobian vs finite differences
  - gate #1 single-batch overfit: LM drives multi-view reprojection to ~0
Run:  python tests/test_core_overfit.py   (also works under pytest)
"""
import torch

from parafit import LMSolver, PinholeCameras, ReprojectionEnergy
from parafit.models.example import RigidKeypointModel

torch.manual_seed(0)
DT = torch.float64


def look_at_w2c(cam_pos):
    C = torch.tensor(cam_pos, dtype=DT)
    z = -C / C.norm()
    up = torch.tensor([0.0, 1.0, 0.0], dtype=DT)
    x = torch.cross(up, z, dim=0); x = x / x.norm()
    y = torch.cross(z, x, dim=0)
    Rcw = torch.stack([x, y, z], dim=0)     # world->cam
    t = -Rcw @ C
    w2c = torch.eye(4, dtype=DT)
    w2c[:3, :3] = Rcw
    w2c[:3, 3] = t
    return w2c


def build_problem(B=2):
    canonical = 0.08 * torch.randn(8, 3, dtype=DT)
    model = RigidKeypointModel(canonical)
    cam_positions = [(1.0, 0.0, 0.3), (-1.0, 0.0, 0.3), (0.0, 1.0, 0.3), (0.0, -1.0, 0.3)]
    w2c = torch.stack([look_at_w2c(c) for c in cam_positions], dim=0)  # (V,4,4)
    V = w2c.shape[0]
    K = torch.tensor([[500.0, 0, 256], [0, 500.0, 256], [0, 0, 1]], dtype=DT)
    K = K.view(1, 1, 3, 3).expand(B, V, 3, 3).contiguous()
    w2c = w2c.view(1, V, 4, 4).expand(B, V, 4, 4).contiguous()
    cams = PinholeCameras(K, w2c)
    # Ground-truth pose.
    gt = torch.zeros(B, 6, dtype=DT)
    gt[:, 0:3] = 0.3 * torch.randn(B, 3, dtype=DT)     # rotation
    gt[:, 3:6] = 0.05 * torch.randn(B, 3, dtype=DT)    # translation
    gt_landmarks = model.forward(gt).landmarks
    target_uv, _ = cams.project_and_jac(gt_landmarks)
    return model, cams, gt, gt_landmarks, target_uv


def test_jacobian_finite_difference():
    model, *_ = build_problem(B=2)
    theta = 0.2 * torch.randn(2, 6, dtype=DT)
    state = model.forward(theta)
    Jan = model.landmark_jacobian(theta, state)        # (B,J,3,6)
    eps = 1e-6
    Jnum = torch.zeros_like(Jan)
    for k in range(6):
        d = torch.zeros_like(theta); d[:, k] = eps
        lp = model.forward(theta + d).landmarks
        lm = model.forward(theta - d).landmarks
        Jnum[..., k] = (lp - lm) / (2 * eps)
    err = (Jan - Jnum).abs().max().item()
    print(f"[jac] max |analytic - finite-diff| = {err:.2e}")
    assert err < 1e-6, err


def test_multiview_overfit():
    model, cams, gt, gt_landmarks, target_uv = build_problem(B=2)
    init = torch.zeros(2, 6, dtype=DT)
    energy = ReprojectionEnergy(cams, target_uv)
    res = LMSolver(max_iters=30, damping=1e-3, record=True).solve(model, init, [energy])
    final = model.forward(res.params).landmarks
    mpjpe_mm = (final - gt_landmarks).norm(dim=-1).mean().item() * 1000.0
    reproj_px = (cams.project_and_jac(final)[0] - target_uv).norm(dim=-1).mean().item()
    print(f"[overfit] final cost={res.diagnostics['final_cost'].mean():.3e}  "
          f"reproj={reproj_px:.4f}px  landmark-err={mpjpe_mm:.4f}mm")
    print(f"[overfit] cost history: {[f'{c:.2e}' for c in res.diagnostics['cost_history'][:8]]} ...")
    assert reproj_px < 1e-3, reproj_px
    assert mpjpe_mm < 1e-3, mpjpe_mm


if __name__ == "__main__":
    test_jacobian_finite_difference()
    test_multiview_overfit()
    print("OK")
