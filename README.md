# parafit

**A pluggable analytical solver for parametric model fitting.**

`parafit` fits parametric models -- hands (MANO, UmeTrack), bodies (SMPL-X), and
6-DoF objects -- to multi-view evidence with a batched, differentiable
Levenberg-Marquardt solver. Its **energies**, **models**, and **execution
backend** are all pluggable. It is the shared solver core extracted from the
UA-Fit / POEM-v2 line of work, so UA-Fit, FuseFit, SHOW3D, and future projects
depend on one implementation instead of re-porting the same math.

## Why

Three ideas, cleanly separated:

- **Models** (`parafit.models`) -- a parametric model owns its parameter layout,
  a differentiable forward (FK+LBS -> landmarks/verts), and an optional
  *analytic* Jacobian (autograd fallback otherwise).
- **Energies** (`parafit.energies`) -- each residual term contributes a
  Gauss-Newton block `(A, g, cost)`. Compose them explicitly; no global state.
- **Solver** (`parafit.LMSolver`) -- adaptive LM with per-sample accept/reject,
  swappable backend (torch now; CUDA / TensorRT planned via the fixed-iteration
  exportable path).

## Install

Core is torch-only; pull only the extras you need:

```bash
pip install parafit                 # core solver + reprojection/anchor/pose-prior energies
pip install "parafit[mano]"         # + MANO hand model
pip install "parafit[umetrack]"     # + UmeTrack hand model
pip install "parafit[contact]"      # + object-SDF contact energy
pip install "parafit[all]"          # everything
```

With **uv** (recommended; resolves the git-only manotorch dep):

```bash
uv pip install "parafit[mano]"
# or in a project:  uv add parafit --extra mano
```

Install torch for your CUDA build first. MANO weights are not shipped (license):
point the loader at your local `mano_v1_2` assets.

## Quick start (core, no MANO needed)

```python
import torch
from parafit import LMSolver, ReprojectionEnergy, PinholeCameras
from parafit.models.example import RigidKeypointModel

model = RigidKeypointModel(canonical_points)          # (J,3)
cams = PinholeCameras(K, w2c)                          # (B,V,3,3), (B,V,4,4)
energy = ReprojectionEnergy(cams, target_uv)          # (B,V,J,2)
result = LMSolver(max_iters=15).solve(model, init_params, [energy])
print(result.params, result.diagnostics["final_cost"])
```

Run the end-to-end test (multi-view rigid fit + finite-difference Jacobian check):

```bash
python tests/test_core_overfit.py
```

## Status

v0.0.1 -- core solver + reprojection / 3D-anchor / pose-prior energies are
implemented and tested. `ManoModel`, `UmeTrackModel`, and `ContactSDFEnergy` are
grounded port stubs (each names its source file in the thesis-hope tree).
Roadmap: MANO/UmeTrack ports -> multi-body scene (hand+object) -> CUDA/TensorRT
backend. See `docs/parafit-library-design.md`.

## License

MIT.
