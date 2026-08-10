"""parafit -- a pluggable analytical solver for parametric model fitting.

Fit parametric models (hands, bodies, 6-DoF objects) to multi-view evidence
with a batched Levenberg-Marquardt solver whose energies and models are
composable, and whose execution backend is swappable (torch now; CUDA /
TensorRT planned). Extracted from the UA-Fit / POEM-v2 solver.

Quick start (core only, no MANO needed)::

    import torch
    from parafit import LMSolver, ReprojectionEnergy, PinholeCameras
    from parafit.models.example import RigidKeypointModel
    # build model, cameras, target_uv ... then:
    result = LMSolver(max_iters=15).solve(model, init_params, [ReprojectionEnergy(cams, target_uv)])
"""
from parafit.core.solver import LMSolver
from parafit.core.types import GNBlock, Observations, ParamSpec, SolveResult, State
from parafit.energies.base import Energy, gauss_newton_block
from parafit.energies.reprojection import ReprojectionEnergy
from parafit.io.camera import PinholeCameras
from parafit.models.base import Model
from parafit.registry import get_energy, get_model, register_energy, register_model

__version__ = "0.1.0"

__all__ = [
    "LMSolver",
    "Model",
    "Energy",
    "ReprojectionEnergy",
    "PinholeCameras",
    "Observations",
    "ParamSpec",
    "State",
    "GNBlock",
    "SolveResult",
    "gauss_newton_block",
    "get_model",
    "get_energy",
    "register_model",
    "register_energy",
]
