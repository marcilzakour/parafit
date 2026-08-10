"""The Model ABC: any parametric articulated/rigid model the solver can fit.

A Model owns its parameter layout (``ParamSpec``), a differentiable forward
(FK + LBS producing landmarks/verts), and, optionally, an *analytic* Jacobian
of the landmarks w.r.t. the parameter vector. If ``landmark_jacobian`` returns
``None``, the solver falls back to autograd, so a new model works on day one
and can be accelerated later.

Concrete models (MANO, SMPL-X, UmeTrack, 6-DoF object) live under ``models/``
behind optional-dependency extras; the solver core imports none of them.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor

from parafit.core.types import ParamSpec, State


class Model(ABC):
    #: Parameter layout; concrete models set this in __init__.
    param_spec: ParamSpec

    @property
    def n_params(self) -> int:
        return self.param_spec.total

    @abstractmethod
    def forward(self, params: Tensor) -> State:
        """``params`` (B, P) -> State with ``landmarks`` (B, J, 3) in world frame."""

    def landmark_jacobian(self, params: Tensor, state: State) -> Optional[Tensor]:
        """d(landmarks)/d(params) as (B, J, 3, P), or ``None`` for autograd fallback."""
        return None

    # Optional reparameterization (e.g. UmeTrack normalized flexion in [0,1]
    # mapped into per-subject joint_limits via a sigmoid). Identity by default.
    def to_raw(self, params: Tensor) -> Tensor:
        return params

    def from_raw(self, raw: Tensor) -> Tensor:
        return raw
