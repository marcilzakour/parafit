"""Shared types for the parafit solver core.

The batched objects (``State``, ``GNBlock``, ``Observations``) are
``tensordict`` tensorclasses with a leading batch dimension ``B``. That makes
them first-class batchable containers: move a whole solve to another device with
``state.to("cuda")``, index a subset of the batch with ``state[mask]``, stack
independent solves with ``torch.stack``, and add two Gauss-Newton blocks with
``block_a + block_b`` (elementwise over every field). ``ParamSpec`` is plain
layout metadata (not batched), and ``SolveResult`` is a light return container.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from tensordict import tensorclass
from torch import Tensor


@dataclass
class ParamSpec:
    """Layout of the optimization vector as named contiguous blocks.

    ``blocks`` maps a name -> (start, size). ``total`` is the full DOF. Example
    for MANO: {"pose": (0, 48), "trans": (48, 3)}, total=51. For a rigid body:
    {"rot": (0, 3), "trans": (3, 3)}, total=6.
    """

    blocks: dict[str, tuple[int, int]]

    @property
    def total(self) -> int:
        return sum(size for _, size in self.blocks.values())

    def slice(self, name: str) -> slice:
        start, size = self.blocks[name]
        return slice(start, start + size)


@tensorclass
class State:
    """Output of ``Model.forward``: everything an energy might read (batch ``B``).

    ``landmarks`` (B, J, 3) are the fittable 3D keypoints (world frame).
    ``verts`` / ``transforms`` are optional (mesh / contact energies).
    ``landmark_jac`` (B, J, 3, P) is filled lazily by the solver so energies that
    need d(landmarks)/d(params) share one computation.
    """

    landmarks: Tensor
    verts: Optional[Tensor] = None
    transforms: Optional[Tensor] = None
    landmark_jac: Optional[Tensor] = None


@tensorclass
class GNBlock:
    """One energy's contribution to the normal equations (batch ``B``).

    ``A`` (B, P, P), ``g`` (B, P), ``cost`` (B,). Blocks are summed across
    energies; tensorclass ``+`` adds every field elementwise, so
    ``block_a + block_b`` is exactly the accumulation the solver needs.
    """

    A: Tensor
    g: Tensor
    cost: Tensor


@tensorclass
class Observations:
    """A batch of ``V`` calibrated multi-view 2D observations (batch ``B``).

    Bundles the camera calibration and the target evidence so a multi-view fit
    moves/indexes/stacks as one object. ``K`` (B, V, 3, 3), ``w2c`` (B, V, 4, 4),
    ``target_uv`` (B, V, J, 2), ``precision`` optional (B, V, J, 2, 2) block /
    (B, V, J) scalar. Consumed by ``ReprojectionEnergy.from_observations``.
    """

    K: Tensor
    w2c: Tensor
    target_uv: Tensor
    precision: Optional[Tensor] = None


@dataclass
class SolveResult:
    """Return container for a solve. ``params`` (B, P) is the fitted vector,
    ``state`` the final :class:`State`, ``diagnostics`` a free-form dict
    (final cost, damping trajectory, optional cost history)."""

    params: Tensor
    state: State
    diagnostics: dict = field(default_factory=dict)
