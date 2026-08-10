"""Shared dataclasses for the parafit solver core."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

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


@dataclass
class State:
    """Output of ``Model.forward``: everything an energy might read.

    ``landmarks`` (B, J, 3) are the fittable 3D keypoints (world frame).
    ``verts``/``transforms`` are optional (mesh energies, contact).
    ``landmark_jac`` (B, J, 3, P) is filled lazily by the solver so energies
    that need d(landmarks)/d(params) share one computation.
    """

    landmarks: Tensor
    verts: Optional[Tensor] = None
    transforms: Optional[Tensor] = None
    landmark_jac: Optional[Tensor] = None
    extra: dict = field(default_factory=dict)


@dataclass
class GNBlock:
    """One energy's contribution to the normal equations.

    ``A`` (B, P, P), ``g`` (B, P), ``cost`` (B,). Summed across energies.
    """

    A: Tensor
    g: Tensor
    cost: Tensor

    def __add__(self, other: "GNBlock") -> "GNBlock":
        return GNBlock(self.A + other.A, self.g + other.g, self.cost + other.cost)


@dataclass
class SolveResult:
    params: Tensor
    state: State
    diagnostics: dict = field(default_factory=dict)
