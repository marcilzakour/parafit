"""ContactSDFEnergy -- hand/object no-penetration via a signed-distance field. [extra: contact]

PORT TARGET (thesis-hope):
  mvhpe/POEM-v2/lib/models/dovf/analytic_fitter_unc.py:313  : _object_sdf_block (analytic)
                                                     :388  : _object_sdf_block_autograd
  action_segmentation/hoitcn/hoi_lm.py:131                 : ObjectSDF.query (mesh2sdf grid)

Contract: a ``push_fn(verts (B,Nv,3)) -> outward_push (B,Nv,3)`` (or an SDF whose
``query(pts) -> signed_distance`` is differentiable). The residual is the
penetration depth along the SDF normal over penetrating vertices; the Jacobian
chains through the model's vertex Jacobian. Needs mesh2sdf / trimesh -> [contact].
"""
from __future__ import annotations

from parafit.core.types import GNBlock, State
from parafit.energies.base import Energy, WeightLike


class ContactSDFEnergy(Energy):
    name = "contact_sdf"

    def __init__(self, push_fn, weight: WeightLike = 1.0, max_verts: int | None = None):
        self.push_fn = push_fn
        self.weight = weight         # float, or Tensor 0-dim / (B,); apply via scale_block
        self.max_verts = max_verts

    def linearize(self, model, params, state: State) -> GNBlock:  # pragma: no cover - port stub
        raise NotImplementedError(
            "Port _object_sdf_block: push = push_fn(state.verts); residual = penetration "
            "depth along the SDF normal; chain through the model vertex Jacobian. "
            "See analytic_fitter_unc.py:313."
        )
