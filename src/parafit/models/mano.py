"""ManoModel -- MANO hand under the parafit Model interface. [extra: mano]

PORT TARGET (thesis-hope):
  mvhpe/POEM-v2/lib/models/dovf/analytic_fitter.py
    - mano_kinematic_jac(...)      : analytic articulated Jacobian (SO(3) right-Jac)
    - _joints_pose_jac(...)        : autograd Jacobian alternative
  mvhpe/POEM-v2/lib/models/dovf_mano_mv.py:149  : ManoLayer construction
  mvhpe/POEM-v2/lib/utils/transform.py:908      : mano_to_openpose (16 joints + 5 tips -> 21)

Parameter layout: theta = [pose(48) | trans(3)] = 51 DOF; shape betas(10) is a
fixed side input (predicted by a compact head, not optimized here). Requires
ManoLayer(flat_hand_mean=True). MANO weights are NOT shipped (license): the
loader takes a path to the user's mano_v1_2 assets.
"""
from __future__ import annotations

from parafit.core.types import ParamSpec, State
from parafit.models.base import Model
from parafit.registry import register_model


@register_model("mano", extra="mano")
def _build_mano(*args, **kwargs) -> "ManoModel":
    return ManoModel(*args, **kwargs)


class ManoModel(Model):
    def __init__(self, mano_assets_root: str, betas=None, num_joints: int = 21, center_idx: int = 0):
        from manotorch.manolayer import ManoLayer  # raises ImportError -> parafit[mano]

        self.mano_layer = ManoLayer(
            joint_rot_mode="axisang",
            use_pca=False,
            mano_assets_root=mano_assets_root,
            center_idx=center_idx,
            flat_hand_mean=True,
        )
        self.betas = betas
        self.num_joints = num_joints
        self.center_idx = center_idx
        self.param_spec = ParamSpec({"pose": (0, 48), "trans": (48, 3)})

    def forward(self, params) -> State:  # pragma: no cover - port stub
        raise NotImplementedError(
            "Port MANO forward: out = mano_layer(pose, betas); joints via "
            "mano_to_openpose(J_regressor, out.verts)[:, :num_joints] + trans. "
            "See mvhpe/POEM-v2/lib/models/dovf/analytic_fitter.py."
        )

    def landmark_jacobian(self, params, state):  # pragma: no cover - port stub
        # Port mano_kinematic_jac (analytic) here; return None to use the
        # autograd fallback while the analytic version is being ported.
        return None
