"""UmeTrackModel -- UmeTrack parametric hand under the Model interface. [extra: umetrack]

PORT TARGET (thesis-hope):
  show3d_challenge/scripts/umetrack_torch.py:46  : UmeTrackLayer (torch FK+LBS)
    forward(angles (B,22), wristR (B,3,3), wristT (B,3)) -> verts (B,788,3), landmarks (B,21,3)
  show3d_challenge/scripts/umetrack_hand.py:51   : load_hand_model (profile_umetrack.json,
                                                    joint_limits (22,2), hand_scale)

Parameter layout: theta = [flexion(20) | wrist_rot(3 axis-angle) | wrist_trans(3)]
= 26 DOF (only the first 20 finger DOFs of the 22-vector are used). The joint
limits are enforced by a reparameterization (angle = lo + (hi-lo)*sigmoid(theta))
via ``to_raw``/``from_raw`` -- so every solution is anatomically valid by
construction, matching train_umetrack_ik.py. Today UmeTrack fits with Adam; under
parafit it drops into the same LMSolver (autograd Jacobian first, analytic later).
"""
from __future__ import annotations

from parafit.core.types import ParamSpec, State
from parafit.models.base import Model
from parafit.registry import register_model


@register_model("umetrack", extra="umetrack")
def _build_umetrack(*args, **kwargs) -> "UmeTrackModel":
    return UmeTrackModel(*args, **kwargs)


class UmeTrackModel(Model):
    def __init__(self, subject: str, side: int = 0, device: str = "cuda"):
        # Imports the show3d UmeTrack layer; packaged under extra [umetrack].
        from parafit._vendor.umetrack_torch import UmeTrackLayer  # noqa: F401  (to be vendored)

        self._layer_cls = UmeTrackLayer
        self.subject = subject
        self.side = side
        self.device = device
        # 20 flexion + 6 wrist (axis-angle + translation).
        self.param_spec = ParamSpec({"flexion": (0, 20), "wrist_rot": (20, 3), "wrist_trans": (23, 3)})

    def forward(self, params) -> State:  # pragma: no cover - port stub
        raise NotImplementedError(
            "Port UmeTrack forward: map params -> (angles, wristR, wristT) with the "
            "joint-limit sigmoid reparam, call UmeTrackLayer.forward. "
            "See show3d_challenge/scripts/umetrack_torch.py."
        )
