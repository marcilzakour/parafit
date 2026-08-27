"""ManoModel -- MANO hand under the parafit Model interface. [extra: mano]

Parameters ``theta = [pose(48) | trans(3)] = 51`` DOF; the shape ``betas`` (10)
is a fixed side input (predicted by a compact head upstream, not optimized in
the fit). The forward is manotorch's ``ManoLayer`` (``flat_hand_mean=True``)
followed by ``mano_to_openpose`` (16 regressed joints + 5 fingertips -> 21
OpenPose joints). The analytic landmark Jacobian is the cross-product
articulated Jacobian with the SO(3) right-Jacobian correction, ported from the
UA-Fit solver (``mvhpe/POEM-v2/lib/models/dovf/analytic_fitter.py::mano_kinematic_jac``);
it is validated against finite differences in the tests. MANO weights are not
shipped (license): pass the path to your ``mano_v1_2`` assets.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from parafit.core.lie import so3_right_jacobian
from parafit.core.types import ParamSpec, State
from parafit.models.base import Model
from parafit.registry import register_model

# Fingertip vertex ids (MANO KPID -> vertex), OpenPose reorder, and the distal
# MANO joint each fingertip rides on (thumb, index, middle, ring, pinky).
_TIP_VERTS = [744, 320, 443, 555, 672]
_OPENPOSE_PERM = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
_TIP_DISTAL = [15, 3, 6, 12, 9]


@register_model("mano", extra="mano")
def _build_mano(*args, **kwargs) -> "ManoModel":
    return ManoModel(*args, **kwargs)


def _ancestor_mask(parents, n: int = 16) -> Tensor:
    """A[k, j] = 1 if joint j is on the kinematic path root..k (inclusive)."""
    A = torch.zeros(n, n)
    for k in range(n):
        j = k
        while True:
            A[k, j] = 1.0
            p = int(parents[j])
            if j == 0 or p < 0 or p >= n:
                break
            j = p
        A[k, 0] = 1.0
    return A


def _mano_to_openpose(J_regressor: Tensor, verts: Tensor) -> Tensor:
    """MANO verts (B,778,3) -> 21 OpenPose joints (B,21,3)."""
    joints = torch.matmul(J_regressor, verts)                    # (B,16,3)
    tips = verts[:, _TIP_VERTS]                                  # (B,5,3)
    stacked = torch.cat([joints, tips], dim=1)                   # (B,21,3)
    return stacked[:, _OPENPOSE_PERM]


class ManoModel(Model):
    def __init__(self, mano_assets_root: str, betas: Optional[Tensor] = None,
                 num_joints: int = 21, center_idx: Optional[int] = 0, side: str = "right",
                 flat_hand_mean: bool = True):
        """``flat_hand_mean`` and ``center_idx`` are conventions of whatever
        produced the parameters, not preferences. Mismatching them is a silent
        ~100 mm error, so they are exposed rather than hardcoded.

        ``center_idx=0`` centres the output on the wrist before adding ``trans``;
        ``center_idx=None`` leaves the MANO template offset in, so the wrist sits
        ~9.7 cm from ``trans``. ``flat_hand_mean`` selects whether zero pose means
        a flat hand or the MANO mean pose.

        Measured settings per producer (all verified numerically, see
        ``tests/test_mano_conventions.py``):

        ========================  ================  ==========  ==============
        producer                  flat_hand_mean    center_idx  wrong-setting
        ========================  ================  ==========  ==============
        UA-Fit (parafit default)  True              0           --
        ARCTIC GT params          **False**         None        100 mm
        WiLoR / HaWoR             **True**          None        30.7 mm
        ========================  ================  ==========  ==============

        ARCTIC and the WiLoR/HaWoR family differ, which is not obvious and is
        worth stating plainly. ARCTIC stores axis-angle parameters consumed by an
        axis-angle MANO, which DOES add the mean pose. WiLoR and HaWoR predict
        rotation matrices consumed by ``smplx.MANOLayer``, which does NOT, because
        the mean pose is only ever applied in axis-angle space.

        **``smplx.MANOLayer.flat_hand_mean`` reports ``False`` while behaving as
        ``True``.** It even carries a nonzero ``pose_mean`` (sum ~11.7). Reading
        that flag and configuring downstream code from it is a silent 30.7 mm
        error. This also bites when consuming HaWoR's *stored* output, which is
        axis-angle (converted via ``rotation_matrix_to_angle_axis``) but must
        still be evaluated with ``flat_hand_mean=True`` to match the prediction.
        """
        from manotorch.manolayer import ManoLayer  # raises ImportError -> parafit[mano]

        self.mano_layer = ManoLayer(
            joint_rot_mode="axisang",
            use_pca=False,
            mano_assets_root=mano_assets_root,
            center_idx=center_idx,
            flat_hand_mean=flat_hand_mean,
            side=side,
        )
        self.J_regressor = self.mano_layer.th_J_regressor       # (16,778)
        self.num_joints = num_joints
        self.center_idx = center_idx
        self.flat_hand_mean = flat_hand_mean
        self.betas = betas                                       # (B,10) or None -> zeros
        self._anc = _ancestor_mask(self.mano_layer.kintree_parents)  # (16,16)
        self.param_spec = ParamSpec({"pose": (0, 48), "trans": (48, 3)})

    def _betas_for(self, B: int, ref: Tensor) -> Tensor:
        if self.betas is None:
            return torch.zeros(B, 10, dtype=ref.dtype, device=ref.device)
        b = self.betas.to(ref)
        return b.expand(B, 10) if b.dim() == 2 and b.shape[0] == 1 else b

    def forward(self, params: Tensor) -> State:
        B = params.shape[0]
        pose = params[:, :48]
        trans = params[:, 48:51]
        betas = self._betas_for(B, params)
        out = self.mano_layer(pose, betas)
        verts = out.verts                                        # (B,778,3), centered at center_idx
        jc = _mano_to_openpose(self.J_regressor.to(params), verts)[:, : self.num_joints]
        landmarks = jc + trans.unsqueeze(1)
        return State(landmarks=landmarks, verts=verts + trans.unsqueeze(1),
                     transforms=out.transforms_abs, batch_size=[B])

    def landmark_jacobian(self, params: Tensor, state: State) -> Optional[Tensor]:
        if state.transforms is None:
            return None  # autograd fallback
        B = params.shape[0]
        pose = params[:, :48]
        betas = self._betas_for(B, params)
        J_reg = self.J_regressor.to(params)
        T = state.transforms                                     # (B,16,4,4) uncentered
        Rg = T[:, :, :3, :3]                                     # (B,16,3,3)
        pg = T[:, :, :3, 3]                                      # (B,16,3)
        theta = pose.reshape(B, 16, 3)
        Jr = so3_right_jacobian(theta)                           # (B,16,3,3)
        axes = Rg @ Jr                                           # (B,16,3,3): col m = world axis for dtheta_{j,m}
        # true fingertip positions (skinned ~rigidly to the distal joint)
        shapedirs = self.mano_layer.th_shapedirs.to(params)      # (778,3,10)
        bS = torch.matmul(shapedirs, betas.transpose(0, 1)).permute(2, 0, 1)  # (B,778,3)
        v_rest = self.mano_layer.th_v_template.to(params) + bS   # (B,778,3)
        J_rest = torch.matmul(J_reg, v_rest)                     # (B,16,3)
        tips_rest = v_rest[:, _TIP_VERTS]                        # (B,5,3)
        d = _TIP_DISTAL
        tip_pos = pg[:, d] + torch.einsum("bdij,bdj->bdi", Rg[:, d], tips_rest - J_rest[:, d])  # (B,5,3)
        P = torch.cat([pg, tip_pos], dim=1)                      # (B,21,3) stack order
        A = torch.cat([self._anc.to(params), self._anc.to(params)[d]], dim=0)  # (21,16)
        S = P.shape[1]
        diff = P.unsqueeze(2) - pg.unsqueeze(1)                  # (B,S,16,3)
        a_e = axes.unsqueeze(1).expand(B, S, 16, 3, 3)
        d_e = diff.unsqueeze(-1).expand(B, S, 16, 3, 3)
        cr = torch.cross(a_e, d_e, dim=3) * A.view(1, S, 16, 1, 1)
        dP = cr.permute(0, 1, 3, 2, 4).reshape(B, S, 3, 48)      # (B,21,3,48) stack order
        dJc = dP[:, _OPENPOSE_PERM]                              # to openpose order
        if self.center_idx is not None:
            # Mirror the layer: it subtracts joint ``center_idx``, so the
            # derivative must subtract that joint's derivative too. Under
            # ``center_idx=None`` the layer does not centre, and subtracting here
            # would make the Jacobian inconsistent with the forward.
            dJc = dJc - dJc[:, self.center_idx: self.center_idx + 1]
        dJc = dJc[:, : self.num_joints]                          # (B,J,3,48)
        dtrans = torch.eye(3, dtype=params.dtype, device=params.device)
        dtrans = dtrans.view(1, 1, 3, 3).expand(B, self.num_joints, 3, 3)
        return torch.cat([dJc, dtrans], dim=-1)                 # (B,J,3,51)
