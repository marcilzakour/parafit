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


def _hat(v: Tensor) -> Tensor:
    z = torch.zeros(v.shape[:-1], dtype=v.dtype, device=v.device)
    return torch.stack([torch.stack([z, -v[..., 2], v[..., 1]], -1),
                        torch.stack([v[..., 2], z, -v[..., 0]], -1),
                        torch.stack([-v[..., 1], v[..., 0], z], -1)], -2)


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
                 num_joints: int = 21, center_idx: int = 0, side: str = "right"):
        from manotorch.manolayer import ManoLayer  # raises ImportError -> parafit[mano]

        self.mano_layer = ManoLayer(
            joint_rot_mode="axisang",
            use_pca=False,
            mano_assets_root=mano_assets_root,
            center_idx=center_idx,
            flat_hand_mean=True,
            side=side,
        )
        self.J_regressor = self.mano_layer.th_J_regressor       # (16,778)
        self.num_joints = num_joints
        self.center_idx = center_idx
        self.betas = betas                                       # (B,10) or None -> zeros
        self._anc = _ancestor_mask(self.mano_layer.kintree_parents)  # (16,16)
        self.tip_corrective = False   # True: EXACT analytic landmark Jacobian (pose correctives + skinning blend; joints are regressed from the posed vertices, so every landmark carries these terms)
        self._exact_idx = None
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

    def _vertex_jacobian(self, params: Tensor, state: State, verts_idx) -> Tensor:
        """(B, V, 3, 48) EXACT analytic Jacobian of selected MANO vertices: pose blend shapes + the full skinning blend.
            v = sum_k w_k [ p_k + R_k (v_rest + Pc(theta) - J_k) ],   Pc = posedirs @ vec(R_1..15 - I)
            dv/dtheta_j = a_j x (sum_k w_k anc_kj q_k - (sum_k w_k anc_kj) p_j) + (sum_k w_k R_k) dPc/dtheta_j
        a_j are the world DOF axes (as in the rigid rows), q_k the k-th bone's contribution, and
        d vec(R_j)/d delta_m = R_j [ (J_r(theta_j))_{:,m} ]_x for the right-perturbation convention the solver uses."""
        from parafit.core.lie import so3_exp
        B = params.shape[0]; ml = self.mano_layer; dtp = params.dtype; dev = params.device
        T = state.transforms; Rg = T[:, :, :3, :3]; pg = T[:, :, :3, 3]
        theta = params[:, :48].reshape(B, 16, 3); Jr = so3_right_jacobian(theta); axes = Rg @ Jr
        betas = self._betas_for(B, params)
        shapedirs = ml.th_shapedirs.to(params); v_shaped = ml.th_v_template.to(params) + torch.einsum("vdk,bk->bvd", shapedirs, betas)
        J_rest = torch.matmul(self.J_regressor.to(params), v_shaped)                                      # (B,16,3)
        w = ml.th_weights.to(params)[verts_idx]                                                           # (V,16)
        posed = ml.th_posedirs.to(params)[verts_idx]                                                      # (V,3,135)
        R_loc = so3_exp(theta.reshape(-1, 3)).reshape(B, 16, 3, 3)
        I3 = torch.eye(3, dtype=dtp, device=dev)
        feat = (R_loc[:, 1:] - I3).reshape(B, 135)
        v_posed = v_shaped[:, verts_idx] + torch.einsum("vdk,bk->bvd", posed, feat)                       # (B,V,3)
        q = pg[:, None] + torch.einsum("bkij,bvkj->bvki", Rg, v_posed[:, :, None, :] - J_rest[:, None, :, :])   # (B,V,16,3)
        anc = self._anc.to(params)
        m = torch.einsum("vk,kj->vj", w, anc)                                                             # (V,16)
        sq = torch.einsum("vk,kj,bvki->bvji", w, anc, q)                                                  # (B,V,16,3)
        d = sq - m[None, :, :, None] * pg[:, None]                                                        # (B,V,16,3)
        V = len(verts_idx)
        cr = torch.cross(axes[:, None].expand(B, V, 16, 3, 3), d[..., None].expand(B, V, 16, 3, 3), dim=3)
        dP = cr.permute(0, 1, 3, 2, 4).reshape(B, V, 3, 48)
        # corrective term (vectorised over the 15 non-root joints)
        hatJ = _hat(Jr[:, 1:].permute(0, 1, 3, 2).reshape(B, 15 * 3, 3)).reshape(B, 15, 3, 3, 3)          # [.,j,m] = hat of the m-th axis
        dR = torch.einsum("bjik,bjmkl->bjmil", R_loc[:, 1:], hatJ)                                        # (B,15,3,3,3): d R_j / d delta_{j,m}
        dfeat = params.new_zeros(B, 15, 9, 16, 3)
        val = dR.reshape(B, 15, 3, 9).permute(0, 1, 3, 2)                                        # (B,15,9,3): d feat-block(j-1) / d theta_j
        for j in range(15): dfeat[:, j, :, j + 1, :] = val[:, j]                                 # block j-1 of feat depends only on theta_{j+1}
        dfeat = dfeat.reshape(B, 135, 48)
        dPc = torch.einsum("vdk,bkp->bvdp", posed, dfeat)
        Rw = torch.einsum("vk,bkij->bvij", w, Rg)
        return dP + torch.einsum("bvij,bvjp->bvip", Rw, dPc)

    def _tip_jacobian(self, params: Tensor, state: State) -> Tensor:
        """Exact (B,5,3,48) Jacobian of the fingertip VERTICES: they carry the pose blend shapes and are skinned by a blend
        of joint transforms, which the rigid-ride rows ignore (the 16 joint centres are exact there). Closed form:
            v = sum_k w_k [ p_k + R_k (v_rest + Pc(theta) - J_k) ],  Pc = posedirs @ vec(R_1..15 - I)
            dv/dtheta_j = a_j x (sum_k w_k anc_kj q_k - (sum_k w_k anc_kj) p_j) + (sum_k w_k R_k) dPc/dtheta_j
        with q_k the k-th bone's contribution and a_j the world DOF axes (the same `axes` the rigid rows use)."""
        from parafit.core.lie import so3_exp
        B = params.shape[0]; ml = self.mano_layer; dtp = params.dtype
        T = state.transforms; Rg = T[:, :, :3, :3]; pg = T[:, :, :3, 3]
        theta = params[:, :48].reshape(B, 16, 3); Jr = so3_right_jacobian(theta); axes = Rg @ Jr        # (B,16,3,3) col m = world axis of DOF (j,m)
        betas = self._betas_for(B, params)
        shapedirs = ml.th_shapedirs.to(params); v_shaped = ml.th_v_template.to(params) + torch.einsum("vdk,bk->bvd", shapedirs, betas)
        J_rest = torch.matmul(self.J_regressor.to(params), v_shaped)                                     # (B,16,3)
        tips = _TIP_VERTS; w = ml.th_weights.to(params)[tips]                                            # (5,16)
        posed = ml.th_posedirs.to(params)[tips]                                                          # (5,3,135)
        R_loc = so3_exp(theta.reshape(-1, 3)).reshape(B, 16, 3, 3)
        I3 = torch.eye(3, dtype=dtp, device=params.device)
        feat = (R_loc[:, 1:] - I3).reshape(B, 135)
        v_posed = v_shaped[:, tips] + torch.einsum("vdk,bk->bvd", posed, feat)                           # (B,5,3)
        q = pg[:, None] + torch.einsum("bkij,bvkj->bvki", Rg, v_posed[:, :, None, :] - J_rest[:, None, :, :])   # (B,5,16,3) bone contributions
        anc = self._anc.to(params)                                                                        # (16,16): anc[k,j] = 1 if j is j<=k on the chain
        m = torch.einsum("vk,kj->vj", w, anc)                                                             # (5,16)
        sq = torch.einsum("vk,kj,bvki->bvji", w, anc, q)                                                  # (5,16) weighted points per joint
        d = sq - m[None, :, :, None].permute(0, 1, 2, 3) * pg[:, None]                                    # (B,5,16,3)
        out = torch.cross(axes[:, None, :, :, :].expand(B, 5, 16, 3, 3), d[..., None].expand(B, 5, 16, 3, 3), dim=3)   # (B,5,16,3,3dof)
        dP = out.permute(0, 1, 3, 2, 4).reshape(B, 5, 3, 48)
        # corrective term: dPc/dtheta_{j,m} = posed @ d feat/d theta_{j,m};  d vec(R_j)/d delta_m = R_j [ (Jr_j)_{:,m} ]_x
        Rw = torch.einsum("vk,bkij->bvij", w, Rg)                                                         # (B,5,3,3)
        dfeat = params.new_zeros(B, 135, 48)
        for j in range(1, 16):
            for mdof in range(3):
                gen = _hat(Jr[:, j, :, mdof])                                                              # (B,3,3)
                dR = torch.matmul(R_loc[:, j], gen)                                                        # (B,3,3)
                dfeat[:, (j - 1) * 9: j * 9, j * 3 + mdof] = dR.reshape(B, 9)
        dPc = torch.einsum("vdk,bkp->bvdp", posed, dfeat)                                                  # (B,5,3,48)
        dP = dP + torch.einsum("bvij,bvjp->bvip", Rw, dPc)
        return dP

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
        if self.tip_corrective:
            if self._exact_idx is None:                                              # support of the joint regressor + the 5 tips
                nz = (self.J_regressor.abs() > 1e-8).any(0).nonzero().flatten().tolist()
                self._exact_idx = sorted(set(nz) | set(_TIP_VERTS))
                self._exact_pos = {v: i for i, v in enumerate(self._exact_idx)}
                self._exact_reg = self.J_regressor[:, self._exact_idx].clone()        # (16,V)
                self._exact_tip = [self._exact_pos[v] for v in _TIP_VERTS]
            dV = self._vertex_jacobian(params, state, self._exact_idx)                # (B,V,3,48)
            dJ16 = torch.einsum("jv,bvdp->bjdp", self._exact_reg.to(params), dV)      # joints are regressed from the POSED vertices
            dP = torch.cat([dJ16, dV[:, self._exact_tip]], dim=1)
        dJc = dP[:, _OPENPOSE_PERM]                              # to openpose order
        dJc = dJc - dJc[:, self.center_idx: self.center_idx + 1]  # center like the layer
        dJc = dJc[:, : self.num_joints]                          # (B,J,3,48)
        dtrans = torch.eye(3, dtype=params.dtype, device=params.device)
        dtrans = dtrans.view(1, 1, 3, 3).expand(B, self.num_joints, 3, 3)
        return torch.cat([dJc, dtrans], dim=-1)                 # (B,J,3,51)
