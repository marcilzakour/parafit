"""One MHR hand as a parafit ``Model`` for hand-only fitting (ego window solver).

The MHR rig is a whole body; here the body is held at rest and one hand is cut free: its wrist gets a free
6-DoF root (axis-angle + translation, metres) and its fingers keep the rig's own pose parameters (27 per
hand: the ``<side>_{thumb,index,middle,ring,pinky}*`` columns of the parameter transform, which encode the
rig's coupled finger DOFs). Identity / scale stay at the rig mean (like the mean-shape MANO in the window).

theta layout (n_params = 33): [root axis-angle (3) | fingers (27) | root translation (3)]  (rotation block
first and translation last, so the MANO-style helpers (Kabsch init, translation-only warm start) carry over).
Landmarks: 21 joints in OpenPose order (wrist, thumb x4, index x4, middle x4, ring x4, pinky x4; tips = the
rig's ``*_null`` end joints); ``State.verts`` = the hand's mesh vertices (LBS, correctives optional) when
``full_mesh=True``. Units: MHR is in cm, this model returns metres.
Jacobian: rigid-ride chain Jacobian of the underlying model on the hand joints (exact for joint landmarks),
rotated by the root, plus the closed-form root rows.
"""
from __future__ import annotations
from typing import Optional
import torch
from torch import Tensor
from parafit.core.lie import so3_exp, point_rotation_jacobian
from parafit.core.types import ParamSpec, State
from parafit.models.base import Model
from parafit.models.mhr import MhrModel, N_MODEL, N_IDENTITY

_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
# OpenPose 21 <- MHR joint names (per side prefix); thumb0 is the carpal (CMC-at-wrist) joint, thumb1..3 + null the chain
_OP_NAMES = ["wrist", "thumb1", "thumb2", "thumb3", "thumb_null", "index1", "index2", "index3", "index_null", "middle1", "middle2", "middle3", "middle_null",
             "ring1", "ring2", "ring3", "ring_null", "pinky1", "pinky2", "pinky3", "pinky_null"]
_OP_NAMES_ALT = ["wrist", "thumb0", "thumb1", "thumb2", "thumb_null"] + _OP_NAMES[5:]   # alternative thumb mapping (probe picks by MANO rest-pose fit)


class MhrHandModel(Model):
    def __init__(self, mhr: MhrModel, side: str = "right", full_mesh: bool = False, thumb_alt: bool = False):
        self.mhr, self.side, self.full_mesh = mhr, side, full_mesh
        pre = "l_" if side == "left" else "r_"
        names = mhr.joint_names; pn = mhr.param_names
        op = _OP_NAMES_ALT if thumb_alt else _OP_NAMES
        self.joint_idx = torch.tensor([names.index(pre + n) for n in op], dtype=torch.long)
        self.wrist_idx = names.index(pre + "wrist")
        self.finger_cols = torch.tensor([i for i, n in enumerate(pn) if n.startswith(pre) and any(t in n for t in _FINGERS)], dtype=torch.long)
        self.n_fingers = int(self.finger_cols.numel())
        self.param_spec = ParamSpec({"root_rot": (0, 3), "fingers": (3, self.n_fingers), "root_trans": (3 + self.n_fingers, 3)})
        self.num_joints = 21; self.center_idx = 0
        dev, dt = mhr.device, mhr.dtype
        # rest body: wrist position (cm) and, for the mesh, the hand's vertex subset
        with torch.no_grad():
            st0 = mhr.forward(torch.zeros(1, N_MODEL + N_IDENTITY, dtype=dt, device=dev), full_mesh=full_mesh)
        self.t_w0 = st0.landmarks[0, self.wrist_idx].clone()                                    # (3,) cm
        if full_mesh:
            left = torch.tensor([n.startswith("l_") for n in names]); dom = mhr.dominant
            vm = mhr.hand_vertex_mask & (left[dom] if side == "left" else ~left[dom])
            self.vert_idx = vm.nonzero().flatten()
            remap = torch.full((mhr.n_verts,), -1, dtype=torch.long); remap[self.vert_idx] = torch.arange(self.vert_idx.numel())
            f = remap[mhr.faces]; self.faces = f[(f >= 0).all(1)]
        else:
            self.vert_idx = None; self.faces = None
        self._dev, self._dt = dev, dt

    def _full_theta(self, params: Tensor) -> Tensor:
        B = params.shape[0]; th = torch.zeros(B, N_MODEL + N_IDENTITY, dtype=params.dtype, device=params.device)
        th[:, self.finger_cols.to(params.device)] = params[:, 3:3 + self.n_fingers]
        return th

    def forward(self, params: Tensor) -> State:
        B = params.shape[0]; w = params[:, :3]; t = params[:, -3:]
        st = self.mhr.forward(self._full_theta(params), full_mesh=self.full_mesh)
        R = so3_exp(w)                                                                              # (B,3,3)
        xl = (st.landmarks[:, self.joint_idx.to(params.device)] - self.t_w0.to(params)) / 100.0   # (B,21,3) m, wrist-local (rest body)
        lm = torch.einsum("bij,bnj->bni", R, xl) + t.unsqueeze(1)
        verts = None
        if self.full_mesh and st.verts is not None:
            vl = (st.verts[:, self.vert_idx.to(params.device)] - self.t_w0.to(params)) / 100.0
            verts = torch.einsum("bij,bnj->bni", R, vl) + t.unsqueeze(1)
        return State(landmarks=lm, verts=verts, batch_size=[B])

    def landmark_jacobian(self, params: Tensor, state: State) -> Optional[Tensor]:
        B = params.shape[0]; P = self.n_params; dev = params.device
        th = self._full_theta(params); st = self.mhr.forward(th, full_mesh=False)                  # landmarks-only re-forward (cheap; State is immutable)
        Jf = self.mhr.landmark_jacobian(th, st)                                                     # (B,L,3,249) cm per unit
        if Jf is None: return None
        Jl = Jf[:, self.joint_idx.to(dev)][:, :, :, self.finger_cols.to(dev)] / 100.0                # (B,21,3,27) wrist-local (wrist is fixed under finger params)
        xl = (st.landmarks[:, self.joint_idx.to(dev)] - self.t_w0.to(params)) / 100.0; R = so3_exp(params[:, :3])
        J = torch.zeros(B, 21, 3, P, dtype=params.dtype, device=dev)
        J[:, :, :, 3:3 + self.n_fingers] = torch.einsum("bij,bnjp->bnip", R, Jl)
        J[:, :, :, :3] = point_rotation_jacobian(params[:, :3], xl)                                # d(R(w) x)/dw, (B,21,3,3)
        J[:, :, :, -3:] = torch.eye(3, dtype=params.dtype, device=dev).view(1, 1, 3, 3).expand(B, 21, 3, 3)
        return J
