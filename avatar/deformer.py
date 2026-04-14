from typing import Tuple

import torch
from torch import nn, Tensor

from flame import blend_shapes, flame_fn
from avatar.gaussian_model import GaussianModel
from utils.math import build_rotation, quat_product_wxyz, rotmat_to_unitquat_wxyz
from utils.geometry import compute_vertex_normals, dot, length, safe_normalize

class GaussianDeformer(nn.Module):
    """ Rig gaussians to the FLAME mesh. """

    def __init__(self, args, device, flame, shape_param: Tensor):
        super().__init__()
        self.args = args
        self.device = device
        self.flame = flame
        self.flame_joint_center = torch.einsum("bik,ij->bjk", [blend_shapes(shape_param, flame.shapedirs_identity) + flame.v_template, flame.J_regressor.T])
        self.displacements_scale = 1.0

    def capture(self):
        state = self.state_dict()
        state = {k: v for k,v in state.items() if not k.startswith("flame.")}
        return state
  
    def restore(self, state):
        self.load_state_dict(state, strict=False)
    
    def get_mesh_verts(self, pose: Tensor, expression: Tensor, shape: Tensor, gaussians: GaussianModel, flame_scale: float, use_original_flame=False):
        flame, g = self.flame, gaussians       
        full_pose_param, eyelid_param, translation_param = pose[:, 0:15], pose[:, 15:17], pose[:, 17:20]

        # Deform mesh
        if use_original_flame:
            verts, _ = flame_fn(
                flame.v_template,
                shape = shape,
                expression = expression,
                full_pose = full_pose_param,
                eyelids = eyelid_param,
                translation = translation_param,
                lbs_weights = flame.lbs_weights,
                shapedirs_expr = flame.shapedirs_expression,
                posedirs = flame.posedirs,
                shapedirs_id = flame.shapedirs_identity,
                l_eyelid_dirs = flame.l_eyelid,
                r_eyelid_dirs = flame.r_eyelid,
                joint_centers = self.flame_joint_center,
                rot_parents = flame.parents,
            )
        else:
            lbs_weights_exp = torch.relu(g.lbs_weights)
            lbs_weights = lbs_weights_exp / (lbs_weights_exp.sum(dim=-1, keepdim=True) + 1e-5)

            verts, _ = flame_fn(
                g.flame_v_template,
                shape = shape,
                expression = expression,
                full_pose = full_pose_param,
                eyelids = eyelid_param,
                translation = translation_param,
                lbs_weights = lbs_weights,
                shapedirs_expr = g.expression_dirs,
                posedirs = g.pose_dirs,
                shapedirs_id = g.shape_dirs,
                l_eyelid_dirs = g.l_eyelid_dirs,
                r_eyelid_dirs = g.r_eyelid_dirs,
                joint_centers = self.flame_joint_center,
                rot_parents = flame.parents,
            )

        return verts * flame_scale # (B, V, 3)

    def forward(self, pose: Tensor, expression: Tensor, shape: Tensor, gaussians: GaussianModel, flame_scale: float,
                get_mesh_verts=False, use_original_flame=False):
        """
        Return:
        - xyz (B, n, 3)
        - rotations (B, n, 4)
        - scales (B, n, 2)
        - opacities (B, n)
        """

        g = gaussians
        g_tri, g_bary_coords = g.get_binding()
        B = pose.shape[0]
        faces = self.flame.faces
        deform_scales = "none"
        
        xyz_mesh = self.get_mesh_verts(pose, expression, shape, gaussians, flame_scale, use_original_flame) # (B, V, 3)

        # Interpolate mesh positions and normals to gaussians
        g_vidx = faces[g_tri] # (n, 3)

        # (n,3,3), (n,3) ==> (n,3)
        xyz_gaussians = torch.einsum("bvij,vi->bvj", [xyz_mesh[:,g_vidx], g_bary_coords]) # (B, n, 3)

        require_face_scale = (deform_scales == "isotropic")
        face_orien_mat, face_scale = compute_face_orientation(xyz_mesh, faces, require_face_scale) # tuple (B,F,3,3), (B,F,1)
        face_normals = face_orien_mat[:,:,:,2] # (B, F, 3)

        # Displace Gaussians along the triangle normal directions
        if False:
            # Face normals
            nrm = face_normals[:, g_tri] # (B, n, 3)
        else:
            # Vertex normals
            normals_mesh = compute_vertex_normals(xyz_mesh, faces, face_normals) # (B, V, 3)
            nrm = torch.einsum("bvij,vi->bvj", [normals_mesh[:, g_vidx], g_bary_coords]) # (B, n, 3)
        xyz_gaussians = xyz_gaussians + gaussians._disp * nrm * self.displacements_scale

        opacity = g.opacity_activation(g._opacity)
        opacity = opacity.unsqueeze(0).repeat(B, 1, 1) # expand opacity to the batch (no frame-dependent changes)

        # Face orientation quaternion per face
        face_orien_quat = rotmat_to_unitquat_wxyz(face_orien_mat) # (B,F,4)
        # Face orientation quaternion per gaussian
        tri_quat = face_orien_quat[:, g_tri] # (B,n,4)

        rot = quat_product_wxyz(
            g.rotation_activation(g._rotation_base).unsqueeze(0),
            tri_quat # already a unit quaternion
        )

        if deform_scales == "isotropic":
            # We don't multiply by flame_scale here because xyz_mesh accounts for it already 
            scaling = g.scaling_activation(g._scaling_base).unsqueeze(0) * face_scale[:, g_tri] * 40
        elif deform_scales == "anisotropic":
            # We don't multiply by flame_scale here because xyz_mesh accounts for it already 
            scaling = torch.stack([
                g.scaling_activation(g._scaling_base) * compute_anisotropic_scales(xyz_mesh[i], faces, g_tri, rot[i]) * 100
                for i in range(B)
            ])
        elif deform_scales == "none":            
            scaling = g.scaling_activation(g._scaling_base) * flame_scale
            scaling = scaling.unsqueeze(0).repeat(B, 1, 1)
        else:
            raise NotImplementedError(f"Unknown deform_scales '{deform_scales}'")

        out = xyz_gaussians, rot, scaling, opacity
        if get_mesh_verts:
            out = *out, xyz_mesh
        return out

def compute_anisotropic_scales(verts: Tensor, faces: Tensor, g_tri: Tensor, g_rot: Tensor) -> Tensor:
    assert verts.ndim == 2 and verts.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3 and faces.dtype == torch.long
    assert g_tri.ndim == 1 and g_tri.dtype == torch.long
    assert g_rot.ndim == 2 and g_rot.shape[1] == 4

    v0, v1, v2 = verts[faces].unbind(1)
    e0 = v1 - v0 # (F, 3)
    e1 = v2 - v0 # (F, 3)

    # Compute unit tangential vectors of splats
    L = build_rotation(g_rot) # (n, 3, 3)
    s, t, _ = L.unbind(2) 

    # Project tangential vectors of each gaussian onto two edges of its triangle
    e0, e1 = e0[g_tri], e1[g_tri] # (n, 3)
    e0s, e0t = dot(s, e0), dot(t, e0)
    e1s, e1t = dot(s, e1), dot(t, e1)

    return torch.cat([e0s + e1s, e0t + e1t, torch.zeros_like(e0s)], dim=-1) # (n, 3)

def compute_face_orientation(verts: Tensor, faces: Tensor, include_face_scale: bool) -> Tuple[Tensor, Tensor]:
    assert verts.ndim == 3 and verts.shape[2] == 3 # (B,V,3)
    assert faces.ndim == 2 and faces.shape[1] == 3 and faces.dtype == torch.long # (F,3)

    v0, v1, v2 = verts[:,faces].unbind(2) # (B,F,3) each

    T = safe_normalize(v1 - v0)
    N = safe_normalize(torch.cross(T, v2 - v0, dim=-1))
    B = safe_normalize(torch.cross(N, T, dim=-1))

    orientation = torch.stack([T, B, N], dim=-1)

    if include_face_scale:
        s0 = length(v1 - v0)
        s1 = dot(B, (v2 - v0)).abs()
        scale = (s0 + s1) / 2
        return orientation, scale
    else:
        return orientation, None
