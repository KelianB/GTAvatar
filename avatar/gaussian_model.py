#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
from simple_knn._C import distCUDA2
import logging

from flame import sample_flame
from utils.geometry import calculate_mesh_avg_edge_length
from utils.p3d_rasterizer import Pytorch3dRasterizer, vertices_to_face
from utils.math import inverse_sigmoid


class GaussianModel:

    def setup_functions(self, args):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, args, flame):
        sh_degree = 0
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling_base = torch.empty(0)
        self._rotation_base = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions(args)
        self.args = args
        self.flame = flame
    
    def capture(self):
        return {
            "active_sh_degree": self.active_sh_degree,
            "features_dc": self._features_dc,
            "features_rest": self._features_rest,
            "scaling_base": self._scaling_base,
            "rotation_base": self._rotation_base,
            "opacity": self._opacity,
            "max_radii2D": self.max_radii2D,
            "xyz_gradient_accum": self.xyz_gradient_accum,
            "denom": self.denom,
            "spatial_lr_scale": self.spatial_lr_scale,
            "flame_v_template": getattr(self, "flame_v_template", None),
            "shape_dirs": getattr(self, "shape_dirs", None),
            "r_eyelid_dirs": getattr(self, "r_eyelid_dirs", None),
            "l_eyelid_dirs": getattr(self, "l_eyelid_dirs", None),
            "expression_dirs": getattr(self, "expression_dirs", None),
            "pose_dirs": getattr(self, "pose_dirs", None),
            "lbs_weights": getattr(self, "lbs_weights", None),
            "disp": getattr(self, "_disp", None),
            "J_uv_st": getattr(self, "J_uv_st", None),
            "triangle_idx": getattr(self, "triangle_idx", None),
            "bary_coords": getattr(self, "bary_coords", None)
        }
    
    def restore(self, state):
        self.active_sh_degree = state["active_sh_degree"]
        self._features_dc = state["features_dc"]
        self._features_rest = state["features_rest"]
        self._scaling_base = state["scaling_base"]
        self._rotation_base = state["rotation_base"]
        self._opacity = state["opacity"]
        self.max_radii2D = state["max_radii2D"]
        self.xyz_gradient_accum = state["xyz_gradient_accum"]
        self.denom = state["denom"]
        self.spatial_lr_scale = state["spatial_lr_scale"]
        if "flame_v_template" in state: self.flame_v_template = state["flame_v_template"]
        if "shape_dirs" in state: self.shape_dirs = state["shape_dirs"]
        if "r_eyelid_dirs" in state: self.r_eyelid_dirs = state["r_eyelid_dirs"]
        if "l_eyelid_dirs" in state: self.l_eyelid_dirs = state["l_eyelid_dirs"]
        if "expression_dirs" in state: self.expression_dirs = state["expression_dirs"]
        if "pose_dirs" in state: self.pose_dirs = state["pose_dirs"]
        if "lbs_weights" in state: self.lbs_weights = state["lbs_weights"]
        if "disp" in state: self._disp = state["disp"]
        if "J_uv_st" in state: self.J_uv_st = state["J_uv_st"]
        if "triangle_idx" in state: self.triangle_idx = state["triangle_idx"]
        if "bary_coords" in state: self.bary_coords = state["bary_coords"]
        logging.info(f"GaussianModel restored with {self._opacity.shape[0]} gaussians.")

    @property
    def n_gaussians(self):
        return self._opacity.shape[0]

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    '''
    def create_from_verts(self, points, opacity_coeff=0.1, scale_coeff=1, autoscale=True, scales=None):
        device = points.device
        features = torch.zeros((points.shape[0], 3, (self.max_sh_degree + 1) ** 2), dtype=torch.float, device=device)

        if scales is None:
            if autoscale:
                dist2 = torch.clamp_min(distCUDA2(points), 0.0000001)
            else:
                dist2 = torch.ones_like(points[..., 0])
            scales = self.scaling_inverse_activation(torch.sqrt(dist2) * scale_coeff)[...,None].repeat(1, 3)
        else:
            assert scales.shape == (points.shape[0], 3)
            scales = self.scaling_inverse_activation(scales)

        rots = torch.zeros((points.shape[0], 4), device=device)
        rots[:, 0] = 1

        assert 0 < opacity_coeff < 1 # a value of 0 or 1 is impossible with a sigmoid activation
        opacities = self.inverse_opacity_activation(opacity_coeff * torch.ones((points.shape[0], 1), dtype=torch.float, device=device))

        self._xyz = nn.Parameter(points.requires_grad_(False))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling_base = nn.Parameter(scales.requires_grad_(True))
        self._rotation_base = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((points.shape[0]), device=device)
    '''

    def create_uvs_triangle(self, flame, flame_scale: float, init_opacity: float, init_scale_factor: float, gaussians_init_count: float):
        self.cameras_extent = calculate_mesh_avg_edge_length(flame.v_template, flame.faces) * flame_scale
        self.spatial_lr_scale = self.cameras_extent

        xyz, uvs, bary_coords, tri_idx = sample_flame(flame, samples_per_face=gaussians_init_count)
        xyz = xyz.float()
        device = xyz.device
        n = xyz.shape[0]

        # Compute a mask of used UV areas
        verts_uv, faces_uv = flame.verts_uvs.unsqueeze(0), flame.textures_idx.unsqueeze(0)
        S = 1024
        uv_rasterizer = Pytorch3dRasterizer()
        verts_uv = verts_uv * 2 - 1
        verts_uv[..., 1] = -verts_uv[..., 1]
        verts_uv = torch.cat((verts_uv, torch.ones_like(verts_uv[..., 0:1])), -1).to(device)
        rast = uv_rasterizer(verts_uv, faces_uv, S, S, attributes=vertices_to_face(verts_uv, faces_uv), advanced=True)[0]
        self.uvmap_mask = rast[..., -1].view(S, S) > 0

        dist2 = torch.clamp_min((distCUDA2(xyz).float().to(device)), 0.0000001)
        scales = self.scaling_inverse_activation(torch.sqrt(dist2) * init_scale_factor)[...,None].repeat(1, 3)
        scales[:, :] -= 1.0

        rots = torch.zeros((n, 4), device=device)
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(init_opacity * torch.ones((n, 1), dtype=torch.float, device=device))
        self._disp = torch.zeros((n, 1), dtype=torch.float, device=device)
        self.triangle_idx = tri_idx.to(device)
        self.bary_coords = bary_coords[..., :2].to(device) # Use only two barycentric coordinates, the third is redundant
        self.J_uv_st = None
        self._scaling_base = scales
        self._rotation_base = rots
        self._opacity = opacities
        self.max_radii2D = torch.zeros((n), device=device)

        flame_uvs = flame.verts_uvs
        # remap mesh uvs to [-1,1] and invert v
        flame_uvs = flame_uvs * 2 - 1
        flame_uvs[..., 1] = -flame_uvs[..., 1]

        self.flame_uvs = flame_uvs
        self.flame_v_template = flame.v_template.clone()
        self.shape_dirs = flame.shapedirs_identity.clone()
        self.expression_dirs = flame.shapedirs_expression.clone()
        self.pose_dirs = flame.posedirs.clone()
        self.lbs_weights = flame.lbs_weights.clone()
        self.r_eyelid_dirs = flame.r_eyelid.clone()
        self.l_eyelid_dirs = flame.l_eyelid.clone()
        torch.cuda.empty_cache()

    def get_binding(self):
        a, b = self.bary_coords.unbind(-1)
        return self.triangle_idx, torch.stack([a, b, 1-a-b], dim=-1)

    def get_uvs(self):
        triangle_idx, bary_coords = self.get_binding()
        mesh_uvs, mesh_uvfaces = self.flame_uvs, self.flame.textures_idx
        g_vidx = mesh_uvfaces[triangle_idx] # (n, 3)
        g_uv = torch.einsum("vij,vi->vj", [mesh_uvs[g_vidx], bary_coords]) # (n, 2)
        return g_uv

    def training_setup(self, training_args):
        device = self._opacity.device
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.n_gaussians, 1), device=device)
        self.denom = torch.zeros((self.n_gaussians, 1), device=device)

        # Base Gaussian parameters
        self._opacity = torch.nn.Parameter(self._opacity)
        self._scaling_base = torch.nn.Parameter(self._scaling_base) 
        self._rotation_base = torch.nn.Parameter(self._rotation_base)
        self._disp = torch.nn.Parameter(self._disp)
        self.bary_coords = torch.nn.Parameter(self.bary_coords)
        # FLAME bases
        self.shape_dirs = torch.nn.Parameter(self.shape_dirs)
        self.r_eyelid_dirs = torch.nn.Parameter(self.r_eyelid_dirs)
        self.l_eyelid_dirs = torch.nn.Parameter(self.l_eyelid_dirs)
        self.expression_dirs = torch.nn.Parameter(self.expression_dirs)
        self.pose_dirs = torch.nn.Parameter(self.pose_dirs)
        self.lbs_weights = torch.nn.Parameter(self.lbs_weights) 
        self.flame_v_template = torch.nn.Parameter(self.flame_v_template) 

        self.optimizer = torch.optim.Adam([
            {'params': [self._opacity], 'lr': training_args.learn_opacity_lr, "name": "opacity"},
            {'params': [self._scaling_base], 'lr': training_args.learn_scaling_lr, "name": "scaling"},
            {'params': [self._rotation_base], 'lr': training_args.learn_rotation_lr, "name": "rotation"},
            {"params": [self._disp],"lr": training_args.learn_disp_lr * self.spatial_lr_scale, "name": "disp"},
            {"params": [self.bary_coords],"lr": training_args.learn_bary_lr, "name": "bary"},
            {"params": [self.shape_dirs],"lr": training_args.learn_flame_shape_dirs_lr, "name": "flame_shape_dirs"},
            {"params": [self.r_eyelid_dirs],"lr": training_args.learn_flame_expression_dirs_lr, "name": "gaussian_r_eyelid_dirs"},
            {"params": [self.l_eyelid_dirs],"lr": training_args.learn_flame_expression_dirs_lr, "name": "gaussian_l_eyelid_dirs"},
            {"params": [self.expression_dirs],"lr": training_args.learn_flame_expression_dirs_lr, "name": "flame_expression_dirs"},
            {"params": [self.pose_dirs],"lr": training_args.learn_flame_pose_dirs_lr, "name": "flame_pose_dirs"},
            {"params": [self.lbs_weights],"lr": training_args.learn_flame_lbs_weights_lr, "name": "flame_lbs_weights"},
            {"params": [self.flame_v_template],"lr": training_args.learn_flame_v_template_lr, "name": "flame_v_template"},
        ], lr=0, eps=1e-15)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling_base.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation_base.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.0049)) # TUNE
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def densification_postfix(self):
        n = self._opacity.shape[0]
        self.xyz_gradient_accum = torch.zeros((n, 1), device="cuda")
        self.denom = torch.zeros((n, 1), device="cuda")
        self.max_radii2D = torch.zeros((n), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, do_split=None, do_prune=None):
        # Pad gradients to account for gaussians added by cloning
        padded_grad = torch.zeros((self._opacity.shape[0]), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        max_scale = torch.max(self.scaling_activation(self._scaling_base)[:, :2], dim=1).values
        selected_pts_mask = torch.logical_and(selected_pts_mask, max_scale > self.percent_dense*scene_extent)

        if do_split is not None:
            do_split(selected_pts_mask)
        self.densification_postfix()

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
        if do_prune is not None:
            do_prune(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, do_clone=None):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        max_scale = torch.max(self.scaling_activation(self._scaling_base)[:, :2], dim=1).values
        selected_pts_mask = torch.logical_and(selected_pts_mask, max_scale <= self.percent_dense*scene_extent)
        
        if do_clone is not None:
            do_clone(selected_pts_mask)
        self.densification_postfix()

    def densify_and_prune(self, max_grad, min_opacity, max_screen_size, do_clone=None, do_split=None, do_prune=None):
        extent = self.cameras_extent
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # before = self._opacity.shape[0]
        # Clone small gaussians with gradient above threshold
        self.densify_and_clone(grads, max_grad, extent, do_clone=do_clone)
        # clone = self._opacity.shape[0]
        # Split large gaussians with gradient above threshold
        self.densify_and_split(grads, max_grad, extent, do_split=do_split, do_prune=do_prune)
        # split = self._opacity.shape[0]

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            max_scale = torch.max(self.scaling_activation(self._scaling_base)[:, :2], dim=1).values
            big_points_ws = max_scale > 1 * extent # TUNE
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        if do_prune is not None:
            do_prune(prune_mask)

        # prune = self._opacity.shape[0]
        # logging.info(f"Densification :: Clone: {clone - before} / Split: {split - clone} / Prune: {split - prune} / NET: {prune - before} / TOTAL: {prune}")
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
