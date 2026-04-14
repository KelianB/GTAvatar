from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from avatar import Avatar


def prune_optimizer(optimizer: torch.optim.Optimizer, mask: torch.Tensor, param_names: list[str]=None) -> Dict[str, torch.Tensor]:
    ''' Apply a mask to an optimizer's parameters and their corresponding state. Returns the updated parameter tensors.  '''
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        if param_names is None or group["name"] in param_names:
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

    return optimizable_tensors


def concat_tensors_to_optimizer(optimizer: torch.optim.Optimizer, tensors_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    ''' Concatenate tensors with an optimizer's parameters and initialize the corresponding state. Returns the updated parameter tensors.  '''
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        if group["name"] in tensors_dict:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

    return optimizable_tensors

""" The following methods were extracted from the densification logic of gaussian_model.py """

def do_clone(avatar: Avatar, selected_pts_mask: Tensor):
    args, gaussians, shader = avatar.args, avatar.gaussians, avatar.shader

    optimizable_tensors = {
        "opacity": gaussians._opacity,
        "scaling" : gaussians._scaling_base,
        "rotation" : gaussians._rotation_base,
        "disp": gaussians._disp,
        "bary": gaussians.bary_coords,
    }
    if args.shader_type == "primitive_pbr":
        optimizable_tensors["shader_material"] = shader._material

    for key in optimizable_tensors:
        optimizable_tensors[key] = optimizable_tensors[key][selected_pts_mask]

    optimizable_tensors = concat_tensors_to_optimizer(gaussians.optimizer, optimizable_tensors)
    gaussians._opacity = optimizable_tensors["opacity"]
    gaussians._scaling_base = optimizable_tensors["scaling"]
    gaussians._rotation_base = optimizable_tensors["rotation"]
    gaussians._disp = optimizable_tensors["disp"]
    gaussians.bary_coords = optimizable_tensors["bary"]
    gaussians.triangle_idx = torch.cat((gaussians.triangle_idx, gaussians.triangle_idx[selected_pts_mask]), dim=0)
    if args.shader_type == "primitive_pbr":
        shader._material = optimizable_tensors["shader_material"]

def do_split(avatar: Avatar, selected_pts_mask: Tensor):
    args, gaussians, shader = avatar.args, avatar.gaussians, avatar.shader
    N = 2

    old_scales = gaussians.scaling_activation(gaussians._scaling_base[selected_pts_mask]).repeat(N,1)
    optimizable_tensors = {
        "scaling": gaussians.scaling_inverse_activation(old_scales / (0.8*N)),
        "rotation": gaussians._rotation_base[selected_pts_mask].repeat(N,1),
        "opacity": gaussians._opacity[selected_pts_mask].repeat(N,1),
    }
    if args.shader_type == "primitive_pbr":
        optimizable_tensors["shader_material"] = shader._material[selected_pts_mask].repeat(N,1)

    disp = gaussians._disp[selected_pts_mask].repeat(N,1)
    d_std = 0
    optimizable_tensors["disp"] = torch.normal(mean=disp, std=d_std)

    bary = gaussians.bary_coords[selected_pts_mask].repeat(N,1)
    bary_std = 0.1
    optimizable_tensors["bary"] = torch.normal(mean=bary, std=bary_std)

    optimizable_tensors = concat_tensors_to_optimizer(gaussians.optimizer, optimizable_tensors)
    gaussians._opacity = optimizable_tensors["opacity"]
    gaussians._scaling_base = optimizable_tensors["scaling"]
    gaussians._rotation_base = optimizable_tensors["rotation"]
    gaussians._disp = optimizable_tensors["disp"]
    gaussians.bary_coords = optimizable_tensors["bary"]
    gaussians.triangle_idx = torch.cat((gaussians.triangle_idx, gaussians.triangle_idx[selected_pts_mask].repeat(N)), dim=0)
    if args.shader_type == "primitive_pbr":
        shader._material = optimizable_tensors["shader_material"]

def do_prune(avatar: Avatar, prune_mask: Tensor):
    args, gaussians, shader = avatar.args, avatar.gaussians, avatar.shader
    
    mask = ~prune_mask

    param_names = ["opacity", "scaling", "rotation", "disp", "bary"]
    if args.shader_type == "primitive_pbr":
        param_names.append("shader_material")
    optimizable_tensors = prune_optimizer(gaussians.optimizer, mask, param_names=param_names)
    gaussians._opacity = optimizable_tensors["opacity"]
    gaussians._scaling_base = optimizable_tensors["scaling"]
    gaussians._rotation_base = optimizable_tensors["rotation"]
    gaussians._disp = optimizable_tensors["disp"]
    gaussians.bary_coords = optimizable_tensors["bary"]
    gaussians.triangle_idx = gaussians.triangle_idx[mask]
    if args.shader_type == "primitive_pbr":
        shader._material = optimizable_tensors["shader_material"]
