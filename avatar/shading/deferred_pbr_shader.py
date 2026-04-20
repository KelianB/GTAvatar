from typing import List, Dict, Callable

import torch
from torch import nn, Tensor
import numpy as np
import nvdiffrast.torch as dr

from dataset import Camera
from avatar import RenderSettings
from avatar.environment_light import EnvironmentLight, lookup_env_light
from avatar.parametric_albedo import ParametricAlbedo
from utils.geometry import safe_normalize, dot
from utils.math import inverse_sigmoid


class DeferredPBRShader(nn.Module):
    def __init__(self, args, device: torch.device, num_seq: int):
        super().__init__()
        self.device = device
        self.args = args

        # Activations
        self.min_roughness, self.max_roughness = args.min_roughness, args.max_roughness
        self.roughness_activation = lambda x: torch.sigmoid(x) * (self.max_roughness - self.min_roughness) + self.min_roughness
        self.inverse_roughness_activation = lambda x: inverse_sigmoid((x - self.min_roughness) / (self.max_roughness - self.min_roughness))

        self.albedo_activation = torch.sigmoid
        self.inverse_albedo_activation = inverse_sigmoid

        self.min_reflectance, self.max_reflectance = args.min_spec, args.max_spec
        self.spec_activation = lambda x: torch.sigmoid(x) * (self.max_reflectance - self.min_reflectance) + self.min_reflectance
        self.inverse_spec_activation = lambda x: inverse_sigmoid((x - self.min_reflectance) / (self.max_reflectance - self.min_reflectance))

        match args.env_activation:
            case "relu":
                env_activation = torch.nn.ReLU()
                inverse_env_activation = torch.nn.ReLU() 
            case "sigmoid":
                env_activation = torch.sigmoid
                inverse_env_activation = inverse_sigmoid
            case "exp":
                # This is essentially log1p encoding, but avoids negative values
                env_activation = lambda x: torch.exp(x)
                inverse_env_activation = lambda x: torch.log(x.clamp(min=1e-4))
            case "softplus":
                env_activation = torch.nn.Softplus()
                inverse_env_activation = lambda x: torch.log(torch.exp(x) - 1)
            case _:
                raise ValueError(f"Unknown env activation {args.env_activation}")
        
        if args.env_multiplier != 1.0:
            # Fold the multiplier into the activation function
            prev_env_activation = env_activation
            env_activation = lambda x: prev_env_activation(x) * args.env_multiplier
            prev_inverse_env_activation = inverse_env_activation
            inverse_env_activation = lambda x: prev_inverse_env_activation(x / args.env_multiplier)

        # Initialize material
        self._init_albedo = self.inverse_albedo_activation(torch.tensor([args.initial_albedo, args.initial_albedo, args.initial_albedo], device=device))
        self._init_roughness = self.inverse_roughness_activation(torch.tensor(args.initial_roughness, device=device))
        self._init_spec = self.inverse_spec_activation(torch.tensor(args.initial_specular, device=device))

        self.parametric_albedo = ParametricAlbedo(device, args.shader_parametric_albedo) if args.shader_parametric_albedo else None 
        self.parametric_albedo_render = args.render_with_parametric_albedo

        # Use an entirely different env light for each sequence
        light_init = inverse_env_activation(args.initial_envmap_intensity * torch.ones((6, args.env_resolution, args.env_resolution, 3), dtype=torch.float, device=device))
        self.light_env = nn.ModuleList([
            EnvironmentLight(light_init, trainable=True, activation=env_activation, mip_levels=args.env_mip_levels).to(device)
            for _ in range(num_seq)
        ])

        self._FG_LUT = torch.as_tensor(np.fromfile("assets/bsdf_256_256.bin", dtype=np.float32).reshape(1, 256, 256, 2), dtype=torch.float32, device=device)

    def material(self, seq_idx: Tensor, train_iter: int) -> Tensor:
        raise NotImplementedError()

    def _activate_material(self, material: Tensor) -> Tensor:
        material = torch.cat((
            self.albedo_activation(material[..., 0:3] +\
                                   (0 if self.parametric_albedo is None else self.inverse_albedo_activation(material[..., 5:8].clamp(min=1e-6, max=1-1e-6)))), # albedo RGB
            self.roughness_activation(material[..., 3:4]), # roughness
            self.spec_activation(material[..., 4:5]), # specular
            material[..., 5:] # optional parametric albedo
        ), dim=-1)
        return material

    def forward(self, cams: List[Camera], seq_idx: Tensor, rast_buffers: Dict[str, Tensor],
                env_light=None, env_rot=None, render_settings: RenderSettings=None):
        # Use depth to compute rasterized positions
        # deformed_pos = torch.stack([cam.depth_to_points(depth) for cam, depth in zip(cams, rast_buffers["depth"])])
        # view_pos = torch.stack([v.camera_center for v in cams])
        # view_pos = view_pos[:, None, None, :] # (B, 1, 1, 3)

        if render_settings.eval_mode:
            # Cache ray directions during eval for speed
            if not hasattr(self, "ray_dirs"):
                self.ray_dirs = cams[0].compute_ray_dirs() # (H, W, 3)
            ray_dirs = self.ray_dirs.unsqueeze(0).repeat(len(cams),1,1,1) # (B, H, W, 3)
        else:
            ray_dirs = torch.stack([cam.compute_ray_dirs() for cam in cams]) # (B, H, W, 3)

        normal = safe_normalize(rast_buffers.shading_normals)

        # Ensure mip levels are computed for learned env maps
        if any(not hasattr(lgt, "diffuse") for lgt in self.light_env):
            self.update_env_lights()

        # Retrieve rasterized material (albedo, roughness, specular intensity)
        rast_material = rast_buffers["attr"] # (B,S,S,5|8)

        if env_light is None:
            env_light = [self.light_env[sidx] for sidx in seq_idx]

        # Use PBR to compute the shaded image
        render, computed_light = physical_render(self._FG_LUT, rast_material, normal, ray_dirs, env_light, env_rot,
                                                 render_settings=render_settings)
        
        rast_buffers["material"] = rast_material * rast_buffers["rend_alpha"]
        rast_buffers["light_diffuse"] = computed_light["diffuse"]
        rast_buffers["light_specular"] = computed_light["specular"]

        if self.parametric_albedo_render:
            # Shade again, with the parametric albedo
            param_albedo = rast_material[..., 5:8]
            rast_material2 = torch.cat((param_albedo, rast_material[..., 3:5]), dim=-1)
            rast_buffers["render_param_albedo"]  = physical_render(rast_material2, normal, ray_dirs, env_light, env_rot,
                                                                   render_settings=render_settings, precomputed_light=computed_light)[0]

        return render

    def update_env_lights(self, seq_idx=None):
        """ Update mip levels of learned environment maps. """
        seq_idx = range(len(self.light_env)) if seq_idx is None else seq_idx.unique()
        for sidx in seq_idx:
            self.light_env[sidx].prefilter()

    def capture(self):
        return self.state_dict()
  
    def restore(self, state):
        self.load_state_dict(state)
        self.update_env_lights()


def physical_render(
        FG_LUT: Tensor,
        material: Tensor,
        normals: Tensor,
        ray_dirs: Tensor,
        env_light: EnvironmentLight | list[EnvironmentLight] | None = None,
        env_rot: Tensor | None = None,
        precomputed_light: Dict[str, Tensor] = None,
        render_settings: RenderSettings=None
        ):
    albedo = material[..., :3] # (B, H, W, 3)
    roughness = material[..., 3:4] # (B, H, W, 1)
    spec = material[..., 4:5] # (B, H, W, 1)

    roughness = (roughness * render_settings.roughness_scale).clamp(max=1)

    if precomputed_light is None:
        # wo = safe_normalize(view_pos - deformed_pos) # (B, H, W, 3)
        wo = -safe_normalize(ray_dirs) # (B, H, W, 3)
        ndotv = dot(normals, wo).detach()
        # Reflect the camera direction on the normal vector 
        wr = 2 * normals * ndotv - wo # (B, H, W, 3)

        # Query environment light(s)
        diffuse, specular = lookup_env_light(env_light, normals, wr, roughness, rot=env_rot)

        # Compute the FG term using the LUT
        fg_uv = torch.cat((ndotv.clamp(min=1e-6), roughness), dim=-1)
        # Sample the (256, 256, 2) FG-LUT using wo.n_d and roughness
        fg_lookup = dr.texture(FG_LUT, fg_uv, filter_mode="linear", boundary_mode="clamp")
        
        # General case for a metallic material (see nvdiffrec):
        # metallic = 0.1
        # ks  = (1 - metallic) * 0.04 + albedo * metallic
        # albedo = albedo * (1 - metallic)

        F0 = spec

        # ks = F0 + (torch.max(1- roughness, F0) - F0) * torch.pow(2, (-5.55473 * ndotv - 6.698316)*ndotv)
        # ks = F0

        # During training, metallic is always 0
        metallic = render_settings.metallic
        ks  = (1 - metallic) * F0 + albedo * metallic
        albedo = albedo * (1 - metallic)

        reflectance = ks * fg_lookup[...,0:1] + fg_lookup[...,1:2]

        precomputed_light = {"diffuse": diffuse, "specular": specular, "reflectance": reflectance}
    else:
        diffuse, specular, reflectance = precomputed_light["diffuse"], precomputed_light["specular"], precomputed_light["reflectance"]

    #################### Final shading ####################
    render = diffuse*albedo*render_settings.diffuse_scale + specular*reflectance*render_settings.specular_scale # (B, H, W, 3)
    render = render * render_settings.brightness_scale

    return render, precomputed_light
