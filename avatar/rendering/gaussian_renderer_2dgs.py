import torch
from torch import Tensor
import math
from functools import reduce

from diff_surfel_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from dataset import Camera
from avatar import RenderSettings

'''
For using regular 2DGS instead of our textured variant:
(
    git clone https://github.com/hbb1/diff-surfel-rasterization.git --recurse-submodules /tmp/diff-surfel-rasterization &&
    cp assets/diff-surfel-rasterization.patch /tmp/diff-surfel-rasterization &&
    cd /tmp/diff-surfel-rasterization &&
    git checkout e0ed0207b3e0669960cfad70852200a4a5847f61 &&
    git apply diff-surfel-rasterization.patch &&
    pip install --no-build-isolation .
);
rm -rf /tmp/diff-surfel-rasterization

Config changes are also required:
    shader_type = primitive_pbr
    loss_roughness_texture_tv_weight = 0
    loss_texture_gaps_reg_weight = 0
    loss_normal_map_cosine_weight = 0
    loss_albedo_L1_reg_weight = 0
'''

DEPTH_MODE = "expected"
# DEPTH_MODE = "median"

def render(cameras: list[Camera], pos: Tensor, rot: Tensor, scaling: Tensor, opacity: Tensor, colors: Tensor, background_color: Tensor, settings: RenderSettings):
    """
    Render a batch of scenes.
    pos: (B, n, 3)
    rot: (B, n, 4)
    scaling: (B, n, 1)
    opacity: (B, n, 1)
    colors: (B, n, c)
    background_color: (3)
    """

    device = colors.device
    n_channels = colors.shape[-1]
    n_gaussians = pos.shape[1]

    # Pad colors to the right number of channels
    # n_channels_expected = 5
    # padding = torch.ones((*colors.shape[:-1], n_channels_expected-n_channels), dtype=colors.dtype, device=colors.device)
    # padding.requires_grad_()
    # colors = torch.cat((colors, padding), dim=-1)

    # Add channels to the background color to match the gaussian colors
    # With the gsplat rasterizer, add +1 channel for the depth, since we use render_mode="RGB+ED"
    background_channels = colors.shape[-1]
    background_color = torch.cat((background_color, background_color[0]*torch.ones((background_channels-background_color.shape[-1]), dtype=background_color.dtype, device=device)), dim=-1)

    render_outputs = dict()
    screenspace_points_tensors = []

    for i, cam in enumerate(cameras):
        means = pos[i]
        quats = rot[i]
        scales = scaling[i]
        opa = opacity[i] # (N, 1)
        col = colors[i] # (V, D)

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(means, dtype=means.dtype, requires_grad=True, device=device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(cam.FoVx * 0.5)
        tanfovy = math.tan(cam.FoVy * 0.5)

        # Apply masks for visualization/debugging
        masks = []
        if settings.gaussians_mask is not None:
            masks.append(settings.gaussians_mask)
        if settings.decimation_ratio > 0:
            g = torch.Generator(device)
            g.manual_seed(settings.decimation_seed)
            indices = torch.randperm(n_gaussians, generator=g, device=device)[:math.floor((1-settings.decimation_ratio) * n_gaussians)]
            m = torch.zeros((n_gaussians), dtype=torch.bool, device=device)
            m[indices] = True
            masks.append(m)
        if settings.opacity_culling != 0:
            masks.append(opa >= settings.opacity_culling)

        if masks:
            masks = [m.view(n_gaussians) for m in masks]
            mask = reduce(torch.logical_and, masks)
            means, quats, scales, opa, col, screenspace_points = means[mask], quats[mask], scales[mask], opa[mask], col[mask], screenspace_points[mask]

        if settings.override_opacity != -1:
            opa = opa*0 + settings.override_opacity
        
        if settings.random_colors:
            g = torch.Generator(device)
            g.manual_seed(0)
            col = torch.rand(col.shape, generator=g, device=device)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(cam.image_height),
            image_width=int(cam.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=background_color,
            scale_modifier=1.0,
            viewmatrix=cam.world_view_transform,
            projmatrix=cam.full_proj_transform,
            sh_degree=0,
            campos=cam.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        rendered_image, radii, allmap = rasterizer(
            means3D=means,
            means2D=screenspace_points,
            shs=None,
            colors_precomp=col,
            opacities=opa,
            scales=scales[..., :2],
            rotations=quats,
            cov3D_precomp=None
        )

        render_alpha = allmap[1:2]
        render_normal = allmap[2:5]
        # transform normals from view space to world space
        render_normal = (render_normal.permute(1,2,0) @ (cam.world_view_transform[:3,:3].T)).permute(2,0,1)
        render_depth_median = torch.nan_to_num(allmap[5:6], 0, 0)
        render_depth_expected = torch.nan_to_num(allmap[0:1] / render_alpha, 0, 0)
        render_dist = allmap[6:7] # depth distortion map

        depth_ratio = {"expected": 0, "median": 1}[DEPTH_MODE]
        # surf depth is either median or expected by setting depth_ratio to 1 or 0
        # for bounded scene, use median depth, i.e., depth_ratio = 1; 
        # for unbounded scene, use expected depth, i.e., depth_ratio = 0, to reduce disk aliasing.
        depth = render_depth_expected * (1-depth_ratio) + (depth_ratio) * render_depth_median

        rendered_colors = rendered_image.permute(1,2,0).unsqueeze(0)[..., :n_channels]
        alphas = render_alpha.permute(1,2,0).unsqueeze(0)
        normals_worldspace = render_normal.permute(1,2,0).unsqueeze(0)
        normals_viewspace = render_normal.permute(1,2,0).unsqueeze(0)
        distort_loss = render_dist.permute(1,2,0).unsqueeze(0)
        depth = depth.permute(1,2,0).unsqueeze(0)

        # Assume the depth points form the 'surface' and generate pseudo surface normal for regularizations.
        surf_normal = cam.depth_to_normal(depth.squeeze(0))
        # remember to multiply with accum_alpha since render_normal is unnormalized.
        surf_normal = surf_normal.unsqueeze(0) * alphas.detach()
    
        render_pkg = {
            "render": rendered_colors,
            "rend_alpha": alphas,
            "rend_normal": normals_worldspace,
            "rend_normal_viewspace": normals_viewspace,
            "depth_distort_loss": distort_loss,
            "depth": depth,
            "surf_normal": surf_normal,
            "radii": radii.unsqueeze(0),
        }
        screenspace_points_tensors.append(screenspace_points)

        for key, tensor in render_pkg.items():
            render_outputs[key] = render_outputs.get(key, []) + [tensor]

    # Stack all tensors, except screenspace_points to keep gradients intact
    render_outputs = dict((key, torch.cat(tensors)) for key, tensors in render_outputs.items())
    render_outputs["screenspace_points"] = screenspace_points_tensors

    return render_outputs
