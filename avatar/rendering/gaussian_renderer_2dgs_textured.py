import torch
from torch import Tensor
import math
from functools import reduce

from diff_surfel_rasterization_uv_tex import GaussianRasterizationSettings, CudaTexture2D, rasterize_gaussians
from dataset import Camera
from avatar import RenderSettings


DEPTH_MODE = "expected"
# DEPTH_MODE = "median"

cuda_texture = None
cuda_texture_nrm = None

def discard_hw_textures():
    global cuda_texture, cuda_texture_nrm
    cuda_texture = None
    cuda_texture_nrm = None

def render(cameras: list[Camera], pos: Tensor, rot: Tensor, scaling: Tensor, opacity: Tensor, texture: Tensor, texture_normals: Tensor,
           background_color: Tensor, uv0: Tensor, uv_jacobian: Tensor, settings: RenderSettings,
           uv_dist_threshold=1000.0):
    """
    Render a batch of scenes.
    pos: (B, n, 3)
    rot: (B, n, 4)
    scaling: (B, n, 1)
    opacity: (B, n, 1)
    texture: (t, t, c)
    background_color: (3)
    uv0: (n, 2)
    uv_jacobian: (n, 2, 2)
    texture_normals: (tn, tn, 3)
    """

    device = texture.device
    n_channels = texture.shape[-1]
    n_gaussians = pos.shape[1]


    # Pad colors to the right number of channels
    # n_channels_expected = 5
    # padding = torch.ones((*texture.shape[:-1], n_channels_expected-n_channels), dtype=texture.dtype, device=texture.device)
    # padding.requires_grad_()
    # texture = torch.cat((texture, padding), dim=-1)

    # Add channels to the background color to match the gaussian colors
    # With the gsplat rasterizer, add +1 channel for the depth, since we use render_mode="RGB+ED"
    background_channels = texture.shape[-1]
    background_color = torch.cat((background_color, background_color[0]*torch.ones((background_channels-background_color.shape[-1]), dtype=background_color.dtype, device=device)), dim=-1)
    
    if settings.random_colors:
        g = torch.Generator(device)
        g.manual_seed(0)
        texture = torch.rand(texture.shape, generator=g, device=device)
        uv0 = torch.rand(uv0.shape, generator=g, device=device)
        uv_jacobian = uv_jacobian * 0

    # 3-channel hardware accelerated textures are not supported - pad the normal map to 4 channels
    texture_normals = torch.cat((texture_normals, torch.zeros_like(texture_normals[..., (0,)])), dim=-1)
 
    if settings.hw_textures:
        global cuda_texture, cuda_texture_nrm
        if cuda_texture is None:
            cuda_texture = CudaTexture2D(texture, True)
        if cuda_texture_nrm is None:
            cuda_texture_nrm = CudaTexture2D(texture_normals, True)
        texture = cuda_texture
        texture_normals = cuda_texture_nrm

    render_outputs = dict()
    screenspace_points_tensors = []

    for i, cam in enumerate(cameras):
        means = pos[i]
        quats = rot[i]
        scales = scaling[i]
        opa = opacity[i] # (N, 1)
        uvJ = uv_jacobian[i]

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
            means, quats, scales, opa, uv0, uvJ, screenspace_points = means[mask], quats[mask], scales[mask], opa[mask], uv0[mask], uvJ[mask], screenspace_points[mask]

        if settings.override_opacity != -1:
            opa = opa*0 + settings.override_opacity

        raster_settings = GaussianRasterizationSettings(
            image_height=int(cam.image_height),
            image_width=int(cam.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=background_color,
            scale_modifier=1.0,
            viewmatrix=cam.world_view_transform,
            projmatrix=cam.full_proj_transform,
            campos=cam.camera_center,
            debug=False,
        )

        rendered_image, radii, allmap = rasterize_gaussians(
            means3D=means,
            means2D=screenspace_points,
            opacities=opa,
            scales=scales[..., :2],
            rotations=quats,
            uv0=uv0,
            uvJ=uvJ,
            texture=texture,
            texture_normals=texture_normals,
            raster_settings=raster_settings,
            uv_dist_threshold=uv_dist_threshold,
        )

        render_alpha = allmap[1:2]
        render_normal = allmap[8:11] # in world space
        render_normal_withoutmap = allmap[2:5] # in world space
        render_depth_median = torch.nan_to_num(allmap[5:6], 0, 0)
        render_depth_expected = torch.nan_to_num(allmap[0:1] / render_alpha, 0, 0)
        render_depth_dist = allmap[6:7] # depth distortion map
        render_uv_dist = allmap[7:8] # uv distortion map

        # depth = render_depth_expected * (1-depth_ratio) + (depth_ratio) * render_depth_median
        depth = render_depth_expected if DEPTH_MODE == "expected" else render_depth_median

        rendered_colors = rendered_image.permute(1,2,0).unsqueeze(0)[..., :n_channels]
        alphas = render_alpha.permute(1,2,0).unsqueeze(0)
        normals_worldspace = render_normal.permute(1,2,0).unsqueeze(0)
        normals_viewspace = render_normal.permute(1,2,0).unsqueeze(0)
        render_normal_withoutmap = render_normal_withoutmap.permute(1,2,0).unsqueeze(0)
        depth_distort_loss = render_depth_dist.permute(1,2,0).unsqueeze(0)
        uv_distort_loss = render_uv_dist.permute(1,2,0).unsqueeze(0)
        depth = depth.permute(1,2,0).unsqueeze(0)
            
        if settings.eval_mode:
            surf_normal = normals_worldspace * 0
        else:
            # Assume the depth points form the 'surface' and generate pseudo surface normal for regularizations.
            surf_normal = cam.depth_to_normal(depth.squeeze(0))
            # remember to multiply with accum_alpha since render_normal is unnormalized.
            surf_normal = surf_normal.unsqueeze(0) * alphas.detach()

        render_pkg = {
            "render": rendered_colors,
            "rend_alpha": alphas,
            "rend_normal": normals_worldspace,
            "rend_normal_withoutmap": render_normal_withoutmap,
            "rend_normal_viewspace": normals_viewspace,
            "depth_distort_loss": depth_distort_loss,
            "depth": depth,
            "surf_normal": surf_normal,
            "radii": radii.unsqueeze(0).max(dim=-1).values, # (n,2) to (1,n)
            "uv_distort_loss": uv_distort_loss,
        }
        screenspace_points_tensors.append(screenspace_points)

        for key, tensor in render_pkg.items():
            render_outputs[key] = render_outputs.get(key, []) + [tensor]

    # Stack all tensors, except screenspace_points to keep gradients intact
    render_outputs = dict((key, torch.cat(tensors)) for key, tensors in render_outputs.items())
    render_outputs["screenspace_points"] = screenspace_points_tensors

    return render_outputs
