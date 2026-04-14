from typing import Dict, Union

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.transforms.functional import resize, InterpolationMode
import lpips
from nvdiffrec.render.util import cubemap_to_latlong

from avatar import Avatar, AvatarOutput
from dataset.dataset_util import SemanticMask
from utils.mesh import Mesh, find_edges
from utils.metrics import img_ssim
from utils.visualization import linear_to_srgb

"""
Many of the loss functions defined here are remnants of past experiments and are no longer used.
"""

class LossFn(nn.Module):
    weight: float
    start_iter: int

    def __init__(self, avatar: Avatar, weight: float, start_iter=0, half_life=0):
        super().__init__()
        self.weight = weight
        self.start_iter = start_iter
        self.half_life = half_life

    def forward(self, avatar: Avatar, views: Dict, out: AvatarOutput):
        raise NotImplementedError()

class LossFnWithTextures(LossFn):
    def __init__(self, avatar, *args, **kwargs):
        super().__init__(avatar, *args, **kwargs)
        if kwargs["weight"] > 0 and avatar.args.shader_type != "texture_pbr":
            raise NotImplementedError(f"{self.__class__.__name__} cannot be used with shader type {avatar.args.shader_type}")

lpips_module = None
def get_cached_lpips():
    global lpips_module
    if lpips_module is None:
        lpips_module = lpips.LPIPS(net="vgg")
    return lpips_module

class LossLPIPS(LossFn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lpips_module = get_cached_lpips() if kwargs["weight"] > 0 else None

    def forward(self, avatar, views, out: AvatarOutput):
        assert  self.lpips_module is not None, "LossLPIPS cannot be used because LPIPS was never loaded."
        head_mask = views["mask"]
        im1 = out.render * head_mask
        im2 = views["img"] * head_mask
        return self.lpips_module(im1.permute(0,3,1,2), im2.permute(0,3,1,2), normalize=True).mean()

class LossPhotoHuber(LossFn):
    def forward(self, avatar, views, out: AvatarOutput):
        return huber_loss(out.render, views["img"], 0.1)

class LossPhotoL1(LossFn):
    def forward(self, avatar, views, out: AvatarOutput):
        return (out.render - views["img"]).abs().mean()

class LossPhotoL1ParamAlbedo(LossFn):
    def forward(self, avatar, views, out: AvatarOutput):
        return (out.rast_buffers["render_param_albedo"] - views["img"]).abs().mean()

class LossPhotoL2(LossFn):
    def forward(self, avatar, views, out: AvatarOutput):
        return (out.render - views["img"]).pow(2).mean()

class LossMask(LossFn):
    def forward(self, avatar, views, out: AvatarOutput):
        gt_mask = views["mask"]
        render_mask = out.rast_buffers["rend_alpha"]
        return (render_mask - gt_mask).abs().mean()

class LossPhotoDSSIM(LossFn):
    def forward(self, avatar, views, out: AvatarOutput):
        return 1.0 - img_ssim(out.render.permute(0,3,1,2), views["img"].permute(0,3,1,2)).mean()

class LossMeshNormalConsistency(LossFn):
    """ Cosine similarity between adjacent faces of the deformed FLAME mesh. """
    def __init__(self, avatar, *args, **kwargs):
        super().__init__(avatar, *args, **kwargs)
        if kwargs["weight"] > 0:
            flame = avatar.flame
            self.mesh = Mesh(flame.v_template, flame.faces, avatar.device)
            self.mesh.compute_connectivity()

    def forward(self, avatar, views, out: AvatarOutput):
        deformed_verts = out.deformed_mesh_verts
        assert deformed_verts.shape[0] == 1
        mesh = self.mesh.with_vertices(deformed_verts.squeeze(0))
        loss = 1 - torch.cosine_similarity(mesh.face_normals[mesh.connected_faces[:, 0]], mesh.face_normals[mesh.connected_faces[:, 1]], dim=1)
        return (loss**2).mean()

class LossMeshLaplacian(LossFn):
    """ Laplacian loss on the deformed FLAME vertices. """
    def __init__(self, avatar, *args, **kwargs):
        super().__init__(avatar, *args, **kwargs)
        if kwargs["weight"] > 0:
            flame = avatar.flame
            mesh = Mesh(flame.v_template, flame.faces, avatar.device)
            mesh.compute_connectivity()
            self.laplacian = mesh.laplacian

    def forward(self, avatar, views, out: AvatarOutput):
        deformed_verts = out.deformed_mesh_verts
        loss = 0.0
        for i in range(deformed_verts.shape[0]):
            loss += self.laplacian.mm(deformed_verts[i]).norm(dim=1).pow(2).mean()
        return loss

class LossRelativeMeshLaplacian(LossFn):
    """ Laplacian loss on the difference between the deformed FLAME mesh with and without optimized vertices. """
    def __init__(self, avatar, *args, **kwargs):
        super().__init__(avatar, *args, **kwargs)
        if kwargs["weight"] > 0:
            flame = avatar.flame
            mesh = Mesh(flame.v_template, flame.faces, avatar.device)
            mesh.compute_connectivity()
            self.laplacian = mesh.laplacian

    def forward(self, avatar, views, out: AvatarOutput):
        pose = views["flame_pose"].detach()
        expr = views["flame_expression"].detach()
        # Compute deformed verts with the original FLAME model
        with torch.no_grad():
            deformed_verts_og = avatar.deformer.get_mesh_verts(pose, expr, avatar.shape_param, avatar.gaussians, avatar.flame_scale.detach(), use_original_flame=True)
        offsets = out.deformed_mesh_verts - deformed_verts_og # (B, V, 3)
        loss = 0.0
        for i in range(pose.shape[0]):
            loss += self.laplacian.mm(offsets[i]).norm(dim=1).pow(2).mean()
        return loss

class LossFlameReg(LossFn):
    """ Regularize all FLAME attributes. """
    def forward(self, avatar, views, out: AvatarOutput):
        flame = avatar.flame
        g = avatar.gaussians
        gt_v_template, gt_shape_dirs, gt_expression_dirs, gt_pose_dirs, gt_lbs_weights, gt_r_eyelid_dirs, gt_l_eyelid_dirs = flame.v_template, flame.shapedirs_identity, flame.shapedirs_expression, flame.posedirs, flame.lbs_weights, flame.r_eyelid, flame.l_eyelid
        
        ord = 2 # L2
        l = 0.0
        l += torch.linalg.vector_norm(g.flame_v_template - gt_v_template, dim=1, ord=ord) # (V)
        l += torch.linalg.vector_norm(g.shape_dirs - gt_shape_dirs, dim=1, ord=ord).sum(1) # (V) (sum over shapes)
        l += torch.linalg.vector_norm(g.expression_dirs - gt_expression_dirs, dim=1, ord=ord).sum(1) # (V) (sum over expressions)
        l += torch.linalg.vector_norm(g.pose_dirs - gt_pose_dirs, dim=1, ord=ord).sum(1) # (V) (sum over 36 shapes)
        l += torch.linalg.vector_norm(g.r_eyelid_dirs.squeeze(0) - gt_r_eyelid_dirs.squeeze(0), dim=1, ord=ord) # (V)
        l += torch.linalg.vector_norm(g.l_eyelid_dirs.squeeze(0) - gt_l_eyelid_dirs.squeeze(0), dim=1, ord=ord) # (V)
        l += torch.linalg.vector_norm(g.lbs_weights - gt_lbs_weights, dim=1, ord=ord) # (V)
        assert l.ndim == 1
        return l.mean() # mean over vertices

class LossMeshEdgeLengths(LossFn):
    """ Regularize the lengths of edges in the optimized deformed FLAME mesh to be similar to those of the original deformed FLAME mesh. """
    def __init__(self, avatar, *args, **kwargs):
        super().__init__(avatar, *args, **kwargs)
        if kwargs["weight"] > 0:
            self.edges = find_edges(avatar.flame.faces.to(torch.int64))
            
    def forward(self, avatar, views, out: AvatarOutput):
        pose = views["flame_pose"]
        expr = views["flame_expression"]
        assert pose.shape[0] == expr.shape[0] == 1, f"LossMeshEdgeLengthsReg assumes batch size of 1 - found {pose.shape[0]}"
        deformed_verts = out.deformed_mesh_verts
        # Compute deformed verts with the original FLAME model
        with torch.no_grad():
            deformed_verts_og = avatar.deformer.get_mesh_verts(pose.detach(), expr.detach(), avatar.shape_param, avatar.gaussians, avatar.flame_scale.detach(), use_original_flame=True)
    
        e0, e1 = self.edges.unbind(1)
        learned_edge_lengths  = torch.linalg.norm(deformed_verts.squeeze(0)[e0] - deformed_verts.squeeze(0)[e1], dim=-1)
        original_edge_lengths = torch.linalg.norm(deformed_verts_og.squeeze(0)[e0] - deformed_verts_og.squeeze(0)[e1], dim=-1)
        return (learned_edge_lengths - original_edge_lengths).pow(2).mean()

class LossNormalsSupervise(LossFn):
    """ Supervise the normal G-buffer with pseudo ground-truths if available. """
    def forward(self, avatar, views, out: AvatarOutput):
        l = torch.tensor(0.0, device=avatar.device)
        for mask, gt_nrm, render_nrm in zip(views["mask"], views["normals"], out.rast_buffers.rend_normal_viewspace):
            if gt_nrm is not None:
                # remap from [0, 1] to [-1, 1]
                gt_nrm = 2 * gt_nrm - 1
                # both have shape (H, W, 3)
                l += cos_loss(gt_nrm, render_nrm, weight=mask)
        return l

class LossCurvature(LossFn):
    def forward(self, avatar, views, out: AvatarOutput):
        mask_vis = out.rast_buffers.rend_alpha.detach() > 1e-5
        curv_n = torch.stack([normal2curv(n, m) for n, m in zip(out.rast_buffers.rend_normal, mask_vis)])
        return curv_n.abs().mean()

class LossSurface(LossFn):
    def forward(self, avatar, views, out: AvatarOutput):
        rast = out.rast_buffers
        mask_vis = rast.rend_alpha.detach() > 1e-5
        normal = rast.rend_normal_withoutmap if "rend_normal_withoutmap" in rast else rast.rend_normal
        d2n = rast.surf_normal
        return cos_loss(normal, d2n, weight=mask_vis)

class LossJawPose(LossFn):
    """ Regularize difference between original and current jaw pose parameters. """
    def forward(self, avatar, views, out: AvatarOutput):
        original_pose = torch.stack([avatar.dataset_train.get_flame_pose(i, avatar.device) for i in views["idx"]])
        jaw_original = original_pose[:, 6:9]
        jaw = views["flame_pose"][:, 6:9]
        d_jaw_params = jaw_original - jaw
        return (d_jaw_params ** 2).sum().mean()

class LossExpr(LossFn):
    """ Regularize difference between original and current expression parameters. """
    def forward(self, avatar, views, out: AvatarOutput):
        expr_original = avatar.expr_train[views["idx"]] # (B, 50)
        expr = views["flame_expression"]
        return (expr - expr_original).pow(2).mean()

class LossEyelids(LossFn):
    """ Regularize difference between original and current eyelid parameters. """
    def forward(self, avatar, views, out: AvatarOutput):
        original_pose = torch.stack([avatar.dataset_train.get_flame_pose(i, avatar.device) for i in views["idx"]])
        eyelids_original = original_pose[:, 15:17]
        eyelids = views["flame_pose"][:, 15:17]
        return (eyelids - eyelids_original).pow(2).mean()

class LossOpacityReg(LossFn):
    """ Push opacities toward 0 and 1. """
    def forward(self, avatar, views, out: AvatarOutput):
        o = out.opacity # (B, n, 1)
        # return torch.exp(-((o - 0.5) ** 2) / 0.05).mean()
    
        # https://github.com/turandai/gaussian_surfels/blob/main/train.py#L125
        opac_mask0 = (o > 0.01) * (o <= 0.50)
        opac_mask1 = (o > 0.50) * (o <= 0.99)
        opac_mask = opac_mask0 * 0.01 + opac_mask1
        return (torch.exp(-(o - 0.5)**2 * 20) * opac_mask).mean()

class LossDepthDistortion(LossFn):
    """ 2DGS Depth distortion loss. """
    def forward(self, avatar, views, out: AvatarOutput):
        return out.rast_buffers.depth_distort_loss.mean()

class LossUVDistortion(LossFn):
    """ UV distortion loss (penalize discrepancy of UVs along a ray). """
    def forward(self, avatar, views, out: AvatarOutput):
        l = out.rast_buffers.uv_distort_loss # (B,H,W,1)
        return l.mean()

class LossWhiteLight(LossFn):
    """ Encourage the envmap to be white/gray. """
    def forward(self, avatar, views, out: AvatarOutput):
        if True:
            # Penalize the envmap colors directly
            envmaps = torch.stack([x.base for x in avatar.shader.light_env]) # (num_seq, 6, learn_envmap_res, learn_envmap_res, 3)
            white = envmaps.mean(dim=-1, keepdim=True).detach() # (R+G+B)/3
            loss = (envmaps - white).abs().mean()
        else:
            # Penalize the sampled colors

            rast = out.rast_buffers
            mask = rast["rend_alpha"].detach() > 1e-5

            diffuse, specular = rast["light_diffuse"], rast["light_specular"]
            white = diffuse.mean(dim=-1, keepdim=True) # (R+G+B)/3
            loss = ((diffuse - white).abs() * mask).mean()
            white = specular.mean(dim=-1, keepdim=True) # (R+G+B)/3
            loss += ((specular - white).abs() * mask).mean()

        return loss

class LossRoughnessRastTV(LossFn):
    """ Total variation loss on the roughness G-buffer. """
    def forward(self, avatar, views, out: AvatarOutput):
        rast = out.rast_buffers
        roughness = rast["material"][..., 3:4] # (B,H,W,1)
        mask = views["mask"] * rast["rend_alpha"]
        return total_variation_loss(roughness, p=2, mask=mask)

class LossRastMaterialReg(LossFn):
    """ Use semantic masks to compute a Z-score on the material properties for the skin. """
    def forward(self, avatar, views, out: AvatarOutput):
        rast = out.rast_buffers
        semantic = views["semantic_mask"]
        skin_mask = semantic[..., (SemanticMask.ALL_SKIN, SemanticMask.EYEBROWS)].sum(dim=-1, keepdim=True)
        mask = skin_mask * views["mask"] * rast["rend_alpha"].detach()
        return material_zscore_loss(rast["material"], mask > 0)

class LossAlbedoTextureTV(LossFnWithTextures):
    """ Total variation loss on the albedo texture. """
    def forward(self, avatar, views, out: AvatarOutput):
        tex_albedo = out.texture_material[..., 0:3]
        return total_variation_loss(tex_albedo, p=2)

class LossRoughnessTextureTV(LossFnWithTextures):
    """ Total variation loss on the roughness texture, within valid UV areas only. """
    def __init__(self, avatar, *args, **kwargs):
        super().__init__(avatar, *args, **kwargs)
        if kwargs["weight"] > 0:
            mask = avatar.gaussians.uvmap_mask.float().unsqueeze(-1) # (H,W,1)
            size = avatar.shader.texture_mat_res
            self.mask = resize(mask.permute(2,0,1), [size, size]).permute(1,2,0)

    def forward(self, avatar, views, out: AvatarOutput):
        roughness = out.texture_material[:, :, 3].unsqueeze(-1) # (H,W,1)
        return total_variation_loss(roughness, p=2, mask=self.mask)

class LossMaterialSmoothnessUVReg(LossFnWithTextures):
    """ Total variation loss on the roughness and specular textures. """
    def forward(self, avatar, views, out: AvatarOutput):
        tex_r_spec = out.texture_material[..., 3:5]
        tex_roughness, tex_spec = tex_r_spec.unbind(-1)
        p = 2 # 1 for L1, 2 for L2
        return total_variation_loss(tex_roughness.unsqueeze(-1), p=p) + total_variation_loss(tex_spec.unsqueeze(-1), p=p)

class LossAlbedoL1Reg(LossFnWithTextures):
    """ L1 loss on the albedo texture. """
    def forward(self, avatar, views, out: AvatarOutput):
        if avatar.shader.parametric_albedo is None:
            albedo = out.texture_material[..., 0:3] # final activated albedo
        else:
            albedo = avatar.shader._material_alb # residual only
        return albedo.abs().mean()

class LossAlbedoL2Reg(LossFnWithTextures):
    """ L2 loss on the albedo texture. """
    def forward(self, avatar, views, out: AvatarOutput):
        if avatar.shader.parametric_albedo is None:
            albedo = out.texture_material[..., 0:3] # final activated albedo
        else:
            albedo = avatar.shader._material_alb # residual only
        return albedo.pow(2).mean()

class LossAlbedoSupervise(LossFn):
    """ Supervise the albedo G-buffer with pseudo ground-truths if available. """
    def forward(self, avatar, views, out: AvatarOutput):
        l = torch.tensor(0.0, device=avatar.device)
        for mask, pseudo_gt_albedo, render_albedo in zip(views["mask"], views["albedo"], out.rast_buffers["material"][..., 0:3]):
            if pseudo_gt_albedo is not None:
                render_albedo = avatar.albedo_display_transform(render_albedo)
                l += ((render_albedo - pseudo_gt_albedo) * mask).abs().mean()
        return l

class LossEnvMapTV(LossFn):
    """ Total variation loss on the learned environment map. """
    def forward(self, avatar, views, out: AvatarOutput):
        assert avatar.shader.learn_envmap, "Cannot use LossEnvMapTV if learn_envmap is false"
        loss = 0.0
        for env_light in avatar.shader.light_env:
            res = env_light.base.shape[1]
            img = cubemap_to_latlong(env_light.base, (res*2, res*2))
            loss += total_variation_loss(img, p=2)
        return loss

class LossDisplacementL1Reg(LossFn):
    """ Regularization on the displacement of the Gaussians, to encourage them to stay close to the FLAME surface. """
    def forward(self, avatar, views, out: AvatarOutput):
        return (avatar.gaussians._disp * avatar.flame_scale.detach()).abs().mean()

class LossTextureGapsReg(LossFnWithTextures):
    """ Force textures to be close to zero outside valid UVs areas. """
    def __init__(self, avatar, *args, **kwargs):
        super().__init__(avatar, *args, **kwargs)
        if kwargs["weight"] > 0:
            self.tex_nrm_identity = torch.tensor([0.0, 0.0, 1.0], device=avatar.device).view(1, 1, 3)
            res = lambda img, size: resize(img.permute(2,0,1), [size, size], interpolation=InterpolationMode.BILINEAR).permute(1,2,0)#.repeat(1,1,shape[2])
            mask = 1 - avatar.gaussians.uvmap_mask.unsqueeze(2).float()
            self.mask_tex_mat = res(mask, avatar.shader.texture_mat_res)
            self.mask_tex_nrm = res(mask, avatar.shader.texture_nrm_res)

    def forward(self, avatar, views, out: AvatarOutput):
        shader = avatar.shader
        # When using parametric albedo, use the residual only (otherwise, use the activated material)
        tex_alb = out.texture_material[..., 0:3] if shader.parametric_albedo is None else shader._material_alb
        l = (tex_alb * self.mask_tex_mat).pow(2).mean()
        tex_r_spec = out.texture_material[..., 3:5] 
        l += (tex_r_spec * self.mask_tex_mat).pow(2).mean()
        tex_nrm = out.texture_normals # (S, S, 3)
        l += (tex_nrm * self.mask_tex_nrm - self.tex_nrm_identity).pow(2).mean()
        return l

class LossNormalMapL1(LossFnWithTextures):
    """ L1 loss on the normal texture. """
    def __init__(self, avatar, *args, **kwargs):
        super().__init__(avatar, *args, **kwargs)
        if kwargs["weight"] > 0:
            self.identity = torch.tensor([0.0, 0.0, 1.0], device=avatar.device).view(1, 1, 3)

    def forward(self, avatar, views, out: AvatarOutput):
        loss = (out.texture_normals - self.identity).abs().sum(-1) # (S, S, 3)
        return loss.mean()

class LossNormalMapL2(LossFnWithTextures):
    """ L2 loss on the normal texture. """
    def __init__(self, avatar, *args, **kwargs):
        super().__init__(avatar, *args, **kwargs)
        if kwargs["weight"] > 0:
            self.identity = torch.tensor([0.0, 0.0, 1.0], device=avatar.device).view(1, 1, 3)

    def forward(self, avatar, views, out: AvatarOutput):
        loss = (out.texture_normals  - self.identity).pow(2).sum(-1) # (S, S, 3)
        return loss.mean()

class LossNormalMapCosine(LossFn):
    """ Cosine loss on the normal texture. """
    def __init__(self, avatar, *args, **kwargs):
        super().__init__(avatar, *args, **kwargs)
        if kwargs["weight"] > 0:
            size = avatar.shader.texture_nrm_res
            self.identity = torch.tensor([0.0, 0.0, 1.0], device=avatar.device).view(1, 1, 3).repeat(size, size, 1)

    def forward(self, avatar, views, out: AvatarOutput):
        return cos_loss(out.texture_normals, self.identity)

class LossAnisotropy(LossFn):
    """ Encourage the Gaussians to be isotropic by penalizing the ratio between the x and y scales. """
    def forward(self, avatar, views, out: AvatarOutput):
        gaussians = avatar.gaussians
        scales = gaussians.scaling_activation(gaussians._scaling_base) # non-deformed scales
        sx, sy = scales[..., :2].unbind(-1)
        loss = torch.max(sx/sy, sy/sx)
        return loss.mean()

class LossRelativeRotReg(LossFn):
    """ Regularization term for rotations of Gaussians relative to the mesh. """
    def __init__(self, avatar, *args, **kwargs):
        super().__init__(avatar, *args, **kwargs)
        self.identity_rot = torch.zeros((4), dtype=torch.float, device=avatar.device)
        self.identity_rot[0] = 1

    def forward(self, avatar, views, out: AvatarOutput):
        rot = avatar.gaussians.rotation_activation(avatar.gaussians._rotation_base)
        identity_rot = self.identity_rot.unsqueeze(0).repeat(rot.shape[0], 1)
        return (rot - identity_rot).pow(2).mean()

class LossBaryL2Reg(LossFn):
    """ Regularize the barycentrics to be close to 1/3, which encourages the Gaussians to be close to the center of their triangle. """
    def forward(self, avatar, views, out: AvatarOutput):
        _, bary = avatar.gaussians.get_binding()
        return (bary - 1/3).pow(2).mean()


losses: Dict[str, type[LossFn]] = {
    "photo_huber": LossPhotoHuber,
    "photo_l1": LossPhotoL1,
    "photo_l1_parametric_albedo": LossPhotoL1ParamAlbedo,
    "photo_l2": LossPhotoL2,
    "photo_dssim": LossPhotoDSSIM,
    "lpips": LossLPIPS,
    "mask": LossMask,
    "flame_reg": LossFlameReg,
    "normals_supervise": LossNormalsSupervise,
    "curvature": LossCurvature,
    "surface": LossSurface,
    "opacity_reg": LossOpacityReg,
    "depth_distortion": LossDepthDistortion,
    "uv_distortion": LossUVDistortion,
    "white_light": LossWhiteLight,
    "material_reg": LossRastMaterialReg,
    "albedo_smoothness_uv_reg": LossAlbedoTextureTV,
    "albedo_L1_reg": LossAlbedoL1Reg,
    "albedo_L2_reg": LossAlbedoL2Reg,
    "albedo_supervise": LossAlbedoSupervise,
    "roughness_rast_tv": LossRoughnessRastTV,
    "material_smoothness_uv_reg": LossMaterialSmoothnessUVReg,
    "roughness_texture_tv": LossRoughnessTextureTV,
    "normal_consistency_mesh": LossMeshNormalConsistency,
    "laplacian_mesh": LossMeshLaplacian,
    "laplacian_mesh_relative": LossRelativeMeshLaplacian,
    "mesh_edge_lengths": LossMeshEdgeLengths,
    "anisotropy": LossAnisotropy,
    "envmap_tv": LossEnvMapTV,
    "disp_L1_reg": LossDisplacementL1Reg,
    "texture_gaps_reg": LossTextureGapsReg,
    "jaw_pose_reg": LossJawPose,
    "expr_reg": LossExpr,
    "eyelids_reg": LossEyelids,
    "relative_rot_reg": LossRelativeRotReg,
    "bary_reg_l2": LossBaryL2Reg,
    "normal_map_l1": LossNormalMapL1,
    "normal_map_l2": LossNormalMapL2,
    "normal_map_cosine": LossNormalMapCosine,
}

########################################################################################################################

def normal2curv(normal, mask):
    # this expects normal and mask with shapes (H, W, C)
    n, m = normal, mask
    n = F.pad(n[None], [0, 0, 1, 1, 1, 1], mode='replicate')
    m = F.pad(m[None].to(torch.float32), [0, 0, 1, 1, 1, 1], mode='replicate').to(torch.bool)
    n_c = (n[:, 1:-1, 1:-1, :]      ) * m[:, 1:-1, 1:-1, :]
    n_u = (n[:,  :-2, 1:-1, :] - n_c) * m[:,  :-2, 1:-1, :]
    n_l = (n[:, 1:-1,  :-2, :] - n_c) * m[:, 1:-1,  :-2, :]
    n_b = (n[:, 2:  , 1:-1, :] - n_c) * m[:, 2:  , 1:-1, :]
    n_r = (n[:, 1:-1, 2:  , :] - n_c) * m[:, 1:-1, 2:  , :]
    curv = (n_u + n_l + n_b + n_r)[0] # (H, W, 3)
    curv = curv * mask
    curv = torch.linalg.vector_norm(curv, dim=-1, ord=1, keepdim=True)
    return curv

def huber_loss(network_output, gt, alpha):
    diff = torch.abs(network_output - gt)
    mask = (diff < alpha).float()
    loss = 0.5*diff**2*mask + alpha*(diff-0.5*alpha)*(1.-mask)
    return loss.mean()

def cos_loss(output, gt, weight=1):
    """ Cosine similarity loss. """
    cos = torch.sum(output * gt * weight, -1)
    return (1 - cos).mean()

def total_variation_loss(img: Tensor, mask: Union[Tensor, None] = None, p=1):
    """ Compute total variation loss on a (B,H,W,C) shaped image tensor. """
    assert (img.ndim == 3 or img.ndim == 4) and (mask is None or mask.ndim == img.ndim), f"total_variation_loss: Expected 3D or 4D tensor for img and mask, got {img.ndim}D and {mask.ndim}D"
    if img.ndim == 3:
        img = img.unsqueeze(0)
    B, H, W, C = img.shape
    assert C < 10, f"total_variation_loss expects img in BHWC shape, got shape {img.shape}"
    vy = img[:,1:,:,:] - img[:,:-1,:,:]
    vx = img[:,:,1:,:] - img[:,:,:-1,:]
    if mask is not None:
        if mask.ndim == 3:
            mask = mask.unsqueeze(0)
        my = ((mask[:,1:,:,:] + mask[:,:-1,:,:]) / 2) > 0.99
        mx = ((mask[:,:,1:,:] + mask[:,:,:-1,:]) / 2) > 0.99
        vy = vy * my
        vx = vx * mx
    if p == 1:
        tv_y = vy.abs().sum()
        tv_x = vx.abs().sum()
    elif p == 2:
        tv_y = vy.pow(2).sum()
        tv_x = vx.pow(2).sum()
    else:
        raise NotImplementedError(f"total_variation_loss p={p}")
    return (tv_x + tv_y) / (B * C * H * W)

def img_grad(img):
    # Gradient along height and width
    dy, dx = torch.gradient(img.permute(0,3,1,2), dim=(2, 3))
    dy, dx = dy.permute(0,2,3,1), dx.permute(0,2,3,1)
    return torch.stack((dx, dy), dim=-1)

def material_zscore_loss(material, mask, float_mask=False):
    # Regularize roughness
    roughness = material[..., 3:4] # (B, H, W, 1)
    roughness_skin = (roughness * mask) if float_mask else roughness[mask]
    z_score = (roughness_skin - 0.500) / 0.100
    loss_roughness = (z_score.abs() - 2).clamp(min=0).mean()
    # Regularize specular
    loss_spec = 0
    # spec = material[..., 4:5] # (B, H, W, 1)
    # k_skin = (spec * mask) if float_mask else spec[mask]
    # z_score = (k_skin - 0.3753) / 0.1655
    # loss_spec = (z_score.abs() - 2).clamp(min=0).mean()
    return loss_roughness + loss_spec
