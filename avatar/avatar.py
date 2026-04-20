import logging
from pathlib import Path
import random
from typing import List, Callable

import torch
from torch import Tensor
import numpy as np

from dataset import MonoFaceDataset, HRAvatarMonocularDataset, DummyDataset, DeviceDataLoader, to_device_recursive
from flame import FLAME
from avatar.gaussian_model import GaussianModel
from avatar.deformer import GaussianDeformer
from avatar.shading import DeferredPBRShader, PrimitiveDeferredPBRShader, TexturedDeferredPBRShader
from avatar.environment_light import EnvironmentLight, get_env_light_background
from avatar.types import RenderSettings, AvatarOutput
from utils.general import DotDict, timeblock
from utils.math import build_scaling_rotation
from utils.geometry import AABB, compute_vertices_face_normals
from utils.p3d_rasterizer import Pytorch3dRasterizer, vertices_to_face
from utils.visualization import blend_img, srgb_to_linear, linear_to_srgb, tonemapping
from utils.tqdm import tqdm

default_render_settings = RenderSettings()

class Avatar:
    gaussians: GaussianModel
    deformer: GaussianDeformer
    shader: DeferredPBRShader

    def __init__(self, args):
        self.args = args

        #################### Select device ####################
        device = torch.device("cpu")
        if torch.cuda.is_available() and args.device >= 0:
            device = torch.device(f"cuda:{args.device}")
            logging.info(f"Using device {device} ({torch.cuda.get_device_name(device)})")
        else:
            logging.info(f"Using device {device}")
        self.device = device

        seed = 1138
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.experiment_dir: Path = args.output_dir / args.run_name
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        logging.info(f"Output directory: {self.experiment_dir.absolute()}")

        # if args.wandb:
        #     import wandb
        #     wandb_path = self.experiment_dir / "wandb"
        #     wandb_path.mkdir(parents=True, exist_ok=True)
        #     os.environ['WANDB_DIR'] = str(wandb_path)
        #     wandb.init(project=args.wandb_workspace, name=run_name)

        self.white_bg = torch.tensor([1, 1, 1], dtype=torch.float32, device=args.device)
        self.black_bg = self.white_bg * 0

        # Apply tonemapping to get a LDR image, then convert to sRGB for display.
        # The target images are assumed to be sRGB, so this should be applied to the rendered image before computing the loss.
        self.display_transform = lambda x: linear_to_srgb(tonemapping(x))
        # Albedo is already [0,1]-bounded and should not be tonemapped, but still needs to be converted to sRGB for display
        self.albedo_display_transform = linear_to_srgb
        self.inverse_albedo_display_transform = srgb_to_linear

    def _init_dataset(self):
        args = self.args
        
        if args.detached:
            dataset_train = DummyDataset()
            dataset_test = DummyDataset()
        else:
            load_normals = args.loss_normals_supervise_weight > 0
            load_albedo = args.loss_albedo_supervise_weight > 0

            train_args = {"seq_start": 0, "seq_end": -args.test_set_num, "sample_ratio": args.sample_idx_ratio, "load_normals": load_normals, "load_albedo": load_albedo}
            test_args =  {"seq_start": -args.test_set_num, "seq_end": None, "sample_ratio": 1, "load_normals": False, "load_albedo": False}

            logging.info("Creating dataset (train)...")
            if args.dataset_type == "HRAvatar":
                dataset_train = HRAvatarMonocularDataset(args.train_dirs, **train_args)
            elif args.dataset_type == "MonoFace":
                dataset_train = MonoFaceDataset(args.train_dirs, head_only=True, **train_args)
            else:
                raise NotImplementedError(f"Unknown --dataset_type '{args.dataset_type}'")

            logging.info("Creating dataset (test)...")
            if args.dataset_type == "HRAvatar":
                dataset_test = HRAvatarMonocularDataset(args.train_dirs, **test_args)
            elif args.dataset_type == "MonoFace":
                dataset_test = MonoFaceDataset(args.train_dirs, head_only=True, **test_args)
            else:
                raise NotImplementedError(f"Please specify --dataset_type ('HRAvatar' or 'MonoFace')" if args.dataset_type is None else f"Unknown --dataset_type '{args.dataset_type}'")

        self.dataset_train = dataset_train
        self.dataset_test = dataset_test

        # Get all FLAME parameters
        self.pose_train = torch.stack([dataset_train.get_flame_pose(i, self.device) for i in range(len(dataset_train))])
        self.expr_train = torch.stack([dataset_train.get_flame_expression(i, self.device) for i in range(len(dataset_train))])      
        self.pose_test = torch.stack([dataset_test.get_flame_pose(i, self.device) for i in range(len(dataset_test))])
        self.expr_test = torch.stack([dataset_test.get_flame_expression(i, self.device) for i in range(len(dataset_test))])      


    @torch.no_grad()
    def init_modules(self):
        device, args = self.device, self.args
        
        self._init_dataset()

        logging.info("Initializing FLAME...")
        self.flame = FLAME(n_shape_params=100, n_expr_params=50, add_teeth=args.add_teeth, add_mouth_interior=args.add_mouth_interior).to(device)
       
        self.shape_param = torch.nn.Parameter(self.dataset_train.shape_params.clone().to(device))
        flame_scale = 4 if args.dataset_type == "HRAvatar" else 1
        self.flame_scale = torch.nn.Parameter(torch.tensor(flame_scale, dtype=torch.float, device=device))       

        self.gaussians = GaussianModel(args, self.flame)
        self.gaussians.create_uvs_triangle(self.flame, self.flame_scale, args.gaussians_init_opacity, args.gaussians_init_scale, args.gaussians_init_count)
        n_gaussians = self.gaussians.n_gaussians
        logging.info(f"Initialized with {n_gaussians} gaussians.")
        
        self.deformer = GaussianDeformer(args, device, self.flame, self.shape_param.detach())

        if args.shader_type == "primitive_pbr":
            self.shader = PrimitiveDeferredPBRShader(args, device, self.dataset_train.num_seq, self.gaussians)
        elif args.shader_type == "texture_pbr":
            self.shader = TexturedDeferredPBRShader(args, device, self.dataset_train.num_seq)
        else:
            raise NotImplementedError()

        if args.learn_tracking == "smirk" and not args.detached:
            from tracking.smirk import SMIRKWrapper
            logging.info("Loading SMIRK tracker")
            self.tracker = SMIRKWrapper(self.args, device)

    def run(self, views, measure_time=False, env_light=None, env_rot=None, render_settings=default_render_settings, train_iter=1e9):
        deformer, shader, gaussians, flame = self.deformer, self.shader, self.gaussians, self.flame
        is_textured_shader = isinstance(shader, TexturedDeferredPBRShader)
        B, H, W = views["img"].shape[0:3]
        H, W = views["camera"][0].image_height, views["camera"][0].image_width

        if env_rot is not None:
            assert env_rot.shape == (B,4,4), f"invalid env_rot shape: {env_rot.shape}, should be ({B},4,4)"

        # ================================================================================
        # Geometry deformation
        # ================================================================================
        with timeblock(enabled=measure_time) as t_geometry:
            pose = views["flame_pose"]
            expr = views["flame_expression"]
            pos, rot, scaling, opacity, mesh_verts = deformer(pose, expr, self.shape_param, gaussians, self.flame_scale, get_mesh_verts=True)
            scaling = scaling * render_settings.scaling_multiplier

        # ================================================================================
        # Compute Jacobians to transform from gaussian tangent planes ST to UV space
        # ================================================================================
        if is_textured_shader:
            with timeblock(enabled=measure_time) as t_juvst:
                gaussians.J_uv_st = compute_jacobians(self.args, flame, gaussians, pos, rot, scaling, mesh_verts, train_iter)

        if is_textured_shader:
            texture_material, texture_normals = shader.get_textures(views["seq_idx"], train_iter)
            features = None
        else:
            texture_material, texture_normals = None, None
            features = shader.get_render_features(views["seq_idx"], train_iter)

        background_color = self.white_bg if render_settings.background_color == "white" else \
                           self.black_bg if render_settings.background_color == "black" else \
                           render_settings.background_color

        # ================================================================================
        # Splatting
        # ================================================================================
        with timeblock(enabled=measure_time) as t_splat:
            if is_textured_shader:
                uvs = gaussians.get_uvs()
                J_uv_st = gaussians.J_uv_st
                # Remove parametric albedo if we're not rendering with it
                texture_material_render = texture_material[..., :5] if not shader.parametric_albedo_render else texture_material

                from avatar.rendering.gaussian_renderer_2dgs_textured import render as render_tex
                rast_buffers = render_tex(views["camera"], pos, rot, scaling, opacity, texture_material_render, texture_normals, background_color,
                                          uvs, J_uv_st,uv_dist_threshold=self.args.loss_uv_distortion_threshold, settings=render_settings)
            else:
                # Remove parametric albedo if we're not rendering with it
                features_render = features[..., :5] if not shader.parametric_albedo_render else features

                from avatar.rendering.gaussian_renderer_2dgs import render as render_2dgs
                rast_buffers = render_2dgs(views["camera"], pos, rot, scaling, opacity, features_render, background_color, settings=render_settings)

        rast_buffers["attr"] = rast_buffers["render"]
        del rast_buffers["render"]
        # Convert rast_buffers to a DotDict for convenience
        rast_buffers = DotDict(rast_buffers)

        match render_settings.shading_normals:
            case "splatted":
                rast_buffers.shading_normals = rast_buffers.rend_normal 
            case "depth":
                rast_buffers.shading_normals = rast_buffers.surf_normal 
            case "mesh":
                rast_buffers.shading_normals = self.get_rasterized_mesh(views, smooth_normals=True)["smooth_normals"]
            case "mesh_flat":
                rast_buffers.shading_normals = self.get_rasterized_mesh(views, flat_normals=True)["flat_normals"]
            case _:
                raise NotImplementedError()

        # ================================================================================
        # Shading
        # ================================================================================
        with timeblock(enabled=measure_time) as t_shade:
            render_img = shader(views["camera"], views["seq_idx"], rast_buffers, env_light=env_light, env_rot=env_rot, render_settings=render_settings)
            alpha = rast_buffers["rend_alpha"]

        # At this point, render_img is a HDR image (linear)
        # The background and image are tonemapped before blending to prevent the HDR background from bleeding through
        # Note: this is technically incorrect because tonemapping is non-linear
        # However, compositing in HDR has the background bleed through the not-quite-opaque Gaussians, creating artifacts

        # Prepare the background (environment map or constant color)
        if env_light is not None and render_settings.use_env_background:
            cam_transforms = torch.stack([cam.world_view_transform for cam in views["camera"]])
            env_rot_bg = cam_transforms if env_rot is None else torch.bmm(-env_rot, cam_transforms)
            bg = get_env_light_background(env_light, env_rot_bg, target_size=(H, W))
            if not isinstance(env_light, list):
                bg = bg.unsqueeze(0).repeat(B, 1, 1, 1)
            bg = self.display_transform(bg)
        else:
            bg = background_color.unsqueeze(0).unsqueeze(0).unsqueeze(0)#.expand(B, H, W, -1)

        blend = lambda img, alpha: blend_img(img, bg, alpha)

        # Blend the render with the background color
        render_img = blend(self.display_transform(render_img), alpha)
        if "render_param_albedo" in rast_buffers:
            rast_buffers["render_param_albedo"] = blend(self.display_transform(rast_buffers["render_param_albedo"]), alpha)

        output = AvatarOutput(render=render_img, deformed_mesh_verts=mesh_verts, texture_material=texture_material, texture_normals=texture_normals,
                              position=pos, opacity=opacity, scaling=scaling, rotation=rot, rast_buffers=rast_buffers, background=bg)

        get_vis = lambda *keys: self._make_visualizations(keys, views, output, env_light, env_rot, render_settings, blend)

        out = (output, get_vis)
        if measure_time:
            times = {k: t.get_s() for k,t in [("geometry", t_geometry), ("juvst", t_juvst), ("splat", t_splat), ("shade", t_shade)]}
            out = *out, times
        return out

    @torch.no_grad()
    def _make_visualizations(self, keys: List[str], views, output: AvatarOutput, env_light: EnvironmentLight | None, env_rot: Tensor | None,
                             render_settings: RenderSettings, blend: Callable[[Tensor, Tensor], Tensor]):
        args, device, shader = self.args, self.device, self.shader
        rast_buffers = output.rast_buffers
        B, H, W, _ = output.render.shape

        alpha = rast_buffers["rend_alpha"]
        vis = dict()

        vis["render"] = output.render
        if "render_param_albedo" in rast_buffers:
            vis["render_param_albedo"] = rast_buffers["render_param_albedo"]

        if "normals" in keys:
            vis["normals"] = blend(0.5 * (rast_buffers["rend_normal"] + 1), alpha)
         
        if "normals" in keys and "rend_normal_withoutmap" in rast_buffers:
            vis["normals_withoutmap"] = blend(0.5 * (rast_buffers["rend_normal_withoutmap"] + 1), alpha)

        if "normals_depth" in keys:
            vis["normals_depth"] = blend(0.5 * (rast_buffers["surf_normal"] + 1), alpha)

        if "depth" in keys:
            depth: Tensor = rast_buffers["depth"]
            rast_mask = rast_buffers["rend_alpha"] > 1e-3
            depth = torch.stack([torch.zeros_like(d) if r.count_nonzero() == 0 else
                                (d - d[r].min()) / (d[r].max() - d[r].min())
                                for d,r in zip(depth, rast_mask)])
            vis["depth"] = depth.repeat(1, 1, 1, 3)

        if "mesh_normals" in keys:
            mesh_normals = self.get_rasterized_mesh(views, smooth_normals=True)["smooth_normals"]
            mask = (mesh_normals.abs().sum(-1, keepdim=True) > 0).float()
            vis["mesh_normals"] = blend(0.5 * (mesh_normals + 1), mask)

        if "mesh_normals_flat" in keys:
            flat_normals = self.get_rasterized_mesh(views, flat_normals=True)["flat_normals"]
            mask = (flat_normals.abs().sum(-1, keepdim=True) > 0).float()
            vis["mesh_normals_flat"] = blend(0.5 * (flat_normals + 1), mask)

        if "material" in keys:
            if args.shader_type in ["primitive_pbr", "texture_pbr"]:
                # Albedo needs conversion to sRGB (no tonemapping as it is already [0,1]-bounded)
                albedo = rast_buffers["material"][..., :3]
                v = torch.cat((self.albedo_display_transform(albedo), rast_buffers["material"][..., 3:5]), dim=-1)
                if rast_buffers["material"].shape[-1] == 8:
                    param_albedo = rast_buffers["material"][..., 5:8]
                    v = torch.cat((v, self.albedo_display_transform(param_albedo)), dim=-1)
                vis["material"] = blend(v, alpha)

        if "shading_sphere" in keys:
            # Shading visualized on a sphere
            if not hasattr(self, "sphere"):
                from pytorch3d.utils import ico_sphere
                sphere_mesh_p3d = ico_sphere(4)
                sphere_verts = sphere_mesh_p3d.verts_list()[0].to(device) # (V, 3)
                sphere_faces = sphere_mesh_p3d.faces_list()[0].to(device) # (F, 3)
                sphere_normals = sphere_verts / sphere_verts.pow(2).sum(-1, keepdim=True).sqrt()

                # Center the sphere in the same position as the head mesh
                base_verts = self.deformer.get_mesh_verts(views["flame_pose"], views["flame_expression"], self.shape_param, self.gaussians, self.flame_scale)
                aabb = AABB(base_verts.view(-1, 3))
                sphere_verts = sphere_verts * aabb.longest_extent * 0.3 + aabb.center

                self.sphere = (sphere_verts, sphere_faces, sphere_normals)
            sphere_verts, sphere_faces, sphere_normals = self.sphere
            verts_ndc = views["camera"][0].world_to_ndc(sphere_verts).unsqueeze(0) # (1, V, 3)
            faces = sphere_faces.unsqueeze(0) # (1, F, 3)
            # Rasterize the sphere
            rasterizer = Pytorch3dRasterizer()
            rast = rasterizer(verts_ndc, faces, H, W, attributes=vertices_to_face(torch.cat((sphere_normals, sphere_verts), dim=-1).unsqueeze(0), faces)).repeat(B, 1, 1, 1)
            rast_normals, rast_pos, rast_mask, rast_z = rast[..., :3], rast[..., 3:6], rast[..., 6:7], rast[..., 7:8]
            sphere_rast_buffers = DotDict({
                "rend_alpha": torch.ones((B, H, W, 1), dtype=torch.float, device=device),
                "shading_normals": rast_normals,
            })
            # Shade
            if args.shader_type in ["primitive_pbr", "texture_pbr"]:
                attr_dim = 8 if args.shader_parametric_albedo else 5
                attr = torch.ones((B, H, W, attr_dim), dtype=torch.float, device=device)
                attr[..., 3] = 0.5 # roughness
                sphere_rast_buffers["attr"] = attr
                vis["shading_sphere"] = shader(views["camera"], views["seq_idx"], sphere_rast_buffers,
                                               env_light=env_light, env_rot=env_rot, render_settings=render_settings)
            else:
                raise NotImplementedError()
            vis["shading_sphere"] = blend(self.display_transform(vis["shading_sphere"]), rast_mask)

        if "textures" in keys:
            # Concatenate textures in a single row for visualization
            if output.texture_material is not None:
                tex_mat, tex_nrm = output.texture_material, output.texture_normals
                vis["textures"] = torch.cat((
                    self.albedo_display_transform(tex_mat[..., :3]), tex_mat[..., 3:4].repeat(1,1,3), tex_mat[..., 4:5].repeat(1,1,3),
                    self.albedo_display_transform(tex_mat[..., 5:8]) if tex_mat.shape[-1] == 8 else None,
                    (tex_nrm + 1) / 2 if tex_nrm is not None else None,
                ), dim=1)

        return vis

    def compute_flame_attrs(self, view, is_train: bool):
        poses, exprs = (self.pose_train, self.expr_train) if is_train else (self.pose_test, self.expr_test)
        idx = view["idx"]
        if self.args.detached:
            # In detached mode, we don't have access to the dataset to predict FLAME parameters on the image.
            # Instead, we rely on values that were loaded from the checkpoint. 
            pose, expr = poses[idx], exprs[idx]
        elif self.args.learn_tracking == "optim":
            pose, expr = poses[idx], exprs[idx]
        elif self.args.learn_tracking in ["smirk"]:
            pose, expr = self.tracker(view, is_train)
            # Save train parameters so they will be included in the checkpoint for use with --detached (without requiring the dataset).
            # Although the parameters will not be up-to-date with the final state of the fine-tuned tracker, the difference should be negligeable.
            if is_train:
                poses[idx], exprs[idx] = pose.detach(), expr.detach()
        else:
            raise NotImplementedError("No pose/expr available")
        return pose, expr

    def get_rasterized_mesh(self, views, smooth_normals=False, flat_normals=False):
        B = views["img"].shape[0]
        H, W = views["camera"][0].image_height, views["camera"][0].image_width
        rasterizer = Pytorch3dRasterizer()
        faces = self.flame.faces.unsqueeze(0).repeat(B,1,1) # (B, F, 3)
        deformed_pos_mesh = self.deformer.get_mesh_verts(views["flame_pose"], views["flame_expression"], self.shape_param, self.gaussians, self.flame_scale)

        mesh_pos_ndc = torch.stack([cam.world_to_ndc(x) for x, cam in zip(deformed_pos_mesh, views["camera"])]) # (B, V, 3)
        
        if flat_normals or smooth_normals:
            mesh_normals = [compute_vertices_face_normals(x, faces[0]) for x in deformed_pos_mesh] # (tensor, tensor) tuples with vertex and face normals
            vert_normals = torch.stack([x[0] for x in mesh_normals]) # (B, V, 3)
            face_normals = torch.stack([x[1] for x in mesh_normals]) # (B, F, 3)

        attr = None
        if flat_normals:
            attr = face_normals.unsqueeze(2).repeat(1,1,3,1) # (B, F, 3, 3)
        if smooth_normals:
            attr_ = vertices_to_face(vert_normals, faces) # (B, F, 3, 3)
            attr = attr_ if attr is None else torch.cat((attr, attr_), -1)

        render, pix_to_tri, pix_to_bary = rasterizer(mesh_pos_ndc, faces, H, W, attributes=attr, advanced=True)

        return {
            "flat_normals": render[...,0:3] if flat_normals else None,
            "smooth_normals": (render[...,3:6] if flat_normals else render[...,0:3]) if smooth_normals else None,
            "pix_to_tri": pix_to_tri,
            "pix_to_bary": pix_to_bary,
        }

    def save(self, checkpoint_name: str):
        logging.info(f"Saving checkpoint {checkpoint_name}")

        if self.args.detached:
            raise RuntimeError("Cannot save checkpoint in detached mode.")

        # If poses and expressions are predicted by a model, include predictions for all test frames in the checkpoint
        # Note that for the train set, we are saving the values from the last time we encountered each view
        if self.args.learn_tracking in ["smirk"]:
            with torch.no_grad():
                dataset = self.dataset_test
                loader = DeviceDataLoader(dataset, device=self.device, batch_size=1, collate_fn=dataset.collate, num_workers=4)
                for views in tqdm(loader, total=len(loader), desc="Computing FLAME parameters for test frames before saving"):
                    pose, expr = self.compute_flame_attrs(views, False)
                    self.pose_test[views["idx"]] = pose
                    self.expr_test[views["idx"]] = expr

        path = self.checkpoints_dir / f"{checkpoint_name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "args": self.args,
            "deformer": self.deformer.capture(),
            "shader": self.shader.capture(),
            "gaussians": self.gaussians.capture(),
            "pose_train": self.pose_train,
            "expr_train": self.expr_train,
            "pose_test": self.pose_test,
            "expr_test": self.expr_test,
            "tracker": self.tracker.capture() if self.args.learn_tracking in ["smirk"] else None,
            "flame_scale": self.flame_scale,
            "shape_param": self.shape_param,
        }, path)

    def restore(self, checkpoint: Path | dict):
        if isinstance(checkpoint, dict):
            state = checkpoint
        else:
            logging.info(f"Loading checkpoint {checkpoint.name}")
            state = torch.load(checkpoint, map_location="cpu")

        # Manually remap to the current device
        state = to_device_recursive(state, self.device)

        self.deformer.restore(state["deformer"])
        self.shader.restore(state["shader"])
        self.gaussians.restore(state["gaussians"])
        self.pose_train = state["pose_train"]
        self.expr_train = state["expr_train"]
        self.pose_test = state["pose_test"]
        self.expr_test = state["expr_test"]
        if self.args.learn_tracking in ["smirk"] and not self.args.detached:
            self.tracker.restore(state["tracker"])
        self.flame_scale = torch.nn.Parameter(state["flame_scale"])
        self.shape_param = torch.nn.Parameter(state["shape_param"])

    def resume(self):
        args = self.args
        if args.detached:
            # The checkpoint was already loaded during argument parsing for restoring the args, so we reuse it here
            from avatar.arguments import detached_ckpt
            self.restore(detached_ckpt)
            # Since detached mode has no datasets, create dummy datasets with saved FLAME parameters
            self.dataset_train = DummyDataset(self.pose_train, self.expr_train)
            self.dataset_test = DummyDataset(self.pose_test, self.expr_test)
        elif args.resume_epoch is not None and args.resume_epoch > 0:
            self.restore(self.checkpoints_dir / f"epoch_{args.resume_epoch:02d}.pt")
        elif args.resume_iter > 0:
            self.restore(self.checkpoints_dir / f"iter_{args.resume_iter:04d}.pt")


def compute_jacobians(
    args,
    flame: FLAME,
    gaussians: GaussianModel,
    pos: Tensor, rot: Tensor, scaling: Tensor, verts: Tensor,
    train_iter: int,
) -> Tensor:
    B = pos.shape[0]
    device = pos.device
    
    # Use this for constant color gaussians
    if train_iter < args.j_uv_st_warmup:
        return torch.zeros((B, pos.shape[1], 2, 2), dtype=torch.float32, device=device)
    
    mesh_faces, mesh_uvs, mesh_uvfaces = flame.faces, gaussians.flame_uvs, flame.textures_idx
    triangle_idx, _ = gaussians.get_binding()
    verts_idx_per_gaussian = mesh_faces[triangle_idx] # (n, 3)
    uvs_idx_per_gaussian = mesh_uvfaces[triangle_idx] # (n, 3)

    J_uv_st_all = []
    for i in range(B):        
        L = build_scaling_rotation(scaling[i], rot[i]) # (n, 3, 3)
        v1, v2, v3 = verts[i][verts_idx_per_gaussian].unbind(-2)
        uv1, uv2, uv3 = mesh_uvs[uvs_idx_per_gaussian].unbind(-2) # (n,2)
        J_uv = torch.stack((uv2-uv1, uv3-uv1), dim=-1) # (n,2,2)
        J_tri = torch.stack((v2-v1, v3-v1), dim=-1) # (n,3,2)
        J_st = L[:, :, :2] # (n,3,2)

        use_pinv = False # if True, uses pseudo-inverse (pinv) instead of least-squares (lstsq)
        if use_pinv:
            v1m, v2m, v3m = verts[i][mesh_faces].unbind(-2)
            J_tri_mesh = torch.stack((v2m-v1m, v3m-v1m), dim=-1) # (n,3,2)
            # Do the inversion once per triangle, then take the matrix for each gaussian
            J_tri_pinv = torch.linalg.pinv(J_tri_mesh)[triangle_idx] # (n,2,3)
            J_uv_st = J_uv @ (J_tri_pinv @ J_st) # (n,2,2)
        else:
            # It is always preferred to use lstsq() when possible, as it is faster and more numerically stable than computing the pseudoinverse explicitly.
            # torch.linalg.lstsq(A, B).solution == A.pinv() @ B
            J_tri_pinv_times_J_st = torch.linalg.lstsq(J_tri, J_st).solution
            J_uv_st = J_uv @ J_tri_pinv_times_J_st # (n,2,2)

        J_uv_st = J_uv_st.permute(0,2,1) # transpose
        J_uv_st_all.append(J_uv_st)

    return torch.stack(J_uv_st_all)
