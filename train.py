import os
import logging
import math
from time import time
from pathlib import Path

import torch
from torch.optim import Adam

from avatar import Avatar, create_parser, parse_args
from avatar.losses import losses
from avatar.densification import do_clone, do_split, do_prune
from dataset import DeviceDataLoader, DatasetCache, to_device_recursive
from utils.logging import setup_logging
from utils.visualization import save_img_columns
from utils.tqdm import tqdm

# This tends to fix "too many open files" errors occurring during training
# Alternatively, set num_workers to 0 in the DataLoader (slows down training)
if False:
    torch.multiprocessing.set_sharing_strategy("file_system")


def train(avatar: Avatar, out_dir: Path):
    args = avatar.args

    avatar.resume()
    avatar.gaussians.training_setup(args)
    device, dataset_train, gaussians, shader = avatar.device, avatar.dataset_train, avatar.gaussians, avatar.shader

    images_dir = out_dir
    images_dir.mkdir(parents=True, exist_ok=True)

    tqdm_refresh_interval = os.getenv("TQDM_TRAIN_REFRESH_INT", None)
    tqdm_refresh_interval = int(tqdm_refresh_interval) if tqdm_refresh_interval is not None else 1

    ###################### Data ######################

    # Pick some views for visualizations during training
    n_debug_img = min(5, len(dataset_train))
    view_indices = args.visualization_views or range(0, len(dataset_train), len(dataset_train) // n_debug_img)
    debug_views = [dataset_train[idx] for idx in view_indices]
    debug_views = to_device_recursive(dataset_train.collate(debug_views), device)

    num_workers = 4
    
    if args.train_views_whitelist:
        # Train on specific views for debugging
        dataset_train = torch.utils.data.Subset(dataset_train, args.train_views_whitelist)
    
    if args.cache:
        dataset_train = DatasetCache(dataset_train)
        # With caching, each worker would hold its own copy of the dataset in memory
        num_workers = 0
    
    dataloader_train = DeviceDataLoader(dataset_train, device=device, batch_size=args.batch_size,
                                        collate_fn=dataset_train.collate, shuffle=True, drop_last=False, num_workers=num_workers)

  
    ##################### Losses #####################

    loss_functions = {key: LossClass(avatar,
                                     weight=getattr(args, f"loss_{key}_weight"),
                                     start_iter=getattr(args, f"loss_{key}_start"),
                                     half_life=getattr(args, f"loss_{key}_halflife"),
                                ).to(device)for key, LossClass in losses.items()}

    ################### Optimizers ###################

    opt_params = []

    if args.shader_type == "primitive_pbr":
        opt_params += [
            {"params": list(shader.light_env.parameters()), "lr": args.learn_shader_light_lr, "name": "shader_light"}, 
            {"params": shader._material, "lr": args.learn_shader_material_lr, "name": "shader_material"},
        ]
    elif args.shader_type == "texture_pbr":
        opt_params += [
            {"params": list(shader.light_env.parameters()), "lr": args.learn_shader_light_lr, "name": "shader_light"}, 
            {"params": shader._material_alb, "lr": args.learn_shader_material_lr, "name": "shader_material_alb"},
            {"params": shader._material_r_spec, "lr": args.learn_shader_material_lr, "name": "shader_material_r_spec"},
        ]
        opt_params.append({"params": shader.texture_normals, "lr": args.learn_texture_normals_lr, "name": "shader_texture_normals"})
    else:
        raise NotImplementedError()
    if args.shader_parametric_albedo:
        opt_params.append({"params": list(shader.parametric_albedo.parameters()), "lr": args.learn_shader_parametric_albedo_lr, "name": "shader_parametric_albedo"})

    if args.learn_tracking == "smirk":
        tracker = avatar.tracker
        opt_params += [
            {"params": tracker.get_tune_parameters(), "lr": args.learn_expr_lr, "name": "smirk"},
        ]
    elif args.learn_tracking == "optim":
        pose_train, expr_train = torch.nn.Parameter(avatar.pose_train), torch.nn.Parameter(avatar.expr_train)   
        opt_params += [
            {"params": pose_train, "lr": args.learn_pose_lr, "name": "pose"},
            {"params": expr_train, "lr": args.learn_expr_lr, "name": "expr"}
        ]
    else:
        logging.info("Using fixed tracking parameters")

    opt_params += [
        {"params": [avatar.flame_scale], "lr": args.learn_flame_scale_lr, "name":"flame_scale"},
        {"params": [avatar.shape_param], "lr": args.learn_shape_param_lr, "name":"shape_param"},
    ]

    optimizer_gaussians: Adam = gaussians.optimizer
    for param_group in opt_params:
        optimizer_gaussians.add_param_group(param_group)

    # ==============================================================================================
    # T R A I N I N G
    # ==============================================================================================
    if args.epochs:
        # Train for the given number of epochs
        epochs = args.epochs
        iterations = epochs * len(dataloader_train)
    else:
        # Train for the given number of iterations
        iterations = args.iterations
        epochs = math.ceil(iterations / len(dataloader_train))
    
    if args.resume_epoch is not None:
        resume_iter = args.resume_epoch * len(dataloader_train)
    else:
        resume_iter = args.resume_iter
    iteration = resume_iter
    last_iteration = resume_iter + iterations

    print("=="*50)
    logging.info(f"Training from iteration {iteration+1} to {last_iteration} (1 epoch = {len(dataloader_train)} iters)")
    print("=="*50)

    start_epoch = resume_iter // len(dataloader_train)
    progress_bar = tqdm(range(start_epoch, start_epoch+epochs))
    start = time()
    for epoch in progress_bar:
        if not args.train_views_whitelist:
            logging.info(f"Beginning epoch {epoch+1} ({args.run_name})")

        if args.last_epoch_lr_factor != 1 and epoch == start_epoch + epochs - 1:
            logging.info(f"Last epoch - multiplying all learning rates by {args.last_epoch_lr_factor}")
            for param_group in optimizer_gaussians.param_groups:
                param_group["lr"] *= args.last_epoch_lr_factor

        for views in dataloader_train:
            if iteration >= last_iteration:
                break
            iteration += 1

            tqdm_refresh = iteration % tqdm_refresh_interval == 0
            progress_bar.set_description(desc=f"Epoch {epoch+1}, Iter {iteration}", refresh=tqdm_refresh)

            is_visualize_iter = args.visualize_frequency > 0 and (iteration == resume_iter+1 or iteration % args.visualize_frequency == 0 or (iteration <= resume_iter+1000 and iteration % 100 == 0))
            is_save_iter = (args.save_frequency > 0 and iteration % args.save_frequency == 0) or (args.iterations and iteration == last_iteration)

            # Retrieve cameras, pose and expression for these frames
            views["flame_pose"], views["flame_expression"] = avatar.compute_flame_attrs(views, is_train=True)

            shader.update_env_lights(views["seq_idx"])

            # Render                       
            output, _ = avatar.run(views, train_iter=iteration)

            loss = 0.0
            for key, loss_fn in loss_functions.items():
                if iteration > loss_fn.start_iter and loss_fn.weight > 0:
                    w = loss_fn.weight
                    if loss_fn.half_life > 0:
                        w = loss_fn.weight * pow(1/2, (iteration - loss_fn.start_iter) / loss_fn.half_life)
                    l = loss_fn(avatar, views, output) * w
                    if torch.isnan(l).any():
                        logging.warning(f"Loss function {key} returned NaN!")
                        exit(1)
                    else:
                        loss += l

            progress_bar.set_postfix({"loss": loss.item(), "n": gaussians.n_gaussians}, refresh=tqdm_refresh)

            loss.backward()
            optimizer_gaussians.step()
            optimizer_gaussians.zero_grad()

            if is_visualize_iter:
                with torch.no_grad():
                    # Retrieve cameras, pose and expression for these frames
                    debug_views["flame_pose"], debug_views["flame_expression"] = avatar.compute_flame_attrs(debug_views, is_train=True)
                    shader.update_env_lights(debug_views["seq_idx"])
                    out_debug, get_vis = avatar.run(debug_views, train_iter=iteration)
                    vis = get_vis("material", "normals", "normals_depth", "mesh_normals", "shading_sphere", "textures")

                    save_img_columns([
                        debug_views["img"],
                        vis.get("render_param_albedo", None),
                        vis["render"],
                        vis["shading_sphere"],
                        vis["material"][..., :3], vis["material"][..., 3:4], vis["material"][..., 4:5],
                        vis["normals"],
                        # vis.get("normals_withoutmap", None),
                        vis["normals_depth"],
                        # vis["depth"],
                        # vis["mesh_normals"]
                    ], images_dir / f"iter_{iteration:04d}.png")

                    if "textures" in vis:
                        save_img_columns([vis["textures"]], images_dir / f"iter_{iteration:04d}_texture.png")

            if is_save_iter:
                avatar.save(f"iter_{iteration:04d}")

            # Densification
            if iteration < args.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                for radii, screenspace_points in zip(output.rast_buffers["radii"], output.rast_buffers["screenspace_points"]):
                    visibility_filter = radii > 0
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(screenspace_points, visibility_filter)

                if iteration > args.densify_from_iter and iteration % args.densification_interval == 0:            
                    size_threshold = 20 if iteration > args.opacity_reset_interval else None
                    min_opacity = 0.005
                    gaussians.densify_and_prune(args.densify_grad_threshold, min_opacity, size_threshold,
                                                do_clone=lambda selected_pts_mask: do_clone(avatar, selected_pts_mask),
                                                do_split=lambda selected_pts_mask: do_split(avatar, selected_pts_mask),
                                                do_prune=lambda prune_mask: do_prune(avatar, prune_mask))

                if iteration % args.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
        
        if (args.save_epoch_frequency > 0 and (epoch + 1) % args.save_epoch_frequency == 0) or epoch == start_epoch + epochs - 1:
            avatar.save(f"epoch_{epoch+1:02d}")

    end = time()
    logging.info(f"Training finished. Total time: {round((end-start)/60)} minutes.")


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)

    if not args.iterations and not args.epochs:
        raise ValueError("Please specify --iterations or --epochs.")
    if args.iterations and args.epochs:
        raise ValueError("Please use --iterations or --epochs, not both.")

    out_dir: Path = args.output_dir / args.run_name / "train"
    if args.train_subdir:
        out_dir = out_dir / args.train_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir / "log.txt")

    avatar = Avatar(args)
    avatar.init_modules()

    train(avatar, out_dir)
