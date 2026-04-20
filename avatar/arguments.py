import os
from pathlib import Path

import torch
from configargparse import ArgumentParser, Namespace
from argparse import BooleanOptionalAction


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()
    arg = parser.add_argument

    ############################## META ##############################

    arg("--run_name", type=str, default=None, help="Name of this run")
    arg("--output_dir", type=Path, default="out", help="Path to the output directory")
    arg("--device", type=int, default=0, choices=([-1] + list(range(torch.cuda.device_count()))), help="Which GPU to use; -1 is CPU")
    arg("--resume_iter", type=int, default=0, help="Load checkpoint at a given iteration")
    arg("--resume_epoch", type=int, default=None, help="Load checkpoint at a given epoch")
    arg("--detached", type=Path, default=None, help="Load the given checkpoint without loading a dataset")

    ############################## DATASET ##############################

    arg("--train_dirs", type=Path, nargs="+", help="Path to the training sequences")
    arg("--test_set_num", type=int, default=350, help="Number of frames to use for the test set (and exclude from training)")
    arg("--sample_idx_ratio", type=int, default=1, help="Subsample the video for training (e.g. 2 means using every other frame)")    
    arg("--visualization_views", type=int, nargs="+", default=[], help="Indices of views to use for visualization (debugging)")
    arg("--train_views_whitelist", type=int, nargs="*", default=[], help="Restrict training to these views (debugging)")
    arg("--dataset_type", type=str, default=None, choices=["HRAvatar", "MonoFace"])
    arg("--source_fps", type=int, default=None, help="FPS of the original video (only used when re-exporting videos)")    

    ############################## MODEL ##############################

    # Geometry

    arg("--add_mouth_interior", action=BooleanOptionalAction, default=True, help="Add faces to close the mouth interior")
    arg("--add_teeth", action=BooleanOptionalAction, default=True, help="Add teeth to the template head")
    arg("--gaussians_init_opacity", type=float, default=0.99, help="Initial opacity of gaussians")
    arg("--gaussians_init_scale", type=float, default=1.0, help="Initial scale multiplier of gaussians (there is also an auto scaling factor based on the distance between points)")
    arg("--gaussians_init_count", type=float, default=8, help="Number of gaussians to initialize per mesh triangle. Note that FLAME has about 10k triangles.")

    # Rendering
    arg("--shader_type", type=str, default="texture_pbr", choices=["texture_pbr", "primitive_pbr"])
    arg("--env_resolution", type=int, default=128, help="Resolution of learned environment map")
    arg("--env_mip_levels", type=int, default=None)
    arg("--env_activation", type=str, default="relu", choices=["relu", "sigmoid", "exp", "softplus"], help="Activation function for the learned env map")
    arg("--env_multiplier", type=float, default=1.0, help="Amplitude of the learned env map (e.g. if using sigmoid activation, the final value is sigmoid(x)*mul)")
    arg("--initial_albedo", type=float, default=0.5)
    arg("--initial_roughness", type=float, default=0.5)
    arg("--initial_specular", type=float, default=0.04)
    arg("--initial_envmap_intensity", type=float, default=0.5)
    arg("--min_spec", type=float, default=0.04)
    arg("--max_spec", type=float, default=0.80)
    arg("--min_roughness", type=float, default=0.5)
    arg("--max_roughness", type=float, default=1.0)
    arg("--texture_res_albedo", type=int, default=64, help="Resolution to use for learned albedo texture with shader_type=texture_pbr")
    arg("--texture_res_r_spec", type=int, default=64, help="Resolution to use for learned roughness and specular texture with shader_type=texture_pbr")
    arg("--texture_res_normal", type=int, default=64, help="Resolution to use for learned normal texture with shader_type=texture_pbr")
    arg("--shader_parametric_albedo", type=str, default=None, choices=["BFM", "FLAME"], help="Optimize coefficients of a parametric albedo for regularizations.")

    ############################## TRAINING ##############################

    arg("--batch_size", type=int, default=1, help="Number of views used per iteration")
    arg("--iterations", type=int, default=None, help="Number of iterations to train for (mutually exclusive with --epochs)")
    arg("--epochs", type=int, default=None, help="Number of epochs to train for (mutually exclusive with --iterations)")
    arg("--save_frequency", type=int, default=0, help="Frequency of checkpoints saving (iterations)")
    arg("--save_epoch_frequency", type=int, default=0, help="Frequency of checkpoints saving (epochs).")
    arg("--visualize_frequency", type=int, default=100, help="Frequency of visualizations (iterations)")
    arg("--train_subdir", type=str, default=None, help="Use a subdirectory for training outputs (for quick experiments within the same config)")
    arg("--cache", action=BooleanOptionalAction, default=False, help="Cache the dataset in RAM for faster training (uses 20-30 GB of memory, depending on the dataset)")

    # Objective functions
    from avatar.losses import losses
    for key in losses.keys():
        arg(f"--loss_{key}_weight", type=float, default=0.0, help=f"Weight of the {key} loss")
        arg(f"--loss_{key}_start", type=int, default=0, help=f"Iteration at which to enable the {key} loss")
        arg(f"--loss_{key}_halflife", type=int, default=0, help=f"Half-life of the {key} loss (set to 0 for no decay)")
    
    arg("--loss_uv_distortion_threshold", type=float, default=1000.0, help="UV distance threshold above which to stop the UV distortion loss along a ray")

    # Learning rates
    for p in [
        # Shading
        "shader_parametric_albedo", "texture_normals", "shader_material", "shader_light",
        # FLAME
        "flame_scale", "shape_param", "flame_shape_dirs", "flame_expression_dirs", "flame_pose_dirs", "flame_lbs_weights", "flame_v_template",
        ]:
        arg(f"--learn_{p}_lr", type=float, default=0.0, help=f"Learning rate for {p}")
    arg("--last_epoch_lr_factor", type=float, default=1, help="A factor by which to multiply the learning rate of all parameters at the last epoch")
    # tracking
    arg("--learn_pose_lr", type=float, default=0.0)
    arg("--learn_expr_lr", type=float, default=0.0)
    arg("--learn_tracking", type=str, default="optim", choices=["optim", "smirk"])
    # Gaussian parameters
    arg("--learn_disp_lr", type=float, default=0.0, help="Learning rate for Gaussian displacements along triangle normals")
    arg("--learn_opacity_lr", type=float, default=0.05, help="Learning rate for Gaussian opacities")
    arg("--learn_scaling_lr", type=float, default=0.005, help="Learning rate for Gaussian scales")
    arg("--learn_rotation_lr", type=float, default=0.001, help="Learning rate for Gaussian quaternions")
    arg("--learn_bary_lr", type=float, default=0.0, help="Learning rate for Gaussian barycentric coordinates")

    # Densification and pruning
    arg("--densification_interval", type=int, default=500, help="Frequency at which to densify/prune gaussians")
    arg("--densify_from_iter", type=int, default=1e9, help="From what iteration do we apply densification")
    arg("--densify_until_iter", type=int, default=0, help="Until what iteration do we apply densification")
    arg("--densify_grad_threshold", type=float, default=0.0002, help="Gradient threshold for densification")
    arg("--percent_dense", type=float, default=0.01)
    arg("--opacity_reset_interval", type=int, default=3000, help="Frequency at which to reset gaussian opacities")

    ############################## EXPERIMENTAL ##############################
    arg("--j_uv_st_warmup", type=int, default=0, help="Number of iterations before using Jacobians for shader_type=texture_pbr. Before this, each Gaussian will only use one texel.")
    arg("--render_with_parametric_albedo", action=BooleanOptionalAction, default=False, help="Splat the parametric albedo and use it to render a second shaded image for regularization")
    # Experimental progressive training of texture mip levels
    arg("--texture_mip", action=BooleanOptionalAction, default=False, help="Whether to use mipmaps for learning the texture")
    arg("--texture_mip_normals", action=BooleanOptionalAction, default=False, help="Whether or not to use mipmaps for the normals texture")
    arg("--texture_mip_levels", type=int, default=-1, help="Number of mip levels for the color and normal textures")
    arg("--texture_mip_max_iter", type=int, default=-1, help="Number of training iterations during which the texture mip level should ramp up")

    return parser

detached_ckpt = None

def parse_args(parser: ArgumentParser, cmd_args=None) -> Namespace:
    # Pre-parse args
    args, unknown_args = parser.parse_known_args(cmd_args)
    
    # If --detached is present, load the checkpoint and use its saved args as new defaults. Otherwise, parse normally.
    if args.detached:
        if not args.detached.exists():
            raise ValueError(f"Checkpoint {args.detached} not found")
        global detached_ckpt
        detached_ckpt = torch.load(args.detached, map_location="cpu")
        print(f"Loaded detached checkpoint from {args.detached}")
        # Set the defaults of the main parser to the values from the checkpoint
        parser.set_defaults(**vars(detached_ckpt["args"]))
    else:    
        # Add config args
        parser.add_argument("-c", "--config", is_config_file=True, help="Config file path")
        if "-i" in unknown_args or "--input" in unknown_args:
            parser.add_argument("-i", "--input", is_config_file=True, help="Config file path (overrides fields in the main config)")

    # Parse again
    args = parser.parse_args(cmd_args)

    # In detached mode: overwrite values we don't want to reuse from training
    if args.detached:
        args.output_dir = Path("./output/")
        args.train_dirs = None

    if args.run_name is None:
        args.run_name = Path(args.config).stem
        if "input" in args:
            args.run_name += f"_{Path(args.input).stem}"

    print(f"Run: '{args.run_name}'")

    if args.resume_iter and args.resume_epoch:
        raise ValueError("Please use --resume_iter or --resume_epoch, not both.")

    return args
