import logging
from pathlib import Path
from argparse import BooleanOptionalAction

import torch

from avatar import Avatar, RenderSettings, create_parser, parse_args
from avatar.environment_light import load_envmap
from dataset import DeviceDataLoader
from utils.logging import setup_logging
from utils.visualization import save_img, linear_to_srgb
from utils.tqdm import tqdm

def render_set(args, avatar: Avatar, dataset, out_dir: Path, is_train_dataset: bool, render_settings: RenderSettings):
    device = avatar.device
    
    subset = None
    if is_train_dataset:
        if args.train_frames is not None:
            indices = [i for i in args.train_frames if i < len(dataset)]
            subset = torch.utils.data.Subset(dataset, indices)
            logging.info(f"Rendering only {len(subset)} train frames as specified")
    else:
        if args.test_frames is not None:
            indices = [i for i in args.test_frames if i < len(dataset)]
            subset = torch.utils.data.Subset(dataset, indices)
            logging.info(f"Rendering only {len(subset)} test frames as specified")

    dataloader = DeviceDataLoader(dataset if subset is None else subset, device=device, batch_size=1, collate_fn=dataset.collate, num_workers=4)

    out_dir_gt = out_dir / "gt"
    out_dir_gt.mkdir(parents=True, exist_ok=True)
    out_dir_render = out_dir / "render"
    out_dir_render.mkdir(parents=True, exist_ok=True)
    if args.albedo:
        out_dir_albedo = out_dir / "albedo"
        out_dir_albedo.mkdir(parents=True, exist_ok=True)
    if args.roughness:
        out_dir_roughness = out_dir / "roughness"
        out_dir_roughness.mkdir(parents=True, exist_ok=True)
    if args.specular:
        out_dir_specular = out_dir / "specular"
        out_dir_specular.mkdir(parents=True, exist_ok=True)
    if args.normals:
        out_dir_normals = out_dir / "normals"
        out_dir_normals.mkdir(parents=True, exist_ok=True)

    for views in tqdm(dataloader):
        idx = views["idx"][0]
        if idx % args.interval != 0:
            continue
        views["flame_pose"], views["flame_expression"] = avatar.compute_flame_attrs(views, is_train=is_train_dataset)
        name =  f"{idx:04d}.png"
        _, get_vis = avatar.run(views, render_settings=render_settings)

        required_vis = ["render"]
        if args.albedo or args.roughness or args.specular: required_vis.append("material")
        if args.normals: required_vis.append("normals")
        vis = get_vis(*required_vis)

        save_img(out_dir_gt / name, views["img"].squeeze(0))
        save_img(out_dir_render / name, vis["render"].squeeze(0))

        if args.albedo:
            save_img(out_dir_albedo / name, vis["material"][..., :3].squeeze(0))
        if args.roughness:
            save_img(out_dir_roughness / name, vis["material"][..., 3:4].squeeze(0).repeat(1,1,3))
        if args.specular:
            save_img(out_dir_specular / name, vis["material"][..., 4:5].squeeze(0).repeat(1,1,3))

        if args.normals:
            save_img(out_dir_normals / name, vis["normals"].squeeze(0))


def render_relighting(args, avatar: Avatar, dataset, out_dir: Path, is_train_dataset: bool, env_path: Path, render_settings: RenderSettings):
    device = avatar.device
    dataloader = DeviceDataLoader(dataset, device=device, batch_size=1, collate_fn=dataset.collate, num_workers=4)
    
    out_dir.mkdir(parents=True, exist_ok=True)

    light = load_envmap(env_path, device, hravatar_compat=args.relight_hravatar_mode)
    if args.export_latlong:
        save_img(out_dir / "latlong.png", light.original_latlong)

    if args.relight_hravatar_mode:
        # In order to obtain similar-looking lighting as HRAvatar for comparisons, we need to disable tone-mapping and sRGB conversion,
        # essentially treating the HDR linear renders as the final output. This is wrong!

        # Disable tonemapping and sRGB conversion
        avatar.display_transform = avatar.albedo_display_transform = lambda x: x.clamp(0, 1)
        # Convert the albedo to sRGB  
        old_albedo_activation = avatar.shader.albedo_activation
        avatar.shader.albedo_activation = lambda x: linear_to_srgb(old_albedo_activation(x))
    
    max_steps = min(len(dataset)+1, 500)

    for views in tqdm(dataloader):
        idx = views["idx"][0]
        if idx % args.interval != 0:
            continue
        
        rotate_light_y = idx / max_steps * 2 *  torch.pi
        rot_mat = rotate_y(-rotate_light_y, device=device).unsqueeze(0)

        views["flame_pose"], views["flame_expression"] = avatar.compute_flame_attrs(views, is_train_dataset)
        name =  f"{idx:04d}.png"
        _, get_vis = avatar.run(views, env_light=light, env_rot=rot_mat, render_settings=render_settings)
        vis = get_vis("render")
        save_img(out_dir / name, vis["render"].squeeze(0))

@torch.no_grad()
def main(args, avatar: Avatar):
    avatar.resume()

    out_dir: Path = avatar.experiment_dir / "render"
    out_dir.mkdir(parents=True, exist_ok=True)

    dir_train = out_dir / "train"
    dir_test = out_dir / "test"

    render_settings = RenderSettings(background_color="white")

    if args.render_train:
        logging.info("Rendering train set")
        render_set(args, avatar, avatar.dataset_train, dir_train, True, render_settings)

    if args.render_test:
        logging.info("Rendering test set")
        render_set(args, avatar, avatar.dataset_test, dir_test, False, render_settings)

    if len(args.relight) > 0:
        render_settings = RenderSettings(background_color="black")
        envmaps = args.relight
        for i, envmap_path in enumerate(envmaps):
            envmap_path = Path(envmap_path)
            envmap_name = envmap_path.stem
            logging.info(f"Rendering relighting (test) {i+1}/{len(envmaps)}: {envmap_name}")
            render_relighting(args, avatar, avatar.dataset_test, dir_test / f"dynamic_relight-{envmap_name}", False, envmap_path, render_settings)

    # if create_video:
    #     logging.info("Creating video")
    #     fps = 24 // args.sample_idx_ratio
    #     os.system(f"/usr/bin/ffmpeg -y -framerate {fps} -pattern_type glob -i '{out_dir / '*.png'}' -c:v libx264 -pix_fmt yuv420p {out_dir / 'video.mp4'}")

    logging.info("Done")

def rotate_y(a, device=None):
    s, c = torch.sin(a), torch.cos(a)
    return torch.tensor([[ c, 0, s, 0], 
                         [ 0, 1, 0, 0], 
                         [-s, 0, c, 0], 
                         [ 0, 0, 0, 1]], dtype=torch.float32, device=device)


if __name__ == "__main__":
    parser = create_parser()
    arg = parser.add_argument
    arg("--render_train", action=BooleanOptionalAction, default=False, help="Render the train set")
    arg("--render_test", action=BooleanOptionalAction, default=False, help="Render the test set")
    
    arg("--albedo", action=BooleanOptionalAction, default=False, help="Whether to save the albedo buffers when using render_train or render_test")
    arg("--roughness", action=BooleanOptionalAction, default=False, help="Whether to save the roughness buffers when using render_train or render_test")
    arg("--specular", action=BooleanOptionalAction, default=False, help="Whether to save the specular reflectance buffers when using render_train or render_test")
    arg("--normals", action=BooleanOptionalAction, default=False, help="Whether to save the normals buffers when using render_train or render_test")

    arg("--relight", type=str, nargs="+", default=[], help="One or more envmaps to use for relighting")
    arg("--export_latlong", action=BooleanOptionalAction, default=False)
    arg("--relight_hravatar_mode", action=BooleanOptionalAction, default=False, help="Whether to use the hravatar compatible envmap loading (if true, --envmaps should be directories containing .tga files, otherwise should be .hdr files)")

    arg("--interval", type=int, default=1)
    arg("--train_frames", type=int, nargs="+", default=None)
    arg("--test_frames", type=int, nargs="+", default=None)

    args = parse_args(parser)

    setup_logging(args.output_dir / args.run_name / "log_render.txt")

    avatar = Avatar(args)
    avatar.init_modules()
    main(args, avatar)
