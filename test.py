import os
import logging
from pathlib import Path

import torch
from argparse import BooleanOptionalAction

from avatar import Avatar, RenderSettings, create_parser, parse_args
from dataset import DeviceDataLoader, to_device_recursive
from utils.metrics import img_psnr, img_ssim, lpips
from utils.logging import setup_logging
from utils.visualization import save_img_columns
from utils.tqdm import tqdm

@torch.no_grad()
def test(args, avatar: Avatar, out_dir: Path):
    device = avatar.device

    avatar.resume()

    metrics_names = {"psnr": "PSNR", "lpips": "LPIPS", "ssim": "SSIM"}
    metrics = {k: dict() for k in metrics_names}

    dataset = avatar.dataset_train if args.train_set else avatar.dataset_test
    dataloader = DeviceDataLoader(dataset, device=device, batch_size=1, collate_fn=dataset.collate, num_workers=4)

    render_settings = RenderSettings(hw_textures=args.hw_tex, eval_mode=True)

    # Warm-up for more accurate time measurements
    view = to_device_recursive(dataset.collate([dataset[0]]), device)
    for _ in range(5):
        avatar.run(view, render_settings=render_settings)

    logging.info("Starting evaluation")
    for k, views in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Retrieve cameras, pose and expression for these frames
        views["flame_pose"], views["flame_expression"] = avatar.compute_flame_attrs(views, is_train=False)
        # Render
        with torch.no_grad():
            output, get_vis, times = avatar.run(views, render_settings=render_settings, measure_time=True)

            if args.visualize:
                vis = get_vis("normals", "material", "mesh_normals", "textures")

                save_img_columns([
                    views["img"],
                    vis["render"],
                    vis["normals"],
                    # vis["material"][..., :3], vis["material"][..., 3:4], vis["material"][..., 4:5],
                    # vis["mesh_normals"]
                ], out_dir / f"{views['idx'][0]:04d}.png")

                if k == 0 and "textures" in vis:
                    save_img_columns([vis["textures"]], out_dir / "texture.png")

        for i, vidx in enumerate(views["idx"]):
            render = output.render[i].unsqueeze(0)
            gt = views["img"][i].unsqueeze(0)

            # The difference of doing this is negligeable, but it aligns with methods that save renders as 8-bit images to disk first
            render = (render * 255).round().clamp(min=0,max=255) / 255

            metrics["psnr"][vidx.item()] = img_psnr(render, gt).item()
            metrics["ssim"][vidx.item()] = img_ssim(render, gt).item()
            metrics["lpips"][vidx.item()] = lpips(render, gt).item()

            # Inference time
            assert views["idx"].shape[0] == 1 # ensure we're not batching for measuring time
            for key, val in times.items():
                key = f"time_{key}"
                if key not in metrics:
                    metrics[key] = dict()
                metrics[key][vidx.item()] = val

    logging.info("Done!")

    metrics = {k: torch.tensor(list(v.values()), dtype=torch.float) for k, v in metrics.items()}

    text = "Metrics:"
    for key in metrics:
        if key in metrics_names:
            text += f"\n{metrics_names[key]}: {metrics[key].mean():.8f} ± {metrics[key].std():.8f}"
    text += f"\nGaussians: {avatar.gaussians.triangle_idx.shape[0]}"
    text += "\nTiming:"
    for key in metrics:
        if key.startswith("time_"):
            text += f"\nTime ({key[5:]}): {1000 * metrics[key].mean():.2f} ± {1000 * metrics[key].std():.2f} ms"
    text += f"\nFPS (static): {1 / (metrics['time_splat'] + metrics['time_shade']).mean():.2f}"
    text += f"\nFPS (dynamic): {1 / (metrics['time_geometry'] + metrics['time_juvst'] + metrics['time_splat'] + metrics['time_shade']).mean():.2f}"

    with open(out_dir / "metrics.txt", "w+") as f:
        f.write(text)
    logging.info(text)

    if args.visualize:
        logging.info("Creating video")
        fps = args.source_fps or 24
        os.system(f"/usr/bin/ffmpeg -y -framerate {fps} -pattern_type sequence -i '{out_dir}/%04d.png' -c:v libx264 -pix_fmt yuv420p {out_dir / 'video.mp4'}")

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument("--train_set", action=BooleanOptionalAction, default=False, help="If true, evaluates on the train set instead of the test set")
    parser.add_argument("--visualize", action=BooleanOptionalAction, default=False, help="Save visualizations of the rendered images during testing")
    parser.add_argument("--hw_tex", action=BooleanOptionalAction, default=True, help="Use hardware-accelerated textures during rendering")
    args = parse_args(parser)

    avatar = Avatar(args)

    iter_str = f"epoch_{args.resume_epoch:02d}" if args.resume_epoch else f"{args.resume_iter:04d}"
    out_dir: Path = avatar.experiment_dir / f"{'train' if args.train_set else 'test'}_{iter_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(out_dir / "log.txt")
    avatar.init_modules()
    test(args, avatar, out_dir)
