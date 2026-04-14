import math
from typing import Iterable
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
import imageio


def blend_img(img: Tensor, background: Tensor, alpha: Tensor) -> Tensor:
    # Match the number of channels in the given image
    if background.shape[-1] < img.shape[-1]:
        background = torch.cat((background, background[...,:1] * torch.ones_like(background[...,:1]).repeat(1, 1, 1, img.shape[-1] - background.shape[-1])), -1)
    elif background.shape[-1] > img.shape[-1]:
        background = background[..., :img.shape[-1]]
    # Blend
    return torch.lerp(background, img, alpha)

def tonemapping(x: Tensor) -> Tensor:
    # Simple Reinhard tone mapping
    return x / (1.0 + x)

def linear_to_srgb(x: Tensor) -> Tensor:
    """
    Convert linear RGB to sRGB using the IEC 61966-2-1 piecewise transfer function.
    Input and output are in [0, 1]. Works on any tensor shape.
    """
    return torch.where(
        x <= 0.0031308,
        x * 12.92,
        1.055 * x.clamp(min=0.0031308) ** (1.0 / 2.4) - 0.055
        #           ^ clamping is important to avoid infinite gradients
    )

def srgb_to_linear(x: Tensor) -> Tensor:
    x = x.clamp(min=0.0)
    return torch.where(
        x <= 0.04045,
        x / 12.92,
        ((x + 0.055) / 1.055).clamp(min=0.0) ** 2.4
    )

def convert_uint(x: torch.Tensor):
    return np.clip(np.rint(x.detach().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)

def save_img(out_path: Path | str, img: Tensor):
    # Quick method for saving an image to disk. img shape; (H,W,C)
    if img.shape[-1] == 1:
        img = img.expand(*img.shape[:-1], 3)
    img = convert_uint(img)
    imageio.imsave(out_path, img)

def save_img_columns(columns: list[Tensor | None], path: str):
    """
    Save a list of image columns (tensors of shape ([B,] H, W, C)) as a single image with the columns concatenated horizontally.
    Widths can be different but heights have to be the same. Channels (C) can be 1, 2 or 3.
    """

    # Filter-out None columns
    columns = [x for x in columns if x is not None]

    if len(columns) == 0:
        raise ValueError("save_img_columns: no columns to save")
    
    output_img = None # (B, H, W, 3)
    add_column = lambda img: torch.cat((output_img, img), dim=-2)
    for i, x in enumerate(columns):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        elif x.ndim != 4:
            raise ValueError(f"save_img_columns: invalid number of dimensions in tensor {i} ({x.ndim})")
        
        if x.shape[-1] == 1:
            x = x.repeat(1,1,1,3)
        elif x.shape[-1] == 2:
            x = torch.cat((x, torch.zeros_like(x[..., :1])), dim=-1)
        elif x.shape[-1] != 3:
            raise ValueError(f"save_img_columns: invalid number of channels in tensor {i} ({x.shape[-1]})")

        if i > 0:
            if x.shape[0] != output_img.shape[0]:
                raise ValueError(f"save_img_columns: batch size mismatch in tensor {i} ({x.shape[0]} vs {output_img.shape[0]})")
            if x.shape[1] != output_img.shape[1]:
                raise ValueError(f"save_img_columns: height mismatch in tensor {i} ({x.shape[1]} vs {output_img.shape[1]})")

        # x is now (B, H, W, 3)
        output_img = x if i == 0 else add_column(x)

    # Concatenate batch dimension into rows
    output_img = torch.cat([img for img in output_img], dim=0) # (H*B, W*len(columns), 3)
    # Write the final image
    save_img(path, output_img)

def arrange_grid(imgs: Iterable[torch.Tensor], cols: int) -> torch.Tensor:
    rows = math.ceil(len(imgs) / cols)
    if len(imgs) < cols * rows:
        imgs += [torch.zeros_like(imgs[0]) for _ in range(cols*rows-len(imgs))]
    assert len(imgs) == cols * rows
    rows = [torch.cat(tuple(imgs[y*cols:(y+1)*cols]), dim=1) for y in range(rows)]
    return torch.cat(rows, dim=0)
