import torch
from torch import Tensor
import torch.nn.functional as F

import nvdiffrast.torch as dr
from nvdiffrec.render import util

class texture_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tex: Tensor):
        assert tex.ndim == 3 # (H,W,C)
        return util.avg_pool_nhwc(tex.unsqueeze(0), (2,2)).squeeze(0)

    @staticmethod
    def backward(ctx, dout: Tensor):
        assert dout.ndim == 3 # (H,W,C)
        res = dout.shape[1] * 2
        s = 1.0 / res
        gy, gx = torch.meshgrid(torch.linspace(s, 1-s, res, device="cuda"), 
                                torch.linspace(s, 1-s, res, device="cuda"),
                                indexing='ij')
        coords = torch.stack((gx, gy), dim=-1)
        out = dr.texture(dout.unsqueeze(0).contiguous() * 0.25, coords.unsqueeze(0).contiguous(), filter_mode="linear", boundary_mode="clamp")
        return out

def generate_mipmaps(texture, max_mip_level):
    """
    Generates a list of mipmap levels using average pooling.
    Each mip level is a downscaled version of the input texture.
    
    Args:
        texture (Tensor): (H, W, C) input texture.
        max_mip_level (int): Number of mip levels to generate.

    Returns:
        List[Tensor]: List of textures from level 0 (original) to level N.
    """
    mipmaps = [texture]
    current = texture # (H,W,C)
    for _ in range(1, max_mip_level + 1):
        current = texture_mip.apply(current)
        mipmaps.append(current)
    return mipmaps


def sample_mip_texture(texture, max_mip_level, mip_level):
    """
    Samples the input texture at a specific mip level.

    Args:
        texture (Tensor): (H, W, C) original high-res texture.
        max_mip_level (int): Maximum mip level (number of downscales).
        mip_level (float): Desired mip level (can be fractional, soft-blended).

    Returns:
        Tensor: (H, W, C) texture at the same size as the original, upsampled back.
    """
    mipmaps = generate_mipmaps(texture, max_mip_level)

    mip_level = torch.tensor(mip_level, dtype=torch.float).clamp(min=0, max=max_mip_level)

    low = int(mip_level.floor().item())
    high = min(low + 1, max_mip_level)
    frac = mip_level - low

    # Blend between two mip levels (for smooth transitions)
    tex_low = mipmaps[low]
    tex_high = mipmaps[high]

    # Upsample both to original size
    H, W, _ = texture.shape
    tex_low_up = texture if low == 0 else F.interpolate(tex_low.permute(2, 0, 1).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0)
    tex_high_up = F.interpolate(tex_high.permute(2, 0, 1).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0)

    blended = (1 - frac) * tex_low_up + frac * tex_high_up
    return blended
