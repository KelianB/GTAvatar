import os
from typing import Literal
import logging

import torch
import numpy as np

from utils.visualization import srgb_to_linear


class ParametricAlbedo(torch.nn.Module):
    def __init__(self, device: torch.device, tex_type: Literal["BFM", "FLAME"]):
        super().__init__()

        logging.info(f"Loading texture space '{tex_type}'")

        if tex_type == "BFM":
            tS = 512 # texture-space resolution
            tex_path = "assets/flame/FLAME_albedo_from_BFM.npz"
            if not os.path.exists(tex_path):
                raise FileNotFoundError(f"Basel texture space file not found at {tex_path}. Use the tool from https://github.com/TimoBolkart/BFM_to_FLAME to convert the BFM texture space to FLAME.")
            tex_space = np.load(tex_path)
            texture_mean = torch.from_numpy(tex_space["MU"]).to(torch.float32).view(tS, tS, 3)
            texture_mean = texture_mean.flip(2) # BGR to RGB
            texture_basis = torch.from_numpy(tex_space["PC"]).to(torch.float32).view(tS, tS, 3, -1)
            texture_basis = texture_basis.flip(2) # BGR to RGB
            texture_basis = texture_basis * 10
        elif tex_type == "FLAME":
            tS = 512 # texture-space resolution
            tex_path = "assets/flame/FLAME_texture.npz"
            if not os.path.exists(tex_path):
                raise FileNotFoundError(f"FLAME texture file not found at {tex_path}.")
            tex_space = np.load(tex_path)
            texture_mean = torch.from_numpy(tex_space["mean"]).to(torch.float32).view(tS, tS, 3)
            texture_mean = texture_mean.flip(2) / 255.0 # BGR to RGB
            texture_basis = torch.from_numpy(tex_space["tex_dir"]).to(torch.float32).view(tS, tS, 3, -1)
            texture_basis = texture_basis.flip(2) / 255.0 # BGR to RGB
        else:
            raise NotImplementedError()

        basis_size = texture_basis.shape[-1]

        if False:
            from dataset.dataset_util import save_img
            save_img(f"/bulk/test_texture_mean_{tex_type}.png", texture_mean)
        
        self.register_buffer("texture_mean", texture_mean, persistent=False)
        self.register_buffer("texture_basis", texture_basis, persistent=False)

        coefficients = torch.zeros((basis_size), dtype=torch.float32, requires_grad=True)
        self.coefficients = torch.nn.Parameter(coefficients)

        self.to(device)

    def forward(self) -> torch.Tensor:
        tex_albedo = self.texture_mean + (self.texture_basis * self.coefficients).sum(-1) # (gS, gS, 3)

        # The parametric albedo is assumed to be in sRGB space
        tex_albedo = srgb_to_linear(tex_albedo)

        return tex_albedo
