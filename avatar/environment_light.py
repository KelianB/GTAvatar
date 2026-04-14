import glob
from pathlib import Path
from typing import Union, List, Tuple, Callable
from PIL import Image
import os

import torch
from torch import Tensor
import nvdiffrast.torch as dr
import numpy as np
import imageio.v3 as imageio
import cv2

from nvdiffrec.render.light import cubemap_mip
from nvdiffrec.render import util
from nvdiffrec.render.renderutils import xfm_vectors, diffuse_cubemap, specular_cubemap

# Adapted from nvdiffrec
class EnvironmentLight(torch.nn.Module):
    LIGHT_MIN_RES = 16

    MIN_ROUGHNESS = 0.08
    MAX_ROUGHNESS = 0.5

    def __init__(self, base: Tensor, trainable=False, original_latlong=None, mip_levels=None,
                 activation: Callable[[Tensor], Tensor] | None=None):
        super(EnvironmentLight, self).__init__()

        self.original_latlong = original_latlong
        self.base = base.clone().detach()
        if trainable:
            self.base = torch.nn.Parameter(self.base, requires_grad=True)
        self.activation = activation
        
        if self.base.shape[1] == 32:
            self.LIGHT_MIN_RES = 8
        
        self.mip_levels = mip_levels
        if mip_levels is not None:
            # Calculate resolution of smallest mip level
            min_res = self.base.shape[1] // (2 ** mip_levels)
            # nvdiffrec yields NaNs for cubemaps that are too small - prevent this here
            if min_res < 8:
                raise RuntimeError(f"EnvironmentLight: requested {mip_levels} mip levels for base resolution {self.base.shape[1]}, " +
                                   f"which would result in a minimum resolution of {min_res}x{min_res}. Minimum supported is 8x8.")

    def to(self, device=None):
        self.base = self.base.to(device)
        if hasattr(self, "specular"):
            self.specular = [s.to(device) for s in self.specular]
        if hasattr(self, "diffuse"):
            self.diffuse = self.diffuse.to(device)
        return super().to(device)

    def get_mip(self, roughness):
        rmin, rmax = self.MIN_ROUGHNESS, self.MAX_ROUGHNESS
        return torch.where(roughness < rmax,
                           (roughness.clamp(rmin, rmax) - rmin) / (rmax - rmin) * (len(self.specular) - 2),
                           (roughness.clamp(rmax, 1.00) - rmax) / (1.00 - rmax) + len(self.specular) - 2)

    def prefilter(self):
        ''' Prefilter the environment map and build the mipmap chain. '''

        costheta_cutoff = 0.99
        base = self.base
        # if self.activation:
        #     base = self.activation(base)

        self.specular = [base]
        if self.mip_levels is None:
            while self.specular[-1].shape[1] > self.LIGHT_MIN_RES:
                self.specular.append(cubemap_mip.apply(self.specular[-1]))
        else:
            for _ in range(self.mip_levels):
                self.specular.append(cubemap_mip.apply(self.specular[-1]))
        self.diffuse = diffuse_cubemap(self.specular[-1])

        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) + self.MIN_ROUGHNESS
            self.specular[idx] = specular_cubemap(self.specular[idx], roughness, costheta_cutoff) 
        self.specular[-1] = specular_cubemap(self.specular[-1], 1.0, costheta_cutoff)

        # Note: activating after prefiltering is incorrect but yields more stable results
        if self.activation:
            self.diffuse = self.activation(self.diffuse)
            self.specular = [self.activation(s) for s in self.specular]

# Adapted from HRAvatar, use this for relighting comparisons
class HRAvatarEnvLight(torch.nn.Module):
    def __init__(self, lalong_dir, init_res=1024, mip_map_count=7, device="cuda:0"):
        super(HRAvatarEnvLight, self).__init__()
        diffuse_path=os.path.join(lalong_dir,"diffuse.tga")
        self.device=device
        print("Loading prefiltered environment map...")

        hdr_file = glob.glob(os.path.join(lalong_dir,f'*.hdr'), recursive=True)
        if len(hdr_file) == 0:
            raise RuntimeError(f"No .hdr file found in envmap dir {lalong_dir}")
        self.original_latlong = load_latlong(hdr_file[0], device=device)

        diffuse_lalong_map=torch.tensor(np.array(Image.open(diffuse_path)),device=device)/255

        w, h = init_res, init_res
        specular_lalong_map=[]
        for i in range(mip_map_count):
            w=int(h)
            h=int(w/2)
            specular_path = os.path.join(lalong_dir,f"specular_{i}_{w}x{h}.tga")
            lalong_map=torch.tensor(np.array(Image.open(specular_path)),device=device)/255
            specular_lalong_map.append(lalong_map)
        self.mip_map_count=len(specular_lalong_map)

        # HRAvatar's renders have flipped environment maps compared to how they appear on PolyHaven.
        # This is not an issue as the lighting is coherent, but we flip it here too for easier comparison.
        horizontal_flip = True
        if horizontal_flip:
            diffuse_lalong_map = diffuse_lalong_map.flip([1])
            self.original_latlong = self.original_latlong.flip([1])
            for i, x in enumerate(specular_lalong_map):
                specular_lalong_map[i] = specular_lalong_map[i].flip([1])

        self.specular_lalong_map=specular_lalong_map
        self.device=device
        with torch.no_grad():
            self.diffuse_map=latlong_to_cubemap(diffuse_lalong_map,[diffuse_lalong_map.shape[0],diffuse_lalong_map.shape[0]]).to(device)
            self.diffuse = diffuse_cubemap(self.diffuse_map)
            self.specular_map=[]
            self.specular=[]
            for idx,latlong_map in enumerate(specular_lalong_map):
                self.specular_map.append(latlong_to_cubemap(latlong_map, [latlong_map.shape[0],latlong_map.shape[0]]).to(device))
                self.specular.append(specular_cubemap(self.specular_map[-1], (idx+1)/mip_map_count))
        print("Done")

    def get_mip(self, roughness):
        return (roughness * (self.mip_map_count-1)).contiguous()

def lookup_env_light(env: Union[List[EnvironmentLight | HRAvatarEnvLight], EnvironmentLight | HRAvatarEnvLight],
                     normals: Tensor, wr: Tensor, roughness: Tensor, rot: Tensor | None=None) -> Tuple[Tensor, Tensor]:
    """
    - normals: Tensor (B,H,W,3)
    - wr: Tensor (B,H,W,3)
    - roughness: Tensor (B,H,W,1)
    - rot (optional): Tensor (B,4,4) - Rotation matrices to rotate the env light
    """
    B = normals.shape[0]
    assert normals.ndim == 4 and normals.shape[-1] == 3
    assert wr.ndim == 4 and wr.shape[-1] == 3 and wr.shape[0] == B
    assert roughness.ndim == 4 and roughness.shape[-1] == 1 and roughness.shape[0] == B

    if isinstance(env, list):
        diff_spec = [lookup_env_light(el, normals[i].unsqueeze(0), wr[i].unsqueeze(0), roughness[i].unsqueeze(0), rot=rot)
                     for i,el in enumerate(env)]
        diffuse = torch.cat([x[0] for x in diff_spec], dim=0)
        specular = torch.cat([x[1] for x in diff_spec], dim=0)
        return diffuse, specular

    if rot is not None:
        assert tuple(rot.shape) == (B, 4, 4) 
        # Transform normals and reflected vector to rotate the env light
        normals = xfm_vectors(normals.view(B, -1, 3), rot).view(*normals.shape)
        wr = xfm_vectors(wr.view(B, -1, 3), rot).view(*wr.shape)

    diffuse = dr.texture(env.diffuse.unsqueeze(0), normals.contiguous(), filter_mode="linear", boundary_mode="cube")
    
    miplevel = env.get_mip(roughness)
    specular = dr.texture(env.specular[0].unsqueeze(0), wr, mip=[x.unsqueeze(0) for x in env.specular[1:]], mip_level_bias=miplevel[..., 0], filter_mode="linear-mipmap-linear", boundary_mode="cube")

    return diffuse, specular

# Adapted from nvdiffrec
def latlong_to_cubemap(latlong_map, res, sides=range(6), mtx=None):
    device = latlong_map.device
    cubemap = torch.zeros(6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device=device)
    for s in sides:
        gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device=device), 
                                torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device=device),
                                indexing='ij')
        v = util.safe_normalize(util.cube_to_dir(s, gx, gy))
        if mtx is not None:
            v = xfm_vectors(v.view(1, v.shape[0] * v.shape[1], v.shape[2]), mtx).view(*v.shape)

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s, ...] = dr.texture(latlong_map[None, ...], texcoord[None, ...], filter_mode='linear')[0]
    return cubemap

def load_latlong(filename, device="cuda"):
    img = imageio.imread(filename, flags=cv2.IMREAD_UNCHANGED, plugin="opencv")
    img = torch.tensor(img, dtype=torch.float32, device=device) # HDR image, 0 to inf
    return img

def load_envmap(filename: Union[str, Path], device="cpu", hravatar_compat=False) -> EnvironmentLight:
    if hravatar_compat:
        if not os.path.isdir(filename):
            raise RuntimeError(f"Expected a directory for HRAvatar compatibility mode, got '{filename}'")

        l = HRAvatarEnvLight(filename, device=device)
    else:
        ext = os.path.splitext(filename)[1]
        if ext.lower() != ".hdr":
            raise RuntimeError(f"Environment map extension should be .hdr, got '{filename}'")
        
        # Load from latlong .HDR file
        latlong_img = load_latlong(filename, device=device)
        cubemap = latlong_to_cubemap(latlong_img, [512, 512])
        l = EnvironmentLight(cubemap, original_latlong=latlong_img)
        l = l.to(device)
        l.prefilter()

    return l

def get_env_light_background(env: Union[List[EnvironmentLight], EnvironmentLight], env_rot=None, target_size=(512,512)) -> Tensor:
    assert env_rot is None or env_rot.ndim == 3
    if isinstance(env, list):
        return torch.stack([get_env_light_background(env[i], None if env_rot is None else env_rot[i].unsqueeze(0)) for i in range(len(env))]) 

    # Convert the latitude-longitude environment map to a cubemap while applying rotation, and use the back face as the background
    background = latlong_to_cubemap(env.original_latlong, target_size, sides=[5], mtx=env_rot)[5]
    return background
