import logging

import torch

from flame import FLAME


def sample_flame(flame: FLAME, samples_per_face: int | float = 5):

    xyz, faces = flame.v_template, flame.faces
    F = faces.shape[0]
    device = xyz.device

    fractional_samples_per_face = False
    if samples_per_face < 1:
        fractional_samples_per_face = samples_per_face
        samples_per_face = 1
    else:
        assert samples_per_face == int(samples_per_face)
        samples_per_face = int(samples_per_face)

    bary = torch.rand((F, samples_per_face, 3), device=device)

    # Sampling on mesh suface
    bary = bary / (bary.sum(dim=2, keepdim=True) + 1e-5)
    
    xyz_sampled = bary @ xyz[faces]
    xyz_sampled = xyz_sampled.reshape(-1,3)

    triangles_uvs = flame.verts_uvs[flame.textures_idx]
    triangles_uvs = triangles_uvs * 2 - 1
    triangles_uvs[..., 1] = -triangles_uvs[..., 1]

    uvs_sampled = bary @ triangles_uvs
    uvs_sampled = uvs_sampled.reshape(-1,2)

    tri_idx = torch.arange(0, F, 1/samples_per_face, device=device).floor().to(torch.long)
    bary = bary.reshape(-1, 3)

    if fractional_samples_per_face:
        N = F * samples_per_face
        n = round(N * fractional_samples_per_face)
        logging.info(f"fractional_samples_per_face = {fractional_samples_per_face}, faces: {F} : selecting {n}/{N} random gaussians")
        g = torch.Generator()
        g.manual_seed(0)
        indices = torch.randperm(N, generator=g)[:n]
        xyz_sampled, uvs_sampled, bary, tri_idx = xyz_sampled[indices], uvs_sampled[indices], bary[indices], tri_idx[indices]

    return xyz_sampled, uvs_sampled, bary, tri_idx
