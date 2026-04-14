import torch
from torch import nn, Tensor

from avatar.shading import DeferredPBRShader
from avatar.gaussian_model import GaussianModel

""" DeferredPBRShader with per-gaussian material attributes.  """
class PrimitiveDeferredPBRShader(DeferredPBRShader):
    def __init__(self, args, device: torch.device, num_seq: int, gaussians: GaussianModel=None):
        super().__init__(args, device, num_seq)
        self.gaussians = gaussians

        # Initialize per-gaussian material
        self._material = torch.tensor([*self._init_albedo, self._init_roughness, self._init_spec],
                                      dtype=torch.float32, device=device).unsqueeze(0).repeat(gaussians.n_gaussians, 1)
        self._material = nn.Parameter(self._material)

    def get_render_features(self, seq_idx: Tensor, train_iter: int) -> Tensor:
        B = seq_idx.shape[0]
        mat = self.material(seq_idx, train_iter)
        return mat.unsqueeze(0).repeat(B, 1, 1) # (B, V, 5|8)

    def material(self, seq_idx: Tensor, train_iter: int) -> Tensor:
        mat = self._material # (n_gaussians, 5)
        if self.parametric_albedo is not None:
            param_albedo = self.parametric_albedo() # texture

            # Sample the parametric albedo texture map at the UV of each Gaussian
            uvs = self.gaussians.get_uvs()
            param_albedo = uv_sample(param_albedo, uvs) # (n_gaussians, 3)

            mat = torch.cat((mat, param_albedo), dim=-1) # (n_gaussians, 3+2+3)
        # Careful: this activation may mess with the parametric albedo if it's not just a ReLU
        return self._activate_material(mat)

    def restore(self, state):
        # if the number of gaussians changed
        if state["_material"].shape[0] != self._material.shape[0]:
            self._material = torch.nn.Parameter(state["_material"])
        super().restore(state)


def uv_sample(uv_map, uv, mode="bilinear"):
    S = uv_map.shape[0]
    n = uv.shape[0]
    # create a n x 1 grid of uvs for grid_sample
    uv_grid = uv.view(1, n, 1, 2)
    return torch.nn.functional.grid_sample(
        uv_map.permute(2,0,1).view(1,-1,S,S),
        uv_grid,
        mode=mode, padding_mode="border", align_corners=False
    ).view(-1, n).permute(1,0) # (n,-1)

