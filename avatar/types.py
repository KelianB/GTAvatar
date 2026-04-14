from dataclasses import dataclass
from typing import Literal

from torch import Tensor

from utils.general import DotDict

@dataclass(frozen=True, slots=True)
class RenderSettings:
    # Randomly remove a fraction of Gaussians
    decimation_ratio: float = 0
    decimation_seed: int = 0
    # Override the opacity of all Gaussians
    override_opacity: float = -1
    # Hide Gaussians whose opacity is below this threshold
    opacity_culling: float = 0
    # Mask of Gaussians to render (boolean tensor of shape (n_gaussians,))
    gaussians_mask: Tensor | None = None
    # Render with random colors
    random_colors: bool = False
    # Render using hardware-accelerated textures for colors and normals. Beware: this is not supported for training,
    # and the textures will be cached until discard_hw_textures() is called
    hw_textures: bool = False
    # Evaluation mode enables certain optimizations for faster rendering
    # (e.g. caching ray directions, skipping computation of normals from depths, etc.)
    eval_mode: bool = False
    # Normals to use for shading (splatted | depth | mesh | mesh_flat)
    shading_normals: str = "splatted"
    # Final color to use when splatting
    background_color: Literal["white"] | Literal["black"] | Tensor = "black"
    # When relighting with an environment map, whether to include the background or not
    use_env_background: bool = True
    # Modifiers for the PBR shader
    brightness_scale: float = 1.0
    diffuse_scale: float = 1.0
    specular_scale: float = 1.0
    roughness_scale: float = 1.0
    metallic: float = 0
    # Multiplies the scales of all Gaussians
    scaling_multiplier: float = 1.0

@dataclass(frozen=True, slots=True)
class AvatarOutput:
    render: Tensor
    deformed_mesh_verts: Tensor
    rast_buffers: DotDict[str, Tensor]
    background: Tensor
    # Textures
    texture_material: Tensor | None
    texture_normals: Tensor | None
    # Gaussian attributes (after deformation, in world space)
    position: Tensor
    opacity: Tensor
    scaling: Tensor
    rotation: Tensor