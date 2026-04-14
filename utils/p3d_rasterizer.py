import torch
from torch import nn
from pytorch3d.renderer.mesh import rasterize_meshes
from pytorch3d.structures import Meshes

class Pytorch3dRasterizer(nn.Module):
    """
    This class implements methods for rasterizing a batch of heterogenous Meshes.
    Notice:
        x,y,z are in image space, normalized
        can only render squared image now
    """

    def __init__(self):
        super().__init__()
        self.raster_settings = {
            "blur_radius": 0.0,
            "faces_per_pixel": 1,
            "bin_size": None,
            "max_faces_per_bin": None,
            "perspective_correct": True,
        }

    def forward(self, vertices, faces, image_height, image_width, attributes=None, advanced=False):
        """
        args:
            vertices: (B, V, 3)
            faces: (B, V, 3)
            attributes: (B, F, 3, 3)
        returns:
            (B, H, W, n+1)
            n channels corresponding to the given attributes
            last channels are respectively for visibility mask and depth buffer            
        """
        assert vertices.ndim == 3
        assert faces.ndim == 3
        assert vertices.shape[0] == faces.shape[0]
        B, V, _ = vertices.shape
        _, F, _ = faces.shape 
        assert attributes is None or attributes.shape[:-1] == (B, F, 3)

        x, y, z = vertices.unbind(-1)
        fixed_vertices = torch.stack((-x, -y * image_height/image_width, z), dim=-1)
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        raster_settings = self.raster_settings

        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=(image_height, image_width),
            blur_radius=raster_settings["blur_radius"],
            faces_per_pixel=raster_settings["faces_per_pixel"],
            bin_size=raster_settings["bin_size"],
            max_faces_per_bin=raster_settings["max_faces_per_bin"],
            perspective_correct=raster_settings["perspective_correct"],
        )

        mask_vis = (pix_to_face > -1).float()
        mask_novis = pix_to_face == -1
        pix_to_face[mask_novis] = 0

        if attributes is None:
            pixel_vals = torch.cat((mask_vis, zbuf), dim=-1)
        else:
            N, H, W, K, _ = bary_coords.shape
            D = attributes.shape[-1]
            attributes = attributes.view(B * F, 3, D)
            idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
            pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
            pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
            pixel_vals[mask_novis] = 0  # Replace masked values in output.
            pixel_vals = pixel_vals.squeeze(3) # (B, H, W, 3)
            # pixel_vals = torch.cat((pixel_vals.squeeze(3), mask_vis, zbuf), dim=-1)
            pixel_vals = torch.cat((pixel_vals, mask_vis, zbuf), dim=-1)

        if advanced:
            return pixel_vals, pix_to_face, bary_coords
        else:
            return pixel_vals

def vertices_to_face(vertex_attr, faces):
    """
    vertices: (B, V, 3)
    faces: (B, F, 3)
    return: (B, F, 3, 3)
    """
    assert (vertex_attr.ndim == 3)
    assert (faces.ndim == 3)
    assert (vertex_attr.shape[0] == faces.shape[0])
    assert (faces.shape[2] == 3)

    B, V = vertex_attr.shape[:2]
    device = vertex_attr.device
    faces = faces + (torch.arange(B, dtype=torch.int32).to(device) * V)[:, None, None]
    vertex_attr = vertex_attr.reshape(B * V, -1)
    # pytorch only supports long and byte tensors for indexing
    return vertex_attr[faces.long()]
