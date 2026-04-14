import torch
from torch import nn, Tensor

from utils.cam import getProjectionMatrix, getWorld2View

class Camera(nn.Module):
    def __init__(self, device: torch.device, R: Tensor, T: Tensor, FoVx: float, FoVy: float, image_width: int, image_height: int):
        super().__init__()
        self.device = device
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = image_width
        self.image_height = image_height

        # fx = fov2focal(FoVx, image_width)
        # fy = fov2focal(FoVy, image_height)
        # self.K = torch.tensor([
        #     [fx, 0, (image_width-1)/2],
        #     [0, fy, (image_height-1)/2],
        #     [0, 0, 1],
        # ], dtype=torch.float, device=device) # (3, 3)

        self.znear = 0.01
        self.zfar = 100.0

        self.world_view_transform = getWorld2View(self.R, self.T).transpose(0, 1).to(device)
        self.world_view_transform_inv = self.world_view_transform.inverse()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, device=device).transpose(0,1)
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix

    def to(self, device="cpu"):
        self.R = self.R.to(device)
        self.T = self.T.to(device)
        self.world_view_transform = self.world_view_transform.to(device)
        self.world_view_transform_inv = self.world_view_transform_inv.to(device)
        self.projection_matrix = self.projection_matrix.to(device)
        self.full_proj_transform = self.full_proj_transform.to(device)
        self.device = device
        return self

    def clone(self) -> "Camera":
        return Camera(
            device=self.device,
            R=self.R.clone(),
            T=self.T.clone(),
            FoVx=self.FoVx,
            FoVy=self.FoVy,
            image_width=self.image_width,
            image_height=self.image_height,
        )

    @property
    def camera_center(self):
        return self.world_view_transform_inv[3, :3]

    def set_resolution(self, image_width, image_height):
        self.image_width = image_width
        self.image_height = image_height

    def set_fov(self, fov):
        self.FoVx = fov
        self.FoVy = fov
        # Update the projection matrices
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, device=self.device).transpose(0,1)
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix

    def world_to_ndc(self, x: Tensor) -> Tensor:
        p3d_hom = torch.cat((x, torch.ones_like(x[...,(0,)])), dim=-1) # (V, 4)
        p_hom = p3d_hom @ self.full_proj_transform # (V, 4)
        p_ndc = p_hom[..., :-1] / (p_hom[..., (-1,)] + 1e-7) # (V, 3)
        return p_ndc
    
    def world_to_screen(self, x: Tensor) -> Tensor:
        p_ndc = self.world_to_ndc(x)
        im_size = torch.tensor([self.image_width, self.image_height, 1], device=x.device)
        p_screen = ((p_ndc + 1.0) * im_size - 1) * 0.5 # (V, 3)
        return p_screen

    def compute_ray_dirs(self) -> Tensor:
        # Adapted from https://github.com/hbb1/2d-gaussian-splatting/blob/main/utils/point_utils.py
        W, H = self.image_width, self.image_height
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, W / 2],
            [0, H / 2, 0, H / 2],
            [0, 0, 0, 1]], dtype=torch.float, device=self.device).T

        intrins = (self.projection_matrix @ ndc2pix)[:3,:3]
        
        grid_x, grid_y = torch.meshgrid(
            torch.arange(W, device=self.device, dtype=torch.float), 
            torch.arange(H, device=self.device, dtype=torch.float), indexing='xy')
        points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1)
        rays_d = points.view(-1, 3) @ intrins.inverse() @ self.world_view_transform_inv[:3,:3]
        return rays_d.view(H, W, 3)

    def depth_to_points(self, depth: Tensor) -> Tensor:
        """
            depth: depth map (float Tensor of shape (H,W,1)) 
        """
        # Adapted from https://github.com/hbb1/2d-gaussian-splatting/blob/main/utils/point_utils.py
        rays_d = self.compute_ray_dirs() # (H,W,3)
        rays_o = self.camera_center
        points = depth * rays_d + rays_o.unsqueeze(0).unsqueeze(0)
        return points # (H,W,3)

    def depth_to_normal(self, depth: Tensor) -> Tensor:
        """
            depth: depth map (float Tensor of shape (H,W,1)) 
        """
        # Adapted from https://github.com/hbb1/2d-gaussian-splatting/blob/main/utils/point_utils.py
        points = self.depth_to_points(depth)
        dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
        dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
        normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
        output = torch.zeros_like(points)
        output[1:-1, 1:-1, :] = normal_map
        return output
