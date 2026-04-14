import torch
from torch import Tensor

from .mono_face_dataset import MonoFaceDataset
from .camera import Camera

class DummyDataset(torch.utils.data.Dataset):
    """
    Instantiate a fake Dataset so we can load an avatar without requiring the associated data.
    """
    def __init__(self, poses: Tensor | None = None, expressions: Tensor | None = None):
        self.shape_params = torch.zeros((1, 100), dtype=torch.float32)
        self.num_seq = 1
        if poses is None:
            poses = torch.zeros((1, 20), dtype=torch.float32, device="cpu")
        if expressions is None:
            expressions = torch.zeros((1, 50), dtype=torch.float32, device="cpu")
        assert expressions.shape[0] == poses.shape[0]

        self.poses = poses
        self.exprs = expressions

    def get_flame_pose(self, idx, device):
        return self.poses[idx].to(device)
                
    def get_flame_expression(self, idx, device):
        return self.exprs[idx].to(device)

    def get_mean_expression(self):
        return self.exprs.mean(dim=0)

    def __len__(self):
        return self.poses.shape[0]

    def __getitem__(self, idx):
        device = "cpu"
        R = torch.tensor([[ 1,  0,  0],
                          [ 0, -1,  0],
                          [ 0,  0, -1]], dtype=torch.float, device=device)
        T = torch.tensor([0, 0, 4], dtype=torch.float, device=device)
        size = 512
        camera = Camera(device, R=R, T=T, image_width=size, image_height=size, FoVx=0.336, FoVy=0.336)
        img = torch.zeros((size, size, 3), dtype=torch.float, device=device)
        mask = torch.zeros((size, size, 1), dtype=torch.float, device=device)
        flame_pose = self.get_flame_pose(idx, device)
        flame_expression = self.get_flame_expression(idx, device)

        return {
            "img": img[None],
            "mask": mask[None],
            "semantic_mask": None,
            "flame_pose": flame_pose[None],
            "flame_expression": flame_expression[None],
            "camera": camera,
            "frame_name": "dummy",
            "idx": idx,
            "seq_idx": 0,
            "landmarks": None,
            "landmarks_mediapipe": None,
            "normals": None,
            "albedo": None,
        }

    def collate(self, batch):
        return MonoFaceDataset._collate(batch)
