import torch
import numpy as np
import json
from pathlib import Path
import logging
from typing import List

from .dataset_util import load_img, load_mask, load_semantic, load_K_Rt_from_P, SemanticMask
from .camera import Camera
from utils.cam import focal2fov

class MonoFaceDataset(torch.utils.data.Dataset):
    """
    Instantiate a MonoFaceCompute-style face dataset (https://github.com/KelianB/MonoFaceCompute).
    Supports multiple videos.
    Args:
    - train_dirs: directories where sequences and their pre-computed information are stored.
    - sample_ratio: e.g. for sample_ratio = 4, every fourth frame will be used.
    - head_only: apply a semantic mask to the foreground mask to only keep the head.
    - load_normals: load pre-computed normal maps.
    - load_albedo: load pre-computed albedo maps.
    - seq_start: start frame for each sequence.
    - seq_end: end frame (non-inclusive) for each sequence.
    """
    def __init__(self, train_dirs: List[str], sample_ratio: int, head_only: bool,
                 load_normals=False, load_albedo=False, seq_start=0, seq_end=None):
        self.train_dirs = train_dirs
        self.num_seq = len(train_dirs)
        self.head_only = head_only

        self.load_normals = load_normals
        self.load_albedo = load_albedo
        self.load_semantic = True
        self.load_landmarks = True

        self.landmarks_mp = []
        self.landmarks = []

        per_sequence_intrinsics = []
        self.frame_info = []
        for seq_idx, dir in enumerate(self.train_dirs): 
            json_file = dir / "flame_params_optimized.json"

            with open(json_file, 'r') as f:
                json_data = json.load(f)

                n_max = len(json_data["frames"])
                seq_frames = list(range(n_max))[seq_start:seq_end]

                for item in json_data["frames"]:
                    # keep track of the subfolder
                    item.update({"dir": dir, "seq_idx": seq_idx})
                self.frame_info.extend([json_data["frames"][idx] for idx in seq_frames])
                
                per_sequence_intrinsics.append(json_data["intrinsics"])

            if self.load_landmarks:
                # Load MediaPipe landmarks
                landmarks_mp_file: Path = dir / "landmarks_mp.pt"
                if not landmarks_mp_file.exists():
                    raise RuntimeError(f"Cannot find Mediapipe landmarks for sequence '{dir}'")
                landmarks_mp = torch.load(landmarks_mp_file)[seq_frames]
                self.landmarks_mp.append(landmarks_mp)

                # Load FAN landmarks
                landmarks_fan_file: Path = dir / "landmarks_fan.pt"
                if not landmarks_fan_file.exists():
                    raise RuntimeError(f"Cannot find FAN landmarks for sequence '{dir}'")
                landmarks = torch.load(landmarks_fan_file)[seq_frames]
                # Load Iris landmarks
                iris_file: Path = dir / "landmarks_iris.pt"
                if iris_file.exists():
                    landmarks_iris = torch.load(iris_file)[seq_frames]
                    landmarks = torch.cat((landmarks, landmarks_iris), dim=1)
                self.landmarks.append(landmarks)

        self.frame_info = self.frame_info[::sample_ratio]
        if self.load_landmarks:
            self.landmarks_mp = torch.cat(self.landmarks_mp, dim=0)[::sample_ratio]
            self.landmarks = torch.cat(self.landmarks, dim=0)[::sample_ratio]

        self.size = len(self.frame_info)
        logging.info(f"Dataset: {self.size:d} views ({self.num_seq} {'sequences' if self.num_seq > 1 else 'sequence'})")
        test_path = self.frame_info[0]["dir"] / Path(self.frame_info[0]["file_path"] + ".png")
        self.resolution = load_img(test_path).shape[0:2]

        self.K = torch.eye(3).unsqueeze(0).repeat(self.num_seq, 1, 1)
        for seq_idx, (fx, fy, cx, cy) in enumerate(per_sequence_intrinsics):      
            self.K[seq_idx, 0, 0] = fx * self.resolution[0]
            self.K[seq_idx, 1, 1] = fy * self.resolution[1]
            self.K[seq_idx, 0, 2] = cx * self.resolution[0]
            self.K[seq_idx, 1, 2] = cy * self.resolution[1]

        frames_per_seq = [[] for _ in range(self.num_seq)]
        for idx in range(self.size):
            seq_idx = self.frame_info[idx]["seq_idx"]
            frames_per_seq[seq_idx].append(idx)
        self.frames_per_seq = [torch.tensor(x, dtype=torch.long) for x in frames_per_seq]

        self.shape_params = torch.tensor(json_data["shape_params"]).float().unsqueeze(0)

        # Init eyelid params as 0
        for f in self.frame_info:
            f["pose"] = f["pose"][:15] + [0.0, 0.0] + f["pose"][15:]

    def get_flame_pose(self, idx, device):
        json_dict = self.frame_info[idx]
        return torch.tensor(json_dict["pose"], dtype=torch.float32, device=device)

    def get_flame_expression(self, idx, device):
        json_dict = self.frame_info[idx]
        return torch.tensor(json_dict["expression"], dtype=torch.float32, device=device)

    def get_mean_expression(self):
        all_expression = torch.stack([self.get_flame_expression(i, "cpu") for i in range(self.size)])
        return all_expression.mean(dim=0, keepdim=True)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        json_dict = self.frame_info[idx]
        img_path = json_dict["dir"] / Path(json_dict["file_path"] + ".png")
        seq_idx = json_dict["seq_idx"]

        # ================ semantics =======================
        if self.load_semantic:
            semantic_parent = img_path.parent.parent / "semantic"
            semantic = load_semantic(semantic_parent / img_path.name)

        # ================ img & mask =======================
        img  = load_img(img_path)
        img_size = img.shape[1]

        mask_parent = img_path.parent.parent / "mask"
        if mask_parent.is_dir():
            mask_path = mask_parent / (img_path.stem + ".png")
            mask = load_mask(mask_path)
        elif img.ndim == 4:
            mask = img[..., 3].unsqueeze(-1)
            mask[mask < 0.5] = 0.0
            img = img[..., :3]
        else:
            raise RuntimeError(f"No mask found for image '{img_path}'")
    
        if self.head_only:
            head_mask = semantic[...,(SemanticMask.ALL_SKIN,SemanticMask.EYES,SemanticMask.EYEBROWS,SemanticMask.MOUTH_INTERIOR,SemanticMask.HAIR)].sum(dim=-1, keepdim=True) >= 1
            mask *= head_mask
        img = img * mask
    
        # ================ normals ================
        normals = None
        if self.load_normals:
            # normals_parent = img_path.parent.parent / "normals_dsine"
            # normals_parent = img_path.parent.parent / "normals_stablenormal"
            normals_parent = img_path.parent.parent / "normals_sapiens"
            if normals_parent.is_dir():
                if (normals_parent / img_path.name).is_file():
                    normals = load_img(normals_parent/ img_path.name) # (H, W, 3)
            else:
                raise RuntimeError(f"Could not find normals directory at {normals_parent}")

        # ================ albedo ================
        albedo = None
        if self.load_albedo:
            albedo_parent = img_path.parent.parent / "albedo_intrinsic_anything"
            if albedo_parent.is_dir():
                if (albedo_parent / img_path.name).is_file():
                    albedo  = load_img(albedo_parent / img_path.name) # (H, W, 3)
            else:
                raise RuntimeError(f"Could not find albedo directory at {albedo_parent}")
        
        # ================ flame and camera params =======================
        # flame params
        flame_pose = torch.tensor(json_dict["pose"], dtype=torch.float32)
        flame_expression = torch.tensor(json_dict["expression"], dtype=torch.float32)
        
        # camera to world matrix
        world_mat = torch.tensor(load_K_Rt_from_P(None, np.array(json_dict['world_mat']).astype(np.float32))[1], dtype=torch.float32)
        # camera matrix to openGL format 
        R = world_mat[:3, :3]
        R[1] *= -1
        R[2] *= -1
        T = world_mat[:3, 3]

        fl_x, fl_y = self.K[seq_idx, 0, 0], self.K[seq_idx, 1, 1]
        camera = Camera(img.device, R=R, T=T, image_width=img_size, image_height=img_size,
                        FoVx=focal2fov(fl_x, img_size), FoVy=focal2fov(fl_y, img_size))

        if self.load_landmarks:
            landmarks = self.landmarks[idx].float() * 2 / img_size - 1
            landmarks_mp = self.landmarks_mp[idx,:,:2].float() * 2 / img_size - 1

        frame_name = img_path.stem

        # Add batch dimension
        return {
            "img": img[None],
            "mask": mask[None],
            "semantic_mask": semantic[None],
            "flame_pose": flame_pose[None],
            "flame_expression": flame_expression[None],
            "camera": camera,
            "frame_name": frame_name,
            "idx": idx,
            "seq_idx": seq_idx,
            "landmarks": landmarks[None] if self.load_landmarks else None,
            "landmarks_mediapipe": landmarks_mp[None] if self.load_landmarks else None,
            "normals": normals[None] if normals is not None else None,
            "albedo": albedo[None] if albedo is not None else None,
        }

    def collate(self, batch):
        return MonoFaceDataset._collate(batch)

    ########## Static methods ##########

    def _collate(batch):
        return {
            "img": torch.cat([item["img"] for item in batch], dim=0),
            "mask": torch.cat([item["mask"] for item in batch], dim=0),
            "semantic_mask": torch.cat([item["semantic_mask"] for item in batch], dim=0) if batch[0]["semantic_mask"] is not None else None,
            "flame_pose": torch.cat([item["flame_pose"] for item in batch], dim=0),
            "flame_expression" : torch.cat([item["flame_expression"] for item in batch], dim=0),
            "camera": [item["camera"] for item in batch],
            "frame_name": [item["frame_name"] for item in batch],
            "idx": torch.LongTensor([item["idx"] for item in batch]),
            "seq_idx": torch.LongTensor([item["seq_idx"] for item in batch]),
            "landmarks" : torch.cat([item["landmarks"] for item in batch], dim=0) if batch[0]["landmarks"] is not None else None,
            "landmarks_mediapipe" : torch.cat([item["landmarks_mediapipe"] for item in batch], dim=0) if batch[0]["landmarks_mediapipe"] is not None else None,
            "normals": [item["normals"] for item in batch], # Could have None for some items
            "albedo": [item["albedo"] for item in batch], # Could have None for some items
        }
