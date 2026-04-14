

import os
import json
import torch
import numpy as np
from glob import glob
from PIL import Image
import json
from typing import List

from natsort import natsorted

from .mono_face_dataset import MonoFaceDataset
from .camera import Camera
from utils.cam import fov2focal, focal2fov
from utils.tqdm import tqdm

# Adapted from HRAvatar

class CameraParams():
    R: torch.Tensor
    T: torch.Tensor
    K: torch.Tensor
    FoVx : float
    FoVy: float
    shape_code: torch.Tensor
    translation_code: torch.Tensor
    eyelid_code: torch.Tensor
    full_pose_code: torch.Tensor
    exp_code: torch.Tensor
    image_name: str
    lmk_fan: torch.Tensor
    
    def __init__(self, K, R, T, shapecode, translation_code,eyelid_code, fullposecode, expcode,
                 FoVx, FoVy, image_name, lmk_fan):
        self.K = K
        self.R = R
        self.T = T
        self.shape_code = shapecode
        self.translation_code = translation_code
        self.eyelid_code = eyelid_code
        self.full_pose_code = fullposecode
        self.exp_code = expcode
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.lmk_fan = lmk_fan
    

class HRAvatarMonocularDataset(torch.utils.data.Dataset):
    def __init__(self, train_dirs: List[str], sample_ratio: int,
                 load_normals=False, load_albedo=False, seq_start=0, seq_end=None,
                 white_background=False):
        if sample_ratio != 1:
            raise NotImplementedError("HRAvatarMonocularDataset does not support sample_ratio")
        if len(train_dirs) > 1:
            raise NotImplementedError("HRAvatarMonocularDataset does not support multi-sequence")

        path = train_dirs[0]
        device = "cpu"

        self.device = device
        self.load_image = True
        self.load_landmarks = False
        self.load_normals = load_normals
        self.load_albedo = load_albedo

        n_expr = 50
        n_shape = 100

        if os.path.isdir(path):
            images_path = os.path.join(path, "image")
            if not os.path.exists(images_path):
                images_path = os.path.join(path, "images")
            imagepath_list = glob(images_path + '/*.jpg') + glob(images_path + '/*.png') + glob(images_path + '/*.bmp')
            imagepath_list = natsorted(imagepath_list)

        tracked_params_path = os.path.join(path, "tracked_params.json")
        self.flame_scale = 4.0
        if not os.path.exists(tracked_params_path):
            tracked_params_path = os.path.join(path, "tracked_params_v2.json")
            self.flame_scale = 1.0
        with open(tracked_params_path) as json_file:
            tracked_params_dict = json.load(json_file)

        img_slice = slice(seq_start, seq_end)
        imagepath_list = imagepath_list[img_slice]

        self.imagepath_list = imagepath_list

        # assume same resolution for all images
        with Image.open(imagepath_list[0]) as image:
            self.imagew, self.imageh = image.size[0], image.size[1]

        imagewh = torch.tensor([self.imagew, self.imageh], dtype=torch.float, device=device)

        if self.load_landmarks:
            # Load keypoints file
            with open(os.path.join(path, "keypoint.json")) as f:
                landmarks_fan = json.load(f)

            # Load iris keypoints file
            with open(os.path.join(path, "iris.json")) as f:
                landmarks_iris = json.load(f)

            # Load mediapipe keypoints
            self.landmarks_mp = torch.load(os.path.join(path, "keypoint_mp.pt"))[img_slice]

        self.bg = torch.tensor([1, 1, 1], dtype=torch.float32, device=device) if white_background else torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
        shapecode = torch.tensor(tracked_params_dict["shapecode"], device=device)[:, :n_shape]
        self.cam_params_list=[]
        for imagepath in tqdm(imagepath_list, desc="Processing dataset"):
            imagename = imagepath.split('/')[-1].split('.')[0]
            image_basename = os.path.basename(imagepath)
            imagekey = imagename if imagename in tracked_params_dict.keys() else image_basename
            
            if "translation" in tracked_params_dict[imagekey].keys():
                translation_code = torch.tensor(tracked_params_dict[imagekey]["translation"], dtype=torch.float32, device=device)
            else:
                translation_code = None
            if "eyelids" in tracked_params_dict[imagekey].keys():
                eyelid_code = torch.tensor(tracked_params_dict[imagekey]["eyelids"], dtype=torch.float32, device=device)
            else:
                eyelid_code = None
                
            if  "intrinsics"in tracked_params_dict.keys():
                intrinsics=tracked_params_dict["intrinsics"]#[fx, fy, cx, cy]
                fovx=2*np.arctan2(intrinsics[2],intrinsics[0])
            
            fullposecode = torch.tensor(tracked_params_dict[imagekey]["fullposecode"], dtype=torch.float32, device=device)
            expcode = torch.tensor(tracked_params_dict[imagekey]["expcode"], dtype=torch.float32, device=device)[:, :n_expr]
            
            fovy = focal2fov(fov2focal(fovx, self.imagew), self.imageh)
            K, R, T = self._load_camera(tracked_params_dict, imagekey, fovx, fovy)

            if self.load_landmarks:
                lmk_fan = torch.tensor(landmarks_fan[imagekey], dtype=torch.float, device=device)
                lmk_iris = torch.tensor(landmarks_iris[imagekey], dtype=torch.float, device=device)
                lmk_fan = torch.cat((lmk_fan, lmk_iris.reshape(2, 2)), dim=-2)
                lmk_fan = (lmk_fan / imagewh) * 2 - 1 # to range (-1,1)
            else:
                lmk_fan = None

            self.cam_params_list.append(CameraParams(K=K, R=R, T=T, shapecode=shapecode,
                                                     translation_code=translation_code, eyelid_code=eyelid_code,
                                                     fullposecode=fullposecode, expcode=expcode, FoVx=fovx, FoVy=fovy,
                                                     image_name=imagename, lmk_fan=lmk_fan))

        self.shape_params = self.cam_params_list[0].shape_code
        self.num_seq = 1

        self.size = len(self)

        frames_per_seq = [list(range(self.size))]
        self.frames_per_seq = [torch.tensor(x, dtype=torch.long) for x in frames_per_seq]

        self.K = self.cam_params_list[0].K.unsqueeze(0)
        for i in range(self.size):
            Ki = self.cam_params_list[i].K
            assert (Ki == self.K[0]).all() 

    def __getitem__(self, idx):    
        x = self.cam_params_list[idx]
        cam = Camera(self.device, R=x.R, T=x.T, image_width=self.imagew, image_height=self.imageh, FoVx=x.FoVx, FoVy=x.FoVy)

        img, mask, albedo, normals = self._load_images(self.imagepath_list[idx], self.bg) 

        return {
            "img": img[None],
            "mask": mask[None],
            "semantic_mask": None,
            "flame_pose": self.get_flame_pose(idx)[None],
            "flame_expression": self.get_flame_expression(idx)[None],
            "camera": cam,
            "frame_name": x.image_name,
            "idx": idx,
            "seq_idx": 0,
            "landmarks": x.lmk_fan[None] if self.load_landmarks else None,
            "landmarks_mediapipe": self.landmarks_mp[idx, :, :2][None] if self.load_landmarks else None,
            "normals": normals[None] if normals is not None else None,
            "albedo": albedo[None] if albedo is not None else None,
        }
    
    def __len__(self, ):
        return len(self.imagepath_list)

    def _load_images(self, imagepath, bg):
        bg=bg.numpy()
        pre_path = os.path.dirname(os.path.dirname(imagepath))
        image_basename = os.path.basename(imagepath)
        mask_prepath = os.path.join(pre_path, "mask")
       
        with Image.open(imagepath) as image:
            imagew, imageh = image.size[0], image.size[1]
            image = np.array(image, dtype=np.float32) / 255.0
        if image.shape[2] == 4:
            mask = image[:, :, 3:4]
            image = image[:, :, :3]
        elif os.path.exists(mask_prepath):
            maskpath = os.path.join(mask_prepath, image_basename)
            with Image.open(maskpath) as mask:
                mask = (np.array(mask, dtype=np.float32) / 255.0).mean(axis=2, keepdims=True)
        else:
            mask = np.ones((imageh, imagew, 1))
            
        image = image*mask + (1-mask)*bg
 
        albedo = None
        if self.load_albedo:
            albedo_path = os.path.join(pre_path, "albedo", image_basename)
            if os.path.exists(albedo_path):
                albedo = Image.open(albedo_path)
                albedo = np.array(albedo, dtype=np.float32) / 255.0
                albedo = albedo*mask + (1-mask)*bg
                albedo = torch.from_numpy(albedo).to(self.device)
                
        normals = None
        if self.load_normals:
            normals_path = os.path.join(pre_path, "normals", image_basename)
            if os.path.exists(normals_path):
                normals = Image.open(normals_path)
                normals = np.array(normals, dtype=np.float32) / 255.0
                normals = normals * mask
                normals = torch.from_numpy(normals).to(self.device)

        image = torch.from_numpy(image).to(self.device)
        mask = torch.from_numpy(mask).to(self.device)

        return image, mask, albedo, normals
 
    def _load_camera(self, tracked_params_dict, imagekey, fovx, fovy):
        w2c = np.array([[1 ,0 ,0 ,0 ],
                        [0 ,-1,0 ,0 ],
                        [0 ,0 ,-1,0 ],
                        [0 ,0 ,0 ,1 ]],dtype=np.float32) @ np.array(tracked_params_dict[imagekey]["world_mat"],dtype=np.float32)
        R = np.transpose(w2c[:3,:3])
        T = w2c[:3, 3]
        fx = fov2focal(fovx, self.imagew)
        fy = fov2focal(fovy, self.imageh)
        K = np.array([
            [fx, 0, self.imagew/2],
            [0, fy, self.imageh/2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        K = torch.tensor(K, dtype=torch.float32, device=self.device)
        R = torch.tensor(R, dtype=torch.float32, device=self.device)
        T = torch.tensor(T, dtype=torch.float32, device=self.device)
        return K, R, T
    
    def get_flame_pose(self, idx, device="cpu"):
        cam_param = self.cam_params_list[idx]
        full_pose = cam_param.full_pose_code.to(device) # (1, 15)
        if cam_param.eyelid_code:
            eyelids = cam_param.eyelid_code.to(device) # (1, 2)
        else:
            eyelids = torch.zeros((1, 2), dtype=torch.float, device=device)
        translation = cam_param.translation_code.to(device) # (1, 3)

        pose = torch.cat((full_pose, eyelids, translation), dim=-1).squeeze(0)
        # global_rot, neck_rot, jaw_rot, eyes_rot, eyelids, translation = pose[:,:3], pose[:,3:6], pose[:,6:9], pose[:,9:15], pose[:,15:17], pose[:,17:20]
        return pose

    def get_flame_expression(self, idx, device="cpu"):
        cam_param = self.cam_params_list[idx]
        expr = cam_param.exp_code.squeeze(0)
        return expr.to(device)

    def collate(self, batch):
        return MonoFaceDataset._collate(batch)
