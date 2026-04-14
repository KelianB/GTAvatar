import torch
from torch.utils.data import Dataset, DataLoader
import imageio
import numpy as np
import cv2
from typing import List, Tuple

###############################################################################
# Helpers/utils
###############################################################################

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose

def load_mask(fn):
    mask = imageio.imread(fn, mode='F') # float np.ndarray, range [0.0, 255.0], shape (H, W)
    mask = mask.astype(np.float32) / 255.0
    mask = torch.from_numpy(mask).unsqueeze(-1)  
    mask[mask < 0.5] = 0.0
    return mask

def load_img(fn):
    img = imageio.imread(fn)
    if img.dtype != np.float32: # LDR image
        img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img)
    return img

# this is not an Enum because we want to be able to use the values as indices directly
class SemanticMask:
    ALL_SKIN = 0
    EYES = 1
    EYEBROWS = 2
    MOUTH_INTERIOR = 3
    CLOTH_NECKLACE = 4
    HAIR = 5
    HEAD_NONFACE = 7
    BACKGROUND = 8
    EARS = 9
    NOSE = 10
    UPPER_LIP = 11
    LOWER_LIP = 12
    NECK = 13
    SKIN = 14

    BACKGROUND_CLOTH_NECKLACE_NECK = 15
    SKIN_HAIR = 16
    SKIN_EYEBROWS = 17
    HAT = 18
    DECA_MASK = 19
# BiseNet mapping:
# skin: 1, l_brow: 2, r_brow: 3, l_eye: 4, r_eye: 5, eye_g: 6, l_ear: 7, r_ear: 8, ear_r: 9
# nose: 10, mouth: 11, u_lip: 12, l_lip: 13, neck: 14, neck_l: 15, cloth: 16, hair: 17, hat: 18

def load_semantic(filename):
    img = imageio.imread(filename, mode='F')
    h, w = img.shape
    sem = np.zeros((20, h, w))
    sem[SemanticMask.SKIN] = (img == 1) # skin
    sem[SemanticMask.ALL_SKIN] = ((img == 1) + (img == 10) + (img == 8) + (img == 7) + (img == 14) + (img == 6) + (img == 12) + (img == 13)) >= 1 # skin, nose, ears, neck, lips
    sem[SemanticMask.EYES] = ((img == 4) + (img == 5)) >= 1 # left eye, right eye
    sem[SemanticMask.EYEBROWS] = ((img == 2) + (img == 3)) >= 1 # left eyebrow, right eyebrow
    sem[SemanticMask.MOUTH_INTERIOR] = (img == 11) # mouth interior
    sem[SemanticMask.CLOTH_NECKLACE] = ((img == 15) + (img == 16)) >= 1 # cloth, necklace
    sem[SemanticMask.HAIR] = ((img == 17) + (img == 9)) >= 1 # hair
    sem[SemanticMask.HEAD_NONFACE] = ((img == 17) + (img == 9) + (img == 15) + (img == 16) + (img == 14)) >= 1 # hair, cloth, necklace, neck
    sem[SemanticMask.BACKGROUND] = np.clip(1. - np.sum(sem[:8], 0), 0, 1) # background
    sem[SemanticMask.EARS] = ((img == 7) + (img == 8)) >= 1
    sem[SemanticMask.NOSE] = (img == 10) >= 1
    sem[SemanticMask.UPPER_LIP] = (img == 12) >= 1
    sem[SemanticMask.LOWER_LIP] = (img == 13) >= 1
    sem[SemanticMask.NECK] = (img == 14) >= 1
    sem[SemanticMask.BACKGROUND_CLOTH_NECKLACE_NECK] = (sem[SemanticMask.BACKGROUND] + sem[SemanticMask.CLOTH_NECKLACE] + sem[SemanticMask.NECK]) >= 1
    sem[SemanticMask.SKIN_HAIR] = (sem[SemanticMask.SKIN] + sem[SemanticMask.HAIR]) >= 1
    sem[SemanticMask.SKIN_EYEBROWS] = (sem[SemanticMask.SKIN] + sem[SemanticMask.EYEBROWS]) >= 1
    sem[SemanticMask.HAT] = (img == 18) >= 1

    sem[SemanticMask.DECA_MASK] = (sem[SemanticMask.BACKGROUND] + sem[SemanticMask.EARS] + sem[SemanticMask.HAIR] + sem[SemanticMask.HAT] + sem[SemanticMask.NECK] + sem[SemanticMask.CLOTH_NECKLACE]) == 0

    sem = torch.tensor(sem, dtype=torch.bool).permute(1,2,0) # CHW to HWC
    return sem

def compute_face_crops(images: torch.Tensor, landmarks: torch.Tensor, boxes_only=False, crop_scale=1, offset_y=0) -> Tuple[List[torch.Tensor], torch.Tensor]:
    B, H, W, _ = images.shape

    crops = []
    boxes = torch.zeros((B, 3), dtype=torch.int, device=images.device)

    for i, (img, lmk) in enumerate(zip(images, landmarks)):
        H, W, _ = img.shape
        lmk_x, lmk_y = lmk.unbind(1) # (L,2)
        lmk_x = (lmk_x+1) * (W/2)
        lmk_y = (lmk_y+1) * (H/2)
        # Calculate a crop based on the landmarks
        left, right = lmk_x.min(), lmk_x.max()
        top, bottom = lmk_y.min(), lmk_y.max()
        # Make the box a square and scale it up
        s = (right - left + bottom - top) / 2
        center_x = right - (right - left) / 2.0
        center_y =  bottom - (bottom - top) / 2.0
        s *= crop_scale
        x, y = center_x - s/2, center_y - s/2

        y += offset_y

        # Ensure crop stays within image
        x = x.round().clamp(min=0, max=W-s).int()
        y = y.round().clamp(min=0, max=H-s).int()
        s = s.round().clamp(min=0, max=min(H, W)).int()

        if not boxes_only:
            crops.append(img[y:y+s, x:x+s,:])
        boxes[i, 0] = x
        boxes[i, 1] = y
        boxes[i, 2] = s

    return boxes if boxes_only else (crops, boxes)

#----------------------------------------------------------------------------
# Generic utilities for torch data handling
#----------------------------------------------------------------------------

def to_device_recursive(v, device):
    """ Move a whole structure of tensors (dict, list, tuple) to the given device. """
    if torch.is_tensor(v):
        return v.to(device)
    elif isinstance(v, dict):
        return {k: to_device_recursive(x, device) for k,x in v.items()}
    elif isinstance(v, list):
        return [to_device_recursive(x, device) for x in v]
    elif isinstance(v, tuple):
        return tuple(to_device_recursive(x, device) for x in v)
    elif callable(getattr(v, "to", None)):
        return v.to(device)
    return v

def find_collate(d: Dataset):
    """ Find a 'dataset.collate' function from nested datasets. """
    if hasattr(d, "collate"):
        return d.collate
    elif hasattr(d, "dataset"):
        return find_collate(d.dataset)
    else:
        raise RuntimeError("No collate fn found")

class DeviceDataLoader(DataLoader):
    """
    Wrapper of PyTorch's `DataLoader` class for automatically sending data to device.

    Args:
        dataset (`torch.utils.data.Dataset`):
            The dataset to use to build this dataloader.
        device (`torch.Device`, defaults to cpu):
            A device to send tensors to.
        kwargs:
            All other keyword arguments to pass to the regular `DataLoader` initialization.
    """

    def __init__(self, dataset: Dataset, device='cpu', **kwargs):
        super().__init__(dataset, **kwargs)
        self.device = device

    def __iter__(self):
        for batch in super().__iter__():
            yield to_device_recursive(batch, self.device)

class DatasetCache(Dataset):
    """
    Wrapper of PyTorch's `Dataset` class for caching samples.
    WARNING: needs to be used with a DataLoader that has num_workers set to 0.
    
    Args:
        dataset (`torch.utils.data.Dataset`):
            The child dataset to cache.
        kwargs:
            Keyword arguments to pass to the regular `Dataset` constructor.
    """

    def __init__(self, dataset: Dataset, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.cache = dict()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        if idx not in self.cache:
            self.cache[idx] = self.dataset[idx]
        return self.cache[idx]
       
    def __len__(self):
        return len(self.dataset)

    def release_cache(self):
        self.cache = dict()

    # Fallback: defer all other method calls to underlying dataset    
    def __getattr__(self, *args):
        return self.dataset.__getattribute__(*args)

class ZipDatasets(Dataset):
    def __init__(self, *datasets, **kwargs):
        super().__init__(**kwargs)
        self.datasets = datasets
        self.collate_fns = [find_collate(d) for d in datasets]

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
    
    def collate(self, batch: tuple):
        assert type(batch) in [tuple, list]
        return tuple(collate_fn([tup[i] for tup in batch]) for i, collate_fn in enumerate(self.collate_fns))

