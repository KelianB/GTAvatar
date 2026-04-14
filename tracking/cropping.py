import numpy as np
import torch
from skimage.transform import estimate_transform, warp
from torch import Tensor


def crop_transform(landmarks: Tensor, scale=1.0, image_size=224):
    left = landmarks[:, 0].min()
    right = landmarks[:, 0].max()
    top = landmarks[:, 1].min()
    bottom = landmarks[:, 1].max()

    old_size = (right - left + bottom - top) / 2
    center = torch.tensor([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size * scale)

    # crop image
    src_pts = torch.tensor([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = torch.tensor([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts.numpy(), DST_PTS.numpy())

    return tform

def crop_img(image: Tensor, kpt):
    device = image.device
    image = image.cpu().numpy()
    tform = crop_transform(kpt, scale=1.4, image_size=224)
    warped_image = warp(image, tform.inverse, output_shape=(224, 224), preserve_range=True)
    warped_image = torch.tensor(warped_image, device=device, dtype=torch.float32)
    # warped_kpt_mediapipe = np.dot(tform.params, np.hstack([kpt_mediapipe, np.ones([kpt_mediapipe.shape[0],1])]).T).T
    return warped_image

lmk_detector = None

def run_mediapipe(image: Tensor):
    # image: (H,W,C) float [0,1]

    global lmk_detector
    if lmk_detector is None:
        # Create landmarks detector
        from mediapipe.tasks.python import vision, BaseOptions
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="./assets/face_landmarker.task"),
            num_faces=1,
            min_face_detection_confidence=0.1, min_face_presence_confidence=0.1)
        lmk_detector = vision.FaceLandmarker.create_from_options(options)

    image_numpy = (image.cpu().numpy() * 255.0).astype(np.uint8)

    import mediapipe
    image_mp = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=image_numpy)
    detection_result = lmk_detector.detect(image_mp)
    if len(detection_result.face_landmarks) == 0:
        return None

    face_landmarks = detection_result.face_landmarks[0]
    lmk_torch = torch.zeros((478, 2), device=image.device)
    for i, landmark in enumerate(face_landmarks):
        lmk_torch[i,0] = landmark.x*image_mp.width
        lmk_torch[i,1] = landmark.y*image_mp.height
    return lmk_torch


class CachedImageCropper:
    def __init__(self):
        self._cache = dict()

    def _get_nearest_crop(self, idx):
        if idx in self._cache:
            crop = self._cache[idx]
        else:
            cached_idxs = list(self._cache.keys())
            if len(cached_idxs) == 0:
                return None
            min_idx, max_idx = min(cached_idxs), max(cached_idxs)
            crop = None
            k = 1
            while crop is None and (idx-k >= min_idx or idx+k <= max_idx):
                if idx-k in self._cache:
                    crop = self._cache[idx-k]
                elif idx+k in self._cache:
                    crop = self._cache[idx+k]
                k += 1
        return crop

    def __call__(self, idxs, images, lmks):
        # idx: (B) int
        # image: (B,H,W,C) float [0,1]

        # Get or compute crops
        crops = []
        for i, (idx, image) in enumerate(zip(idxs, images)):
            idx = idx.item()
            crop = self._cache.get(idx, None)
            if crop is None:
                lmk = run_mediapipe(image) if lmks is None else lmks[i]
                if lmk is None:
                    crop = self._get_nearest_crop(idx)
                    if crop is None:
                        print(f"Could not detect landmarks to crop image {idx}, and failed to fall back to a nearby crop.")
                        return None
                else:
                    crop = crop_img(image, lmk)
                    # Cache the crop on the GPU as a uint8 image (0-255)
                    self._cache[idx] = (crop * 255).to(torch.uint8)

            if crop.dtype == torch.uint8:
                crop = crop.to(torch.float32) / 255.0
            crops.append(crop)
        return torch.stack(crops) # (B, 224, 224, 3)
