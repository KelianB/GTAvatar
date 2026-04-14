import math

import numpy as np
import torch
from torch import Tensor


def getWorld2ViewNp(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    Rt = np.zeros((4, 4), dtype=R.dtype)
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt

def getWorld2View(R: Tensor, t: Tensor) -> Tensor:
    Rt = torch.zeros((4, 4), device=R.device, dtype=R.dtype)
    Rt[:3, :3] = R.t()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt

def getProjectionMatrix(znear: float, zfar: float, fovX: float, fovY: float, device: torch.device = "cpu") -> Tensor:
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros((4, 4), dtype=torch.float32, device=device)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov: float, pixels: float) -> float:
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal: float, pixels: float) -> float:
    return 2 * math.atan(pixels / (2 * focal))
