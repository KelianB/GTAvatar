import torch
from torch import Tensor
from roma import quat_product, rotmat_to_unitquat


def inverse_sigmoid(x: Tensor) -> Tensor:
    return torch.log(x/(1-x))

def gaussian_kernel(ksize: int, sigma: float) -> Tensor:
    x = torch.arange(0, ksize) - ksize // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma * sigma))
    return gauss / gauss.sum()

def apply_featurewise_conv1d(signal: Tensor, kernel: Tensor, pad_mode="replicate") -> Tensor:
    # signal: (N, n_features)
    # kernel: (kernel_size)
    _, n_features = signal.shape
    kernel_size = kernel.shape[0]
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(n_features,1,1) # (n_features,1,kernel_size) (we want as many output channels as features)
    signal = signal.permute(1,0).unsqueeze(0) # (1, n_features, N)
    # Pad input signal
    padding = kernel_size // 2 # to maintain the original signal length
    padded_signal = torch.nn.functional.pad(signal, (padding, padding), mode=pad_mode)
    # Perform convolution
    filtered_signal = torch.nn.functional.conv1d(padded_signal, kernel, groups=n_features) # (1, n_features, N)
    return filtered_signal.squeeze(0).permute(1,0) # (N, n_features)


def build_rotation(q: Tensor) -> Tensor:
    norm = torch.sqrt(q[:,0]*q[:,0] + q[:,1]*q[:,1] + q[:,2]*q[:,2] + q[:,3]*q[:,3])

    q = q / norm[:, None]
    r, x, y, z = q.unbind(1)

    R = torch.zeros((q.size(0), 3, 3), device=r.device)
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s: Tensor, r: Tensor) -> Tensor:
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]
    R = build_rotation(r)
    return R @ L

def _xyzw_to_wxyz(xyzw: Tensor) -> Tensor:
    assert xyzw.shape[-1] == 4
    return xyzw[..., [3,0,1,2]]

def _wxyz_to_xyzw(wxyz: Tensor) -> Tensor:
    assert wxyz.shape[-1] == 4
    return wxyz[..., [1,2,3,0]]

def quat_product_wxyz(p: Tensor, q: Tensor) -> Tensor:
    return _xyzw_to_wxyz(quat_product(_wxyz_to_xyzw(p), _wxyz_to_xyzw(q)))

def rotmat_to_unitquat_wxyz(R: Tensor) -> Tensor:
    return _xyzw_to_wxyz(rotmat_to_unitquat(R))
