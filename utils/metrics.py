import math

import torch
import torch.nn.functional as F
from lpips import LPIPS
perc_loss_net = None


def to_BCHW(*tensors):
    out = []
    for t in tensors:
        if t is None:
            out.append(None)
        else:
            assert t.ndim == 4
            if t.shape[1] > 4: # detect BHWC
                out.append(t.permute(0,3,1,2)) # to BCHW
            else:
                out.append(t)
    return tuple(out)

def img_mse(pred, gt, mask=None, error_type='mse', return_all=False, use_mask=False):
    """
    MSE and variants
    Input:
        pred        :  bsize x 3 x h x w
        gt          :  bsize x 3 x h x w
        error_type  :  'mse' | 'rmse' | 'mae' | 'L21'
    MSE/RMSE/MAE between predicted and ground-truth images.
    Returns one value per-batch element
    pred, gt: bsize x 3 x h x w
    """
    assert pred.dim() == 4
    # Ensure shape is BCHW
    pred, gt, mask = to_BCHW(pred, gt, mask)

    bsize = pred.size(0)

    if error_type == 'mae':
        all_errors = (pred-gt).abs()
    elif error_type == 'L21':
        all_errors = torch.norm(pred-gt, dim=1)
    elif error_type == "L1":
        all_errors = torch.norm(pred - gt, dim=1, p=1)
    else:
        all_errors = (pred-gt).square()

    if mask is not None and use_mask:
        assert mask.size(1) == 1

        nc = pred.size(1)
        all_errors = mask.expand(-1, nc, -1, -1) * all_errors
        errors = all_errors.reshape(bsize, -1).sum(1) / gt.nelement()
    else:
        errors = all_errors.reshape(bsize, -1).mean(1)

    if error_type == 'rmse':
        errors = errors.sqrt()

    if return_all:
        return errors, all_errors
    else:
        return errors

def img_psnr(pred, gt, mask=None, rmse=None):
    # https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py
    if torch.max(pred) > 128:   max_val = 255.
    else:                       max_val = 1.

    if rmse is None:
        rmse = img_mse(pred, gt, mask, error_type='rmse', use_mask=mask is not None)

    EPS = 1e-8
    return 20 * torch.log10(max_val / (rmse+EPS))

def _gaussian(window_size, sigma):
    gauss = torch.tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def _create_window(window_size, channel):
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def img_ssim(img1, img2, window_size=11):
    # https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py
    
    # Ensure shape is BCHW
    img1, img2 = to_BCHW(img1, img2)

    if img1.min() < -0 or img1.max() > 255:
       raise ValueError(f"Found values outside of [0,255] range while computing SSIM - this is probably not intended. Found min: {img1.min().item():.4f}, max: {img1.max().item():.4f}")
    if img2.min() < -0 or img2.max() > 255:
       raise ValueError(f"Found values outside of [0,255] range while computing SSIM - this is probably not intended. Found min: {img2.min().item():.4f}, max: {img2.max().item():.4f}")

    if torch.max(img1) > 128: min_val, max_val = 0, 255
    else:                     min_val, max_val = 0, 1

    L = max_val - min_val

    pad = window_size // 2
    channel = img1.size(-3)
    window = _create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    # cs = torch.mean(v1 / v2) # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    return ssim_map.mean(1).mean(1).mean(1) # (B,)


def lpips(pred, gt, with_grad=False):
    """ https://richzhang.github.io/PerceptualSimilarity/ """

    with torch.set_grad_enabled(with_grad):
        pred, gt = to_BCHW(pred, gt)
        assert pred.dim() == 4
        assert gt.dim() == 4

        global perc_loss_net
        if perc_loss_net is None:
            # perc_loss_net = lpips.LPIPS(net="alex").to(pred.device)
            perc_loss_net = LPIPS(net="vgg").to(pred.device)

        # return perc_loss_net(pred, gt, normalize=True)
        return perc_loss_net(pred, gt)
