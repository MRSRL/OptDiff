"""Metrics for MRI and natural images.

All metrics take a reference image (ref) and an output/reconstructed image (pred), both with an initial batch
dimension: complex valued for MRI, real valued [N, 3, H, W] in [0, 1] for natural images (see the scale argument of
psnr and the magnitude and channel_axis arguments of ssim).
"""

import torch
from skimage.metrics import structural_similarity
import lpips

class LPIPSMetric:
    def __init__(self, net='alex', device='cuda'):
        self.device = device
        self.loss_fn = lpips.LPIPS(net=net).to(device)
        self.loss_fn.eval()

    @torch.no_grad()
    def __call__(self, img1, img2):
        """
        img1, img2: tensors of shape (B, C, H, W), range [0, 1]
        """
        # Convert to [-1, 1]
        img1 = img1 * 2 - 1
        img2 = img2 * 2 - 1
        dist = self.loss_fn(img1, img2)  # (B, 1, 1, 1)
        return dist.mean().item()
    
def mse(ref: torch.Tensor, pred: torch.Tensor):
    """Mean squared error.

    Args:
        ref (torch.Tensor): The target. Shape [N,...] of complex or real tensors.
        x (torch.Tensor): The prediction. Same shape as `ref`.
    
    Returns:
        torch.Tensor: The MSE loss for each ref, pred pair element in the batch. Shape [N].
        
    """
    return torch.mean(torch.abs(ref - pred)**2, dim=tuple(range(1, ref.ndim)))

def rmse(ref: torch.Tensor, pred: torch.Tensor):
    """Root Mean squared error.

    Args:
        ref (torch.Tensor): The target. Shape [N,...] of complex or real tensors.
        x (torch.Tensor): The prediction. Same shape as `ref`.
    
    Returns:
        torch.Tensor: The RMSE loss for each ref, pred pair element in the batch. Shape [N].
        
    """
    return torch.sqrt(mse(ref, pred))

def nrmse(ref: torch.Tensor, pred: torch.Tensor):
    """Normalized root mean square error.
    The norm of reference is used to normalize the metric.

    Args:
        ref (torch.Tensor): The target. Shape [N,...] of complex or real tensors.
        pred (torch.Tensor): The prediction. Same shape as `ref`.
    
    Returns:
        torch.Tensor: The NRMSE for each ref, pred pair in the batch. Shape [N].
    """
    trmse = rmse(ref, pred)
    norm = torch.sqrt(torch.mean(torch.abs(ref)**2, dim=tuple(range(1, ref.ndim))))
    return trmse / norm
    

def l2(ref: torch.Tensor, pred: torch.Tensor):
    """L2 distance between reference and prediction.

    Args:
        ref (torch.Tensor): The target. Shape [N,...] of complex or real tensors.
        pred (torch.Tensor): The prediction. Same shape as `ref`.
    
    Returns:
        torch.Tensor: The L2 distance for each ref, pred pair in the batch. Shape [N].
    """
    return torch.sum(torch.abs(ref - pred)**2, dim=tuple(range(1, ref.ndim)))



def l1(ref: torch.Tensor, pred: torch.Tensor):
    """L1 loss.

    Args:
        ref (torch.Tensor): The target. Shape [N,...] of complex or real tensors.
        pred (torch.Tensor): The prediction. Same shape as `ref`.
    
    Returns:
        torch.Tensor: The L1 distance for each ref, pred pair in the batch. Shape [N].
    """
    return torch.mean(torch.abs(ref - pred), dim=tuple(range(1, ref.ndim)))


def psnr(ref: torch.Tensor, pred: torch.Tensor, scale=None):
    """Complex peak signal-to-noise ratio (PSNR)

    Args:
        ref (torch.Tensor): The target. Shape [N,...] of complex or real tensors.
        x (torch.Tensor): The prediction. Same shape as `ref`.
        scale (float, optional): The peak value. If None, the largest magnitude in `ref`. Defaults to None.

    Returns:
        torch.Tensor: The PSNR for each ref, pred pair in the batch. Shape [N].
    """
    if scale is None:
        scale = torch.abs(ref).max()
    return 20 * torch.log10(scale / torch.sqrt(mse(ref, pred)))

def psnr_abs(ref: torch.Tensor, pred: torch.Tensor):
    """Complex peak signal-to-noise ratio (PSNR)

    Args:
        ref (torch.Tensor): The target. Shape [N,...] of complex or real tensors.
        x (torch.Tensor): The prediction. Same shape as `ref`.
    
    Returns:
        torch.Tensor: The PSNR for each ref, pred pair in the batch. Shape [N].
    """
    ref = torch.abs(ref).detach()
    pred = torch.abs(pred).detach()
    scale = torch.abs(ref).max()
    return 20 * torch.log10(scale / torch.sqrt(mse(ref, pred)))


def ssim(ref: torch.Tensor, pred: torch.Tensor, data_range=None, magnitude=True, channel_axis=None):
    """Structural similarity index (SSIM).
    Matches the standard SSIM implementation from: Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004)

    Args:
        ref (torch.Tensor): The target. Shape [N,...] of complex or real tensors.
        pred (torch.Tensor): The prediction. Same shape as `ref`.
        data_range (float, optional): The range of the data. If None, it is computed as the range of the input data. Defaults to None.
        magnitude (bool, optional): Convert the tensors to real values by taking the magnitude first (complex MRI
            images). Defaults to True.
        channel_axis (int, optional): The channel axis of each image, e.g. 0 for [N, 3, H, W] natural images. None for
            single-channel images. Defaults to None.

    Returns:
        torch.Tensor: The SSIM for each ref, pred pair in the batch. Shape [N].
    """

    if magnitude:
        ref = torch.abs(ref)
        pred = torch.abs(pred)
    ref = ref.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    if data_range is None:
        data_range = ref.max() - ref.min()
    return torch.tensor([structural_similarity(ref[i], pred[i], data_range=data_range, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, channel_axis=channel_axis) for i in range(ref.shape[0])], dtype=torch.float32)

