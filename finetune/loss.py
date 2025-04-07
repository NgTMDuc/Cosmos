import torch
import torch.nn.functional as F
import pytorch_msssim

class SSIM3D(torch.nn.Module):
    def __init__(self, window_size=11, reduction='mean'):
        super(SSIM3D, self).__init__()
        self.ssim = pytorch_msssim.SSIM(window_size=window_size, reduction=reduction)

    def forward(self, img1, img2):
        # img1 and img2 have shape [B, C, D, H, W]
        ssim_values = []
        
        # Loop through depth slices
        for i in range(img1.shape[2]):  # img1.shape[2] is depth (D)
            slice_img1 = img1[:, :, i, :, :]
            slice_img2 = img2[:, :, i, :, :]
            ssim_val = self.ssim(slice_img1, slice_img2)
            ssim_values.append(ssim_val)
        
        return torch.mean(torch.stack(ssim_values))