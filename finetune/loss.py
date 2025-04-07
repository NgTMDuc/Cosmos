import torch
import torch.nn.funcitonal as F
import torch.nn as nn
import math

def create_gaussian_kernel_3d(kernel_size = 11, sigma = 1.5, channels = 1):
    def gauss_1d(size, sigma):
        coords = torch.arange(size).float() - size // 2
        return torch.exp(-(coords ** 2) / (2 * sigma ** 2))

    gauss_1d_kernel = gauss_1d(kernel_size, sigma)    
    kernel_3d = gauss_1d_kernel[:, None, None] * gauss_1d_kernel[None, :, None] * gauss_1d_kernel[None, None, :]
    kernel_3d = kernel_3d / kernel_3d.sum()
    kernel_3d = kernel_3d.view(1, 1, kernel_size, kernel_size, kernel_size)
    kernel_3d = kernel_3d.repeat(channels, 1, 1, 1, 1)
    return kernel_3d

class SSIM3DLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, channels=1, reduction='mean'):
        super(SSIM3DLoss, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channels = channels
        self.reduction = reduction
        self.kernel = create_gaussian_kernel_3d(window_size, sigma, channels)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        device = x.device
        kernel = self.kernel.to(device)

        mu_x = F.conv3d(x, kernel, padding=self.window_size//2, groups=self.channels)
        mu_y = F.conv3d(y, kernel, padding=self.window_size//2, groups=self.channels)

        mu_x2 = mu_x ** 2
        mu_y2 = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv3d(x * x, kernel, padding=self.window_size//2, groups=self.channels) - mu_x2
        sigma_y2 = F.conv3d(y * y, kernel, padding=self.window_size//2, groups=self.channels) - mu_y2
        sigma_xy = F.conv3d(x * y, kernel, padding=self.window_size//2, groups=self.channels) - mu_xy

        numerator = (2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)
        denominator = (mu_x2 + mu_y2 + self.C1) * (sigma_x2 + sigma_y2 + self.C2)

        ssim_map = numerator / (denominator + 1e-8)

        if self.reduction == 'mean':
            return 1 - ssim_map.mean()
        elif self.reduction == 'sum':
            return 1 - ssim_map.sum()
        else:
            return 1 - ssim_map