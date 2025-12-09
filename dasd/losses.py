"""
Loss functions for Domain-Adaptive Stable Diffusion (DASD).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MMDAlignmentLoss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) Loss for domain alignment.

    Aligns source and target feature distributions by minimizing
    the distance between their mean embeddings in a reproducing
    kernel Hilbert space (RKHS).

    Args:
        bandwidth: Kernel bandwidth for Gaussian RBF kernel
    """

    def __init__(self, bandwidth=1.0):
        super().__init__()
        self.bandwidth = bandwidth

    def gaussian_kernel(self, x, y):
        """Compute Gaussian RBF kernel between two sets of samples.

        Args:
            x: Source features (N, D)
            y: Target features (M, D)

        Returns:
            Kernel matrix (N, M)
        """
        x_norm = (x ** 2).sum(1, keepdim=True)
        y_norm = (y ** 2).sum(1, keepdim=True)
        dist = x_norm + y_norm.T - 2 * torch.mm(x, y.T)
        return torch.exp(-dist / (2 * self.bandwidth ** 2))

    def forward(self, source_feat, target_feat):
        """Compute MMD loss between source and target features.

        Args:
            source_feat: Source domain features
            target_feat: Target domain features

        Returns:
            MMD loss value
        """
        # Flatten features if needed
        if source_feat.dim() > 2:
            source_feat = source_feat.view(source_feat.size(0), -1)
            target_feat = target_feat.view(target_feat.size(0), -1)

        # Normalize features
        source_feat = F.normalize(source_feat, p=2, dim=1)
        target_feat = F.normalize(target_feat, p=2, dim=1)

        # Compute kernel matrices
        K_ss = self.gaussian_kernel(source_feat, source_feat)
        K_tt = self.gaussian_kernel(target_feat, target_feat)
        K_st = self.gaussian_kernel(source_feat, target_feat)

        # Compute MMD
        mmd = K_ss.mean() + K_tt.mean() - 2 * K_st.mean()

        return torch.sqrt(torch.clamp(mmd, min=1e-8))


class FeatureExtractor:
    """
    Extracts intermediate features from UNet mid-block for alignment.

    Registers a forward hook on the UNet's mid_block to capture
    features during the forward pass.
    """

    def __init__(self, unet):
        """Initialize feature extractor with hook on mid-block.

        Args:
            unet: UNet model to extract features from
        """
        self.features = {}
        self._hook = unet.mid_block.register_forward_hook(
            lambda module, input, output: self.features.update({"mid": output})
        )

    def get(self):
        """Get extracted mid-block features."""
        return self.features.get("mid")

    def clear(self):
        """Clear stored features."""
        self.features = {}

    def remove_hook(self):
        """Remove the registered hook."""
        self._hook.remove()
