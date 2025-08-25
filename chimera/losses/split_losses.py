from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class VICRegLoss(nn.Module):
    """
    Variance–Invariance–Covariance Regularization (VICReg).

    L = inv * MSE(z1, z2)
      + var * [ mean(ReLU(1 - std(z1))) + mean(ReLU(1 - std(z2))) ]
      + cov * [ offdiag(Cov(z1)).^2.mean_dim + offdiag(Cov(z2)).^2.mean_dim ]

    Parameters
    ----------
    inv : float
        Weight for invariance term (MSE).
    var : float
        Weight for variance term (std >= 1 via ReLU barrier).
    cov : float
        Weight for covariance term (feature-wise redundancy reduction).
    eps : float
        Small jitter added before sqrt for numerical stability.
    """

    def __init__(self, inv: float = 25.0, var: float = 25.0, cov: float = 1.0, eps: float = 1e-4) -> None:
        super().__init__()
        self.w_inv, self.w_var, self.w_cov, self.eps = inv, var, cov, eps

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    def __call__(self, z1: torch.Tensor, z2: torch.Tensor):
        repr_loss = F.mse_loss(z1, z2)

        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)

        std_z1 = torch.sqrt(z1.var(dim=0) + self.eps)
        std_z2 = torch.sqrt(z2.var(dim=0) + self.eps)

        std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))

        cov_z1 = (z1.T @ z1)  / (z1.shape[0] - 1)
        cov_z2 = (z2.T @ z2)  / (z2.shape[0] - 1)

        cov_loss = self.off_diagonal(cov_z1).pow(2).sum() + self.off_diagonal(cov_z2).pow(2).sum()

        loss = self.w_inv * repr_loss + self.w_var * std_loss + self.w_cov * cov_loss
        return loss, {"name": "VICReg", "repr": repr_loss, "std": std_loss, "cov": cov_loss}