from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class VICRegLoss(nn.Module):
    """Variance-Invariance-Covariance Regularization (VICReg).

    L = λ_inv * MSE(z1,z2) + λ_var * (var(z1)+var(z2)) + λ_cov * (cov(z1)+cov(z2))
    """
    def __init__(self, inv: float = 25.0, var: float = 25.0, cov: float = 1.0, eps: float = 1e-4) -> None:
        super().__init__()
        self.w_inv, self.w_var, self.w_cov, self.eps = inv, var, cov, eps

    @staticmethod
    def _variance_term(z: torch.Tensor, eps: float) -> torch.Tensor:
        std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
        return torch.relu(1.0 - std).mean()

    @staticmethod
    def _covariance_term(z: torch.Tensor) -> torch.Tensor:
        z = z - z.mean(dim=0)
        n = z.shape[0]
        c = (z.T @ z) / (n - 1)
        off = c - torch.diag(torch.diag(c))
        return (off.pow(2).sum()) / z.shape[1]

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        inv = F.mse_loss(z1, z2)
        var = self._variance_term(z1, self.eps) + self._variance_term(z2, self.eps)
        cov = self._covariance_term(z1) + self._covariance_term(z2)
        return self.w_inv * inv + self.w_var * var + self.w_cov * cov