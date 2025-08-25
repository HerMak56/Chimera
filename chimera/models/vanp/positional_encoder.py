import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max_len, d_model), requires_grad=True
        )

    def forward(self, x: torch.Tensor):
        # calculate the positional encoding
        return x + self.positional_encoding