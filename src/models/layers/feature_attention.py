import torch
import torch.nn as nn


class FeatureAttention(nn.Module):
    """
    Feature-Level Attention.
    Now returns BOTH the weighted input and the importance scores.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        weights = self.attn(x)
        x_weighted = x * weights
        return x_weighted, weights
