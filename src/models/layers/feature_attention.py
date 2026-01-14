import torch
import torch.nn as nn

class FeatureAttention(nn.Module):
    """
    Section 3.2.1: Feature-Level Attention.
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
        # 1. Compute Importance Scores [Batch, Seq, Input_Dim]
        weights = self.attn(x)

        # 2. Re-weight the input
        x_weighted = x * weights

        # 3. Return BOTH (So we can reuse weights later)
        return x_weighted, weights
