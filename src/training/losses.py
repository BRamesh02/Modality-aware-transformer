import torch
import torch.nn as nn


class WeightedMSE(nn.Module):
    """
    Weighted Mean Squared Error.
    Penalizes errors on large targets more than errors on small targets.
    Formula: Loss = (pred - target)^2 * (1 + alpha * |target|)
    """

    def __init__(self, alpha=100.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, preds, targets):
        se = (preds - targets) ** 2
        weights = 1 + self.alpha * torch.abs(targets)

        return torch.mean(se * weights)
