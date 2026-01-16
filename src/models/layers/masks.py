import torch


def causal_mask(sz: int, device=None) -> torch.Tensor:
    """
    Causal (look-ahead) mask: prevents position i from attending to j > i.
    Shape expected by nn.MultiheadAttention (batch_first=True): [sz, sz]
    """
    m = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
    # PyTorch MHA expects float mask with -inf (or bool in newer versions).
    # We'll use float for maximal compatibility.
    mask = torch.zeros(sz, sz, device=device)
    mask = mask.masked_fill(m, float("-inf"))
    return mask
