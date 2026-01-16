import torch
import torch.nn as nn
from src.models.layers.masks import causal_mask


class DecoderLayer(nn.Module):
    """
    Standard Transformer decoder layer:
      1) Masked self-attention on tgt
      2) Cross-attention tgt -> memory (encoder)
      3) FFN
    """

    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask=None,
        mem_key_padding_mask=None,
    ) -> torch.Tensor:
        B, L, D = tgt.shape
        attn_mask = causal_mask(L, device=tgt.device)

        z = self.norm0(tgt)
        x, _ = self.self_attn(
            query=z,
            key=z,
            value=z,
            attn_mask=attn_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        tgt = tgt + self.drop(x)

        z = self.norm1(tgt)
        x, _ = self.cross_attn(
            query=z,
            key=memory,
            value=memory,
            key_padding_mask=mem_key_padding_mask,
            need_weights=False,
        )
        tgt = tgt + self.drop(x)

        z = self.norm2(tgt)
        x = self.ff(z)
        tgt = tgt + self.drop(x)
        return tgt


class CanonicalDecoder(nn.Module):
    def __init__(self, d_model=int, nhead=int, num_layers=int, dropout=float):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, nhead, dropout) for _ in range(num_layers)]
        )

    def forward(
        self, tgt, memory, tgt_key_padding_mask=None, mem_key_padding_mask=None
    ):
        for layer in self.layers:
            tgt = layer(
                tgt,
                memory,
                tgt_key_padding_mask=tgt_key_padding_mask,
                mem_key_padding_mask=mem_key_padding_mask,
            )
        return tgt
