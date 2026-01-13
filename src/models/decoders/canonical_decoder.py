# src/models/layers/positional_encoding.py

import torch
import torch.nn as nn
import math
from src.models.layers.masks import causal_mask


class ClassicDecoderLayer(nn.Module):
    """
    Decoder Transformer classique:
      1) Masked self-attention sur tgt
      2) Cross-attention tgt -> memory (encoder)
      3) FFN
    Post-norm (comme ton MAT)
    """
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

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

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_key_padding_mask=None, mem_key_padding_mask=None) -> torch.Tensor:
        B, L, D = tgt.shape
        attn_mask = causal_mask(L, device=tgt.device)

        # 1) masked self-attn
        x, _ = self.self_attn(
            query=tgt, key=tgt, value=tgt,
            attn_mask=attn_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        tgt = tgt + self.drop(x)
        # tgt = self.norm0(tgt + self.drop(x)) a mettre si on veut se mettre en post norm et s'aligner avec MAT
        # penser à le faire avec encoder aussi dans ce cas norm_first=False

        # 2) cross-attn
        x, _ = self.cross_attn(
            query=tgt, key=memory, value=memory,
            key_padding_mask=mem_key_padding_mask,
            need_weights=False,
        )
        tgt = tgt + self.drop(x)
        # tgt = self.norm1(tgt + self.drop(x))

        # 3) FFN
        x = self.ff(tgt)
        tgt = tgt + self.drop(x)
        # tgt = self.norm2(tgt + self.drop(x))
        return tgt


class ClassicDecoder(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList([ClassicDecoderLayer(d_model, nhead, dropout) for _ in range(num_layers)])

    def forward(self, tgt, memory, tgt_key_padding_mask=None, mem_key_padding_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        mem_key_padding_mask=mem_key_padding_mask)
        return tgt
    