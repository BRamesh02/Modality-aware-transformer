import torch
import torch.nn as nn
from src.models.layers.masks import causal_mask


class MATDecoderLayer(nn.Module):
    """
    MAT decoder layer aligned with paper Sec. 3.2.3:

    Sub-layer 1) Masked MHA over target sequence (autoregressive)
        - Eq. (3.17) Attn^masked = softmax(Q_tar K_tar^T / sqrt(d_k)) V_tar
        - Eq. (3.18) Head computation
        - Eq. (3.19) Masked MHA = Concat(H_1,...,H_h) W^O_tar

    Sub-layer 2) Target-modal MHA (target queries attend to each modality memory)
        - Eq. (3.20) Attn^{tar-mod}_{mod_i} = softmax(Q_tar K_mod_i^T / sqrt(d_k)) V_mod_i
        - Eq. (3.21) Head computation
        - Eq. (3.22) Target-modal MHA = Concat(H_1,...,H_h) W^O_{tar,mod_i}

    Sub-layer 3) Feed-forward (standard Transformer FFN)

    Note: we use post-norm style: residual + LayerNorm after each sub-layer,
    matching the text "residual connections, following by layer normalization".
    """

    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()

        # Sub-layer 1: masked self-attn on target sequence (Eq. 3.17-3.19)
        self.self_attn_tar = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Sub-layer 2: target-modal cross-attn to each modality (Eq. 3.20-3.22)
        self.cross_num = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_text = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Fuse (Q_tar, ctx_num, ctx_text) -> d_model
        self.fuse = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Sub-layer 3: FFN
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

        self.norm0 = nn.LayerNorm(d_model)  # after masked MHA
        self.norm1 = nn.LayerNorm(d_model)  # after target-modal fusion
        self.norm2 = nn.LayerNorm(d_model)  # after FFN
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,  # [B, H, D]
        mem_num: torch.Tensor,  # [B, Tn, D]
        mem_text: torch.Tensor,  # [B, Tt, D]
        tgt_key_padding_mask: torch.Tensor | None = None,  # [B, H] optional
        mem_num_key_padding_mask: torch.Tensor | None = None,  # [B, Tn] optional
        mem_text_key_padding_mask: torch.Tensor | None = None,  # [B, Tt] optional
    ) -> torch.Tensor:
        B, H, D = tgt.shape
        device = tgt.device

        # Sub-layer 1: Masked MHA on target sequence (Eq. 3.17-3.19)
        attn_mask = causal_mask(H, device=device)
        tgt_ctx, _ = self.self_attn_tar(
            query=tgt,
            key=tgt,
            value=tgt,
            attn_mask=attn_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        tgt = self.norm0(tgt + self.drop(tgt_ctx))

        # Sub-layer 2: Target-modal MHA (Eq. 3.20-3.22)
        # Query = Q_tar from masked MHA output (tgt)
        # Key/Value = modality memory (already feature-weighted by encoder)

        ctx_n, _ = self.cross_num(
            query=tgt,
            key=mem_num,
            value=mem_num,
            key_padding_mask=mem_num_key_padding_mask,
            need_weights=False,
        )
        ctx_t, _ = self.cross_text(
            query=tgt,
            key=mem_text,
            value=mem_text,
            key_padding_mask=mem_text_key_padding_mask,
            need_weights=False,
        )

        mixed = self.fuse(torch.cat([tgt, ctx_n, ctx_t], dim=-1))
        tgt = self.norm1(tgt + self.drop(mixed))

        # Sub-layer 3: FFN (standard Transformer)
        ff = self.ff(tgt)
        tgt = self.norm2(tgt + self.drop(ff))

        return tgt


class MATDecoder(nn.Module):
    def __init__(self, d_model=int, nhead=int, num_layers=int, dropout=float):
        super().__init__()
        self.layers = nn.ModuleList(
            [MATDecoderLayer(d_model, nhead, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        tgt,
        mem_num,
        mem_text,
        tgt_key_padding_mask=None,
        mem_num_key_padding_mask=None,
        mem_text_key_padding_mask=None,
    ):
        for layer in self.layers:
            tgt = layer(
                tgt,
                mem_num,
                mem_text,
                tgt_key_padding_mask=tgt_key_padding_mask,
                mem_num_key_padding_mask=mem_num_key_padding_mask,
                mem_text_key_padding_mask=mem_text_key_padding_mask,
            )
        return tgt
