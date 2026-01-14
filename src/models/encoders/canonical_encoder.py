import torch
import torch.nn as nn
import math
from src.models.layers.positional_encoding import PositionalEncoding

class CanonicalEncoder(nn.Module):
    def __init__(
        self,
        num_input_dim: int,
        #text_input_dim: int,
        n_sent: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        sent_dim: int = 32,
        emb_dim: int = 96,
        use_emb: bool = True,
    ):
        """
        The Baseline: A Standard Transformer with Early Fusion.
        """
        super().__init__()

        # --- 1. Balanced Projection (Early Fusion) ---
        # We project both inputs to d_model/2 so they sum to d_model when concatenated.
        # This gives equal bandwidth to both modalities, just like MAT.
        half_dim = d_model // 2

        self.use_emb = use_emb

        self.num_proj = nn.Sequential(
            nn.Linear(num_input_dim, half_dim),
            #nn.BatchNorm1d(60),
            nn.LayerNorm(half_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # self.text_proj = nn.Sequential(
        #     nn.Linear(text_input_dim, half_dim),
        #     nn.BatchNorm1d(60),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        # )

        self.sent_proj = nn.Sequential(
            nn.Linear(n_sent, sent_dim),
            nn.LayerNorm(sent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if self.use_emb:
            self.emb_proj = nn.Sequential(
                nn.Linear(768, emb_dim),
                nn.LayerNorm(emb_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            fuse_in = sent_dim + emb_dim
        else:
            self.emb_proj = None
            fuse_in = sent_dim

        self.text_fuse = nn.Sequential(
            nn.Linear(fuse_in, half_dim),
            nn.LayerNorm(half_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --- 2. Positional Encoding ---
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # --- 3. The Transformer Backbone ---
        # Standard PyTorch Transformer implementation.
        # It treats the fused vector [Price_Info, Text_Info] as one single concept.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Put False here if we want to align with the MAT
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x_num, x_sent, x_emb=None):
        """
        Args:
            x_num:  [Batch, T, num_input_dim]
            x_sent: [Batch, T, n_sent]
            x_emb:  [Batch, T, 768]
        Returns:
            memory: [Batch, 60, 128] (Single stream output)
        """
        # 1. Project Modalities independently
        x_n = self.num_proj(x_num)  # [Batch, 60, 64]
        # x_t = self.text_proj(x_text)  # [Batch, 60, 64]
        t_sent = self.sent_proj(x_sent)              # [B,T,32]
        # t_emb  = self.emb_proj(x_emb)                # [B,T,96]
        # x_t    = self.text_fuse(torch.cat([t_sent, t_emb], dim=-1))  # [B,60,64]

        if self.use_emb:
            if x_emb is None:
                raise ValueError("CanonicalEncoder(use_emb=True) requires x_emb, got None.")
            t_emb = self.emb_proj(x_emb)  # [B,T,emb_dim]
            t_in = torch.cat([t_sent, t_emb], dim=-1)
        else:
            # embeddings disabled: sentiment scalars only
            t_in = t_sent

        x_t = self.text_fuse(t_in)  # [B,T,half_dim]

        # 2. Early Concatenation (The key difference from MAT)
        # We fuse them immediately. The transformer now sees one vector of size 128.
        x_combined = torch.cat([x_n, x_t], dim=2)  # [Batch, 60, 128]

        # 3. Add Positional Encoding
        # (Transpose required for our PE implementation)
        x_combined = x_combined.transpose(0, 1)  # [60, Batch, 128]
        x_combined = self.pos_encoder(x_combined)
        x_combined = x_combined.transpose(0, 1)  # [Batch, 60, 128]

        # 4. Process with Standard Transformer
        memory = self.transformer(x_combined)

        return memory