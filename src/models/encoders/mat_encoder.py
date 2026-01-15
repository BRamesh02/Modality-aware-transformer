import torch
import torch.nn as nn
from src.models.layers.feature_attention import FeatureAttention
from src.models.layers.positional_encoding import PositionalEncoding


class MATEncoderLayer(nn.Module):
    """
    A single layer of the MAT Encoder that handles:
    1. Intra-Modal Attention (Self-Attention within stream)
    2. Inter-Modal Attention (Cross-Attention between streams)
    3. Feed Forward
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()

        self.self_attn_num = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.self_attn_text = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.norm1_num = nn.LayerNorm(d_model)
        self.norm1_text = nn.LayerNorm(d_model)

        self.cross_attn_num = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn_text = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.norm2_num = nn.LayerNorm(d_model)
        self.norm2_text = nn.LayerNorm(d_model)

        self.ff_num = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.ff_text = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.norm3_num = nn.LayerNorm(d_model)
        self.norm3_text = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_num, x_text):
        # --- Step 1: Self Attention (Intra) ---
        # Learn temporal patterns independently
        attn_n, _ = self.self_attn_num(x_num, x_num, x_num)
        x_num = self.norm1_num(x_num + self.dropout(attn_n))

        attn_t, _ = self.self_attn_text(x_text, x_text, x_text)
        x_text = self.norm1_text(x_text + self.dropout(attn_t))

        # --- Step 2: Cross Attention (Inter) ---
        # Exchange information between streams
        # Num Stream looks at Text Stream
        attn_n_cross, _ = self.cross_attn_num(query=x_num, key=x_text, value=x_text)
        x_num_mixed = self.norm2_num(x_num + self.dropout(attn_n_cross))

        # Text Stream looks at Num Stream
        attn_t_cross, _ = self.cross_attn_text(query=x_text, key=x_num, value=x_num)
        x_text_mixed = self.norm2_text(x_text + self.dropout(attn_t_cross))

        # --- Step 3: Feed Forward ---
        ff_n = self.ff_num(x_num_mixed)
        x_num_out = self.norm3_num(x_num_mixed + self.dropout(ff_n))

        ff_t = self.ff_text(x_text_mixed)
        x_text_out = self.norm3_text(x_text_mixed + self.dropout(ff_t))

        return x_num_out, x_text_out
    

class MATEncoder(nn.Module):
    def __init__(
        self,
        num_input_dim: int,
        sent_input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        sent_dim: int = 32,
        emb_dim: int = 96,
        use_emb: bool = True,
    ):
        super().__init__()

        self.use_emb = use_emb

        self.num_feat_attn = FeatureAttention(num_input_dim)
        text_feat_dim = sent_dim + (emb_dim if self.use_emb else 0)
        self.text_feat_attn = FeatureAttention(text_feat_dim)

        self.num_proj = nn.Sequential(
            nn.Linear(num_input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.sent_proj = nn.Sequential(
            nn.Linear(sent_input_dim, sent_dim),
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
            text_in_dim = sent_dim + emb_dim
        else:
            self.emb_proj = None
            text_in_dim = sent_dim

        self.text_proj = nn.Sequential(
            nn.Linear(text_in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.layers = nn.ModuleList(
            [
                MATEncoderLayer(d_model, nhead, d_model * 4, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x_num, x_sent, x_emb=None):
        t_sent = self.sent_proj(x_sent)  # [B,T,sent_dim]

        if self.use_emb:
            if x_emb is None:
                raise ValueError("MATEncoder(use_emb=True) requires x_emb, got None.")
            t_emb = self.emb_proj(x_emb)  # [B,T,emb_dim]
            x_text = torch.cat([t_sent, t_emb], dim=-1)  # [B,T,sent_dim+emb_dim]
        else:
            x_text = t_sent  # [B,T,sent_dim]

        x_num, _ = self.num_feat_attn(x_num)
        x_text, _ = self.text_feat_attn(x_text)

        h_num = self.num_proj(x_num)
        h_text = self.text_proj(x_text)

        h_num = self.pos_encoder(h_num.transpose(0, 1)).transpose(0, 1)
        h_text = self.pos_encoder(h_text.transpose(0, 1)).transpose(0, 1)

        for layer in self.layers:
            h_num, h_text = layer(h_num, h_text)

        return h_num, h_text