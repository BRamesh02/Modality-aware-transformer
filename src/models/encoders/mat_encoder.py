import torch
import torch.nn as nn
from src.models.layers.feature_attention import FeatureAttention
from src.models.layers.positional_encoding import PositionalEncoding


class MATEncoderLayer(nn.Module):
    """
    A single layer of the MAT Encoder using Pre-Norm Architecture.
    Order: Input -> Norm -> Attention -> Dropout -> Residual Add
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()

        # 1. Self Attention
        self.self_attn_num = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.self_attn_text = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1_num = nn.LayerNorm(d_model)
        self.norm1_text = nn.LayerNorm(d_model)

        # 2. Cross Attention
        self.cross_attn_num = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn_text = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm2_num = nn.LayerNorm(d_model)
        self.norm2_text = nn.LayerNorm(d_model)

        # 3. Feed Forward
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
        # Step 1: Self Attention (Pre-Norm)
        # Normalize INPUT, then attend, then add to original input

        # Num Stream
        x_n_norm = self.norm1_num(x_num)
        attn_n, _ = self.self_attn_num(x_n_norm, x_n_norm, x_n_norm)
        x_num = x_num + self.dropout(attn_n)

        # Text Stream
        x_t_norm = self.norm1_text(x_text)
        attn_t, _ = self.self_attn_text(x_t_norm, x_t_norm, x_t_norm)
        x_text = x_text + self.dropout(attn_t)

        # --- Step 2: Cross Attention (Pre-Norm) ---
        # Normalize residual input before Cross Attn
        x_n_norm = self.norm2_num(x_num)
        x_t_norm = self.norm2_text(x_text)

        # Num looks at Text (Query=Num, Key/Value=Text)
        # Note: We use the normalized version of Text as keys/values for stability
        attn_n_cross, _ = self.cross_attn_num(
            query=x_n_norm, key=x_t_norm, value=x_t_norm
        )
        x_num = x_num + self.dropout(attn_n_cross)

        # Text looks at Num
        attn_t_cross, _ = self.cross_attn_text(
            query=x_t_norm, key=x_n_norm, value=x_n_norm
        )
        x_text = x_text + self.dropout(attn_t_cross)

        # Step 3: Feed Forward (Pre-Norm)
        x_n_norm = self.norm3_num(x_num)
        ff_n = self.ff_num(x_n_norm)
        x_num = x_num + self.dropout(ff_n)

        x_t_norm = self.norm3_text(x_text)
        ff_t = self.ff_text(x_t_norm)
        x_text = x_text + self.dropout(ff_t)

        return x_num, x_text


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

        # Feature Attention
        self.num_feat_attn = FeatureAttention(num_input_dim)
        text_feat_dim = sent_dim + (emb_dim if self.use_emb else 0)
        self.text_feat_attn = FeatureAttention(text_feat_dim)

        # Projections
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

        # Stack Layers
        self.layers = nn.ModuleList(
            [
                MATEncoderLayer(d_model, nhead, d_model * 4, dropout)
                for _ in range(num_layers)
            ]
        )

        self.norm_final_num = nn.LayerNorm(d_model)
        self.norm_final_text = nn.LayerNorm(d_model)

    def forward(self, x_num, x_sent, x_emb=None):
        # Prepare Text Input
        t_sent = self.sent_proj(x_sent)  # [B,T,sent_dim]

        if self.use_emb:
            if x_emb is None:
                raise ValueError("MATEncoder(use_emb=True) requires x_emb, got None.")
            t_emb = self.emb_proj(x_emb)  # [B,T,emb_dim]
            x_text = torch.cat([t_sent, t_emb], dim=-1)  # [B,T,sent_dim+emb_dim]
        else:
            x_text = t_sent

        # Feature Attention
        x_num, _ = self.num_feat_attn(x_num)
        x_text, _ = self.text_feat_attn(x_text)

        # Project to d_model
        h_num = self.num_proj(x_num)
        h_text = self.text_proj(x_text)

        # Positional Encoding
        h_num = self.pos_encoder(h_num.transpose(0, 1)).transpose(0, 1)
        h_text = self.pos_encoder(h_text.transpose(0, 1)).transpose(0, 1)

        # Pass through Layers
        for layer in self.layers:
            h_num, h_text = layer(h_num, h_text)

        # Final Normalization (Crucial for Pre-Norm)
        h_num = self.norm_final_num(h_num)
        h_text = self.norm_final_text(h_text)

        return h_num, h_text
