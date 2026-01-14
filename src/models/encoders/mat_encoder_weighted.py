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

        # --- 1. Intra-Modal (Self Attention) ---
        self.self_attn_num = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.self_attn_text = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.norm1_num = nn.LayerNorm(d_model)
        self.norm1_text = nn.LayerNorm(d_model)

        # --- 2. Inter-Modal (Cross Attention) ---
        # Num queries Text, Text queries Num
        self.cross_attn_num = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn_text = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.norm2_num = nn.LayerNorm(d_model)
        self.norm2_text = nn.LayerNorm(d_model)

        # --- 3. Feed Forward ---
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
    

class MATEncoderWeighted(nn.Module):
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
    ):
        super().__init__()

        # 1. Feature Attention (Returns weights now)
        self.num_feat_attn = FeatureAttention(num_input_dim)
        self.text_feat_attn = FeatureAttention(sent_dim+emb_dim)

        # 2. Weight Projectors (NEW)
        # We need to map the Feature Weights (dim 20 or 773) to d_model (128)
        # to apply them to the Encoder Output.
        self.num_weight_proj = nn.Sequential(
            nn.Linear(num_input_dim, d_model),
            nn.Sigmoid(),  # Gating (0-1)
        )
        self.text_weight_proj = nn.Sequential(
             nn.Linear(sent_dim+emb_dim, d_model), nn.Sigmoid()
        )

        # 3. Standard Projections (Input -> d_model)
        self.num_proj = nn.Sequential(
            nn.Linear(num_input_dim, d_model),
            # nn.BatchNorm1d(60),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # self.text_proj = nn.Sequential(
        #     nn.Linear(128, d_model),
        #     nn.LayerNorm(d_model),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        # )

        self.sent_proj = nn.Sequential(
            nn.Linear(n_sent, sent_dim),
            nn.LayerNorm(sent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.emb_proj = nn.Sequential(
            nn.Linear(768, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(sent_dim + emb_dim, d_model),
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

    def forward(self, x_num, x_sent, x_emb):
        """
        x_num : [B,T,F]
        x_sent: [B,T,n_sent]
        x_emb : [B,T,768]
        return: mem_num [B,T,d_model], mem_text [B,T,d_model]
        """
        # A. Feature Attention (Capture the Weights)
        # w_num: [Batch, 60, 20]
        # w_text: [Batch, 60, 773]
        
        #  Build raw text features (128) 
        t_sent = self.sent_proj(x_sent)             # [B,T,32]
        t_emb  = self.emb_proj(x_emb)               # [B,T,96]
        x_text = torch.cat([t_sent, t_emb], dim=-1) # [B,T,128]

        # FeatureAttention (Softmax) + keep weights                       
        x_num, w_num = self.num_feat_attn(x_num)        # w_num : [B,T,F]
        x_text, w_text = self.text_feat_attn(x_text)    # w_text: [B,T,128]

        # B. Project Input to d_model
        h_num = self.num_proj(x_num)        # [B,T,d_model]
        h_text = self.text_proj(x_text)     # [B,T,d_model]

        # C. Positional Encoding
        h_num = self.pos_encoder(h_num.transpose(0, 1)).transpose(0, 1)
        h_text = self.pos_encoder(h_text.transpose(0, 1)).transpose(0, 1)

        # D. MAT Processing (The Context Learning)
        # This mixes information over time and between modalities
        for layer in self.layers:
            h_num, h_text = layer(h_num, h_text)

        # E. RE-WEIGHTING (The Step I Missing)
        # We project the original Feature Weights to the hidden size
        # and use them to "Gate" the final output.
        # This ensures the model output respects the original feature importance.

        # [Batch, 60, 20] -> [Batch, 60, 128]
        gate_num = self.num_weight_proj(w_num)
        # [Batch, 60, 128] -> [Batch, 60, 128]
        gate_text = self.text_weight_proj(w_text)

        # Apply the Gate
        h_num_final = h_num * gate_num      # * (1.0 + 0.2 * gate_num) for residual gating
        h_text_final = h_text * gate_text   # * (1.0 + 0.2 * gate_text)

        return h_num_final, h_text_final
