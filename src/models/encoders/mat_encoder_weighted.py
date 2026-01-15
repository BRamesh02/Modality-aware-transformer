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
        x_n_norm = self.norm1_num(x_num)
        attn_n, _ = self.self_attn_num(x_n_norm, x_n_norm, x_n_norm)
        x_num = x_num + self.dropout(attn_n)

        x_t_norm = self.norm1_text(x_text)
        attn_t, _ = self.self_attn_text(x_t_norm, x_t_norm, x_t_norm)
        x_text = x_text + self.dropout(attn_t)

        x_n_norm = self.norm2_num(x_num)
        x_t_norm = self.norm2_text(x_text)

        attn_n_cross, _ = self.cross_attn_num(query=x_n_norm, key=x_t_norm, value=x_t_norm)
        x_num = x_num + self.dropout(attn_n_cross)

        attn_t_cross, _ = self.cross_attn_text(query=x_t_norm, key=x_n_norm, value=x_n_norm)
        x_text = x_text + self.dropout(attn_t_cross)

        x_n_norm = self.norm3_num(x_num)
        ff_n = self.ff_num(x_n_norm)
        x_num = x_num + self.dropout(ff_n)

        x_t_norm = self.norm3_text(x_text)
        ff_t = self.ff_text(x_t_norm)
        x_text = x_text + self.dropout(ff_t)

        return x_num, x_text
    

class MATEncoderWeighted(nn.Module):
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

        self.num_weight_proj = nn.Sequential(
            nn.Linear(num_input_dim, d_model),
            nn.Sigmoid(), 
        )
        self.text_weight_proj = nn.Sequential(
             nn.Linear(text_feat_dim, d_model), nn.Sigmoid()
        )

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
        else:
            self.emb_proj = None

        self.text_proj = nn.Sequential(
            nn.Linear(text_feat_dim, d_model),
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
        
        self.norm_final_num = nn.LayerNorm(d_model)
        self.norm_final_text = nn.LayerNorm(d_model)

    def forward(self, x_num, x_sent, x_emb=None):
        """
        x_num : [B,T,F]
        x_sent: [B,T,n_sent]
        x_emb : [B,T,768]
        return: mem_num [B,T,d_model], mem_text [B,T,d_model]
        """
        t_sent = self.sent_proj(x_sent)

        if self.use_emb:
            if x_emb is None:
                raise ValueError("MATEncoderWeighted(use_emb=True) requires x_emb, got None.")
            t_emb = self.emb_proj(x_emb)
            x_text = torch.cat([t_sent, t_emb], dim=-1)
        else:
            x_text = t_sent

        x_num, w_num = self.num_feat_attn(x_num)        
        x_text, w_text = self.text_feat_attn(x_text)    

        h_num = self.num_proj(x_num)        
        h_text = self.text_proj(x_text)     

        h_num = self.pos_encoder(h_num.transpose(0, 1)).transpose(0, 1)
        h_text = self.pos_encoder(h_text.transpose(0, 1)).transpose(0, 1)

        for layer in self.layers:
            h_num, h_text = layer(h_num, h_text)
            
        h_num = self.norm_final_num(h_num)
        h_text = self.norm_final_text(h_text)

        gate_num = self.num_weight_proj(w_num)
        gate_text = self.text_weight_proj(w_text)

        h_num_final = h_num * gate_num      
        h_text_final = h_text * gate_text   

        return h_num_final, h_text_final