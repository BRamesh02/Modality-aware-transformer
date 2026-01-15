import torch
import torch.nn as nn
from src.models.encoders.canonical_encoder import CanonicalEncoder
from src.models.decoders.canonical_decoder import CanonicalDecoder
from src.models.layers.positional_encoding import PositionalEncoding


class CanonicalTransformer(nn.Module):
    """
    Baseline Transformer encoder-decoder "classique" autoregressif.
    - Encoder: CanonicalEncoder -> memory [B,T,D]
    - Decoder: CanonicalDecoder avec self-attn masquée + cross-attn
    - Entrée decoder (train): [BOS, y_t, ..., y_{t+H-1}] (H+1 tokens)
    - Sortie: prédictions y_{t+1}..y_{t+H} via états 1..H
    """
    def __init__(self, num_input_dim: int, n_sent: int,
                 d_model: int, nhead: int,
                 enc_layers: int, dec_layers: int,
                 dropout: float, forecast_horizon: int = 1, use_emb: bool = True):
        super().__init__()
        self.H = forecast_horizon
        self.d_model = d_model
        
        self.use_emb = use_emb
        self.encoder = CanonicalEncoder(num_input_dim, n_sent, d_model, nhead, enc_layers, dropout , use_emb=self.use_emb,)

        self.y_in_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.bos = nn.Parameter(torch.zeros(1, 1, d_model))

        self.tgt_pos = PositionalEncoding(d_model, dropout)
        
        self.decoder = CanonicalDecoder(d_model, nhead, dec_layers, dropout)

        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x_num, x_sent, x_emb=None, y_hist=None):
        """
        Train (teacher forcing)
        y_hist: [B,H] = [y_t, y_{t+1}, ..., y_{t+H-1}]
        Retour: y_hat [B,H] = [y_{t+1}, ..., y_{t+H}]
        """
        B = x_num.size(0)
        H = y_hist.size(1)
        assert H == self.H

        if self.use_emb and x_emb is None:
            raise ValueError("CanonicalTransformer(use_emb=True) requires x_emb, got None.")
        
        memory = self.encoder(x_num, x_sent, x_emb)  # [B,T,D]

        # tgt_tokens = [BOS, y_t..y_{t+H-1}] => longueur H+1
        y_emb = self.y_in_proj(y_hist.unsqueeze(-1))          # [B,H,D]
        tgt_tokens = torch.cat([self.bos.expand(B, 1, -1), y_emb], dim=1)  # [B,H+1,D]

        tgt_in = self.tgt_pos(tgt_tokens.transpose(0, 1)).transpose(0, 1)  # [B,H+1,D]

        tgt_out = self.decoder(tgt_in, memory)              # [B,H+1,D]
        y_hat = self.head(tgt_out[:, 1:, :]).squeeze(-1)    # [B,H]
        return y_hat

    @torch.no_grad()
    def predict(self, x_num, x_sent, x_emb=None, y0=None):
        """
        Inference AR:
        y0: [B] = y_t (dernière valeur connue)
        Retour: [B,H] = y_{t+1}..y_{t+H}
        """
        B = x_num.size(0)
        if self.use_emb and x_emb is None:
            raise ValueError("CanonicalTransformer(use_emb=True) requires x_emb, got None.")
        memory = self.encoder(x_num, x_sent, x_emb)

        tgt_tokens = torch.cat(
            [self.bos.expand(B, 1, -1), self.y_in_proj(y0.view(B, 1, 1))],
            dim=1
        )  # [B,2,D]

        preds = []
        for _ in range(self.H):
            tgt_in = self.tgt_pos(tgt_tokens.transpose(0, 1)).transpose(0, 1)
            tgt_out = self.decoder(tgt_in, memory)

            next_y = self.head(tgt_out[:, -1:, :]).squeeze(-1)  # [B,1]
            preds.append(next_y)

            next_emb = self.y_in_proj(next_y.unsqueeze(-1))     # [B,1,D]
            tgt_tokens = torch.cat([tgt_tokens, next_emb], dim=1)

        return torch.cat(preds, dim=1)  # [B,H]