import torch
import torch.nn as nn
from src.models.encoders.mat_encoder_plain import MATEncoderPlain
from src.models.encoders.mat_encoder_weighted import MATEncoderWeighted
from src.models.decoders.mat_decoder import MATDecoder
from src.models.layers.positional_encoding import PositionalEncoding


class MAT(nn.Module):
    """
    MAT encoder-decoder autoregressif (sans LearnedQuery).
    - Encoder: mem_num, mem_text
    - Decoder: attend un tgt [B,H,D] (embeddings de target, masqués causalement)
    - Head: projette chaque état tgt -> y_hat
    """

    def __init__(
        self,
        num_input_dim: int,
        n_sent: int,
        d_model: int = 128,
        nhead: int = 4,
        enc_layers: int = 2,
        dec_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 1,
        encoder_type="weighted"
    ):
        super().__init__()
        self.H = forecast_horizon
        self.d_model = d_model

        if encoder_type == "weighted":
            self.encoder = MATEncoderWeighted(
                num_input_dim, n_sent, d_model, nhead, enc_layers, dropout
            )
        elif encoder_type == "plain":
            self.encoder = MATEncoderPlain(
                num_input_dim, n_sent, d_model, nhead, enc_layers, dropout
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.decoder = MATDecoder(d_model, nhead, dec_layers, dropout)

        # --- Target embedding (scalar -> d_model) ---
        # y_t (1 dim) -> embedding dim D
        self.y_in_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.tgt_pos = PositionalEncoding(d_model, dropout)

        # BOS token (start of decoding)
        self.bos = nn.Parameter(torch.zeros(1, 1, d_model))

        # Regression head applied at each horizon step
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x_num: torch.Tensor,
        x_sent: torch.Tensor,
        x_emb: torch.Tensor,
        y_hist: torch.Tensor,
    ) -> torch.Tensor:
        """
        Teacher forcing forward.

        Args:
            x_* : inputs [B,T,...]
            y_hist: [B,H] = valeurs "connues" utilisées comme entrée du décodeur (décalées)
                    Exemple standard: y_hist = [y_t, y_{t+1}, ..., y_{t+H-1}] pour prédire
                    [y_{t+1}, ..., y_{t+H}] (décalage d'un pas).
        Returns:
            y_hat: [B,H]
        """
        B = x_num.size(0)
        H = y_hist.size(1)
        assert H == self.H, f"y_hist horizon {H} != forecast_horizon {self.H}"

        mem_num, mem_text = self.encoder(x_num, x_sent, x_emb)  # [B,T,D] each

        # Build tgt inputs: [B,H,D] = [BOS, embed(y_hist[:,:-1])] for strict 1-step shift
        # Ici, y_hist est déjà "l'entrée" décalée. On construit:
        # tgt = [BOS] + embed(y_hist[:, :-1])  => longueur H
        y_emb = self.y_in_proj(y_hist.unsqueeze(-1))  # [B,H,D]
        tgt_tokens = torch.cat([self.bos.expand(B, 1, -1), y_emb], dim=1)  # [B,H+1,D]
        tgt_in = self.tgt_pos(tgt_tokens.transpose(0, 1)).transpose(0, 1)  # [B,H+1,D]
        tgt_out = self.decoder(tgt_in, mem_num, mem_text)                   # [B,H+1,D]
        y_hat = self.head(tgt_out[:, 1:, :]).squeeze(-1)                    # [B,H]
        return y_hat

    @torch.no_grad()
    def predict(
        self,
        x_num: torch.Tensor,
        x_sent: torch.Tensor,
        x_emb: torch.Tensor,
        y0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inference autoregressive.

        Args:
            y0: [B] = dernière valeur connue (ou un proxy) pour initialiser l'AR.
        Returns:
            y_hat: [B,H]
        """
        B = x_num.size(0)
        mem_num, mem_text = self.encoder(x_num, x_sent, x_emb)

        prev = y0  # [B]

        # on construit tgt pas à pas
        tgt_tokens = torch.cat([self.bos.expand(B, 1, -1),
                        self.y_in_proj(y0.view(B,1,1))], dim=1)  # [B,2,D]
        preds = []
        for _ in range(self.H):
            tgt_in = self.tgt_pos(tgt_tokens.transpose(0, 1)).transpose(0, 1)
            tgt_out = self.decoder(tgt_in, mem_num, mem_text)
            next_y = self.head(tgt_out[:, -1:, :]).squeeze(-1)  # [B,1]
            preds.append(next_y)
            # just append predicted y for next step
            next_emb = self.y_in_proj(next_y.unsqueeze(-1))
            tgt_tokens = torch.cat([tgt_tokens, next_emb], dim=1)
        return torch.cat(preds, dim=1)