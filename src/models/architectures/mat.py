import torch
import torch.nn as nn
from src.models.encoders.mat_encoder import MATEncoder
from src.models.encoders.mat_encoder_weighted import MATEncoderWeighted
from src.models.decoders.mat_decoder import MATDecoder
from src.models.layers.positional_encoding import PositionalEncoding


class MAT(nn.Module):
    """
    Autoregressive MAT encoder-decoder (no LearnedQuery).
    - Encoder: mem_num, mem_text
    - Decoder: expects tgt [B,H,D] (target embeddings, causally masked)
    - Head: projects each tgt state -> y_hat
    """

    def __init__(
        self,
        num_input_dim: int,
        n_sent: int,
        d_model: int,
        nhead: int,
        enc_layers: int,
        dec_layers: int,
        dropout: float,
        forecast_horizon: int,
        encoder_type=None,
        use_emb: bool = True,
    ):
        super().__init__()
        self.use_emb = use_emb
        self.H = forecast_horizon
        self.d_model = d_model

        if encoder_type == "weighted":
            self.encoder = MATEncoderWeighted(
                num_input_dim,
                n_sent,
                d_model,
                nhead,
                enc_layers,
                dropout,
                use_emb=self.use_emb,
            )
        else:
            self.encoder = MATEncoder(
                num_input_dim,
                n_sent,
                d_model,
                nhead,
                enc_layers,
                dropout,
                use_emb=self.use_emb,
            )

        self.decoder = MATDecoder(d_model, nhead, dec_layers, dropout)

        self.y_in_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.tgt_pos = PositionalEncoding(d_model, dropout)

        self.bos = nn.Parameter(torch.zeros(1, 1, d_model))

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
        x_emb: torch.Tensor = None,
        y_hist: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Teacher forcing forward.

        Args:
            x_*: inputs [B,T,...]
            y_hist: [B,H] known values used as decoder input (shifted).
                Standard example: y_hist = [y_t, y_{t+1}, ..., y_{t+H-1}] to predict
                [y_{t+1}, ..., y_{t+H}] (one-step shift).
        Returns:
            y_hat: [B,H]
        """
        B = x_num.size(0)
        H = y_hist.size(1)
        assert H == self.H, f"y_hist horizon {H} != forecast_horizon {self.H}"

        if self.use_emb and x_emb is None:
            raise ValueError("MAT(use_emb=True) requires x_emb, got None.")
        mem_num, mem_text = self.encoder(x_num, x_sent, x_emb)  # [B,T,D] each

        # Build tgt inputs: [B,H,D] = [BOS, embed(y_hist[:,:-1])] for strict 1-step shift
        # Here, y_hist is already the shifted input. We build:
        # tgt = [BOS] + embed(y_hist[:, :-1])  => longueur H
        y_emb = self.y_in_proj(y_hist.unsqueeze(-1))  # [B,H,D]
        tgt_tokens = torch.cat([self.bos.expand(B, 1, -1), y_emb], dim=1)  # [B,H+1,D]
        tgt_in = self.tgt_pos(tgt_tokens.transpose(0, 1)).transpose(0, 1)  # [B,H+1,D]
        tgt_out = self.decoder(tgt_in, mem_num, mem_text)  # [B,H+1,D]
        y_hat = self.head(tgt_out[:, 1:, :]).squeeze(-1)  # [B,H]
        return y_hat

    @torch.no_grad()
    def predict(
        self,
        x_num: torch.Tensor,
        x_sent: torch.Tensor,
        x_emb: torch.Tensor = None,
        y0: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Autoregressive inference.

        Args:
            y0: [B] last known value (or proxy) to initialize the AR loop.
        Returns:
            y_hat: [B,H]
        """
        B = x_num.size(0)
        if self.use_emb and x_emb is None:
            raise ValueError("MAT(use_emb=True) requires x_emb, got None.")
        mem_num, mem_text = self.encoder(x_num, x_sent, x_emb)

        prev = y0  # [B]

        tgt_tokens = torch.cat(
            [self.bos.expand(B, 1, -1), self.y_in_proj(y0.view(B, 1, 1))], dim=1
        )  # [B,2,D]
        preds = []
        for _ in range(self.H):
            tgt_in = self.tgt_pos(tgt_tokens.transpose(0, 1)).transpose(0, 1)
            tgt_out = self.decoder(tgt_in, mem_num, mem_text)
            next_y = self.head(tgt_out[:, -1:, :]).squeeze(-1)  # [B,1]
            preds.append(next_y)
            next_emb = self.y_in_proj(next_y.unsqueeze(-1))
            tgt_tokens = torch.cat([tgt_tokens, next_emb], dim=1)
        return torch.cat(preds, dim=1)
