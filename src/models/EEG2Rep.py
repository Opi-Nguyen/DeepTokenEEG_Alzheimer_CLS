# src/models/EEG2Rep.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers.Transformer_EncDec import Encoder, EncoderLayer
from src.models.layers.SelfAttention_Family import FullAttention, AttentionLayer
from src.models.layers.Embed import EEG2RepEmbedding, PositionalEmbedding


class Model(nn.Module):
    """
    EEG2Rep (supervised-only, pipeline-compatible)

    Input:
      - x: [B, C, T] (pipeline) or [B, T, C]
    Output:
      - logits: [B, num_class]
    """

    def __init__(self, cfg, enc_in: int, seq_len: int):
        super().__init__()

        # cfg có thể là dict hoặc object
        if not hasattr(cfg, "get"):
            cfg = vars(cfg)

        self.enc_in = int(enc_in)
        self.seq_len = int(seq_len)

        self.output_attention = bool(cfg.get("output_attention", False))
        self.num_class = int(cfg["num_class"])

        d_model = int(cfg.get("d_model", 128))
        dropout = float(cfg.get("dropout", 0.1))
        n_heads = int(cfg.get("n_heads", 4))
        d_ff = int(cfg.get("d_ff", 256))
        e_layers = int(cfg.get("e_layers", 2))
        factor = int(cfg.get("factor", 1))
        activation = str(cfg.get("activation", "gelu"))

        pooling_size = int(cfg.get("pooling_size", 2))

        # Embedding (EEG2RepEmbedding EXPECTS input [B, T, C])
        self.pos_embed = PositionalEmbedding(d_model)
        self.enc_embedding = EEG2RepEmbedding(
            c_in=self.enc_in,
            d_model=d_model,
            pooling_size=pooling_size,
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=self.output_attention,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.projection = nn.Linear(d_model, self.num_class)

    def _to_BTC(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return x_tc: [B, T, C]
        """
        if x.dim() != 3:
            raise ValueError(f"EEG2Rep expects 3D tensor, got {tuple(x.shape)}")

        # already [B, T, C]
        if x.shape[-1] == self.enc_in:
            x_tc = x
        # [B, C, T] -> [B, T, C]
        elif x.shape[1] == self.enc_in:
            x_tc = x.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"Channel mismatch: enc_in={self.enc_in}, got {tuple(x.shape)}")

        # enforce T == seq_len (ổn định)
        T = int(x_tc.shape[1])
        if T < self.seq_len:
            x_tc = F.pad(x_tc, (0, 0, 0, self.seq_len - T), mode="replicate")
        elif T > self.seq_len:
            x_tc = x_tc[:, : self.seq_len, :].contiguous()

        return x_tc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_tc = self._to_BTC(x)                 # [B, T, C]

        enc_out = self.enc_embedding(x_tc)     # [B, T', d_model] (T' sau pooling)
        out = self.norm1(enc_out)
        out = out + self.pos_embed(out)
        out = self.norm2(out)

        out, _ = self.encoder(out, attn_mask=None)  # [B, T', d_model]

        out = out.mean(dim=1)                   # [B, d_model]
        return self.projection(out)             # [B, num_class]
