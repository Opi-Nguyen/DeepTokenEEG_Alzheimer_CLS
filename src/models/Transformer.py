# src/models/Transformer.py
from __future__ import annotations
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers.Embed import DataEmbedding
from src.models.layers.Transformer_EncDec import Encoder, EncoderLayer
from src.models.layers.SelfAttention_Family import FullAttention, AttentionLayer


class Model(nn.Module):
    """
    Transformer (supervised-only, pipeline-compatible)
    Input : x [B, C, T] or [B, T, C]
    Output: logits [B, num_class]
    """
    def __init__(self, cfg=None, enc_in: int = None, seq_len: int = None, configs=None, **kwargs):
        super().__init__()
        if cfg is None:
            cfg = configs
        if cfg is None:
            raise ValueError("Transformer requires cfg/configs")

        # cfg object -> dict
        if not hasattr(cfg, "get"):
            cfg = vars(cfg)

        self.enc_in = int(enc_in)
        self.seq_len = int(seq_len)
        self.num_class = int(cfg["num_class"])

        d_model = int(cfg.get("d_model", 128))
        dropout = float(cfg.get("dropout", 0.1))
        n_heads = int(cfg.get("n_heads", 4))
        d_ff = int(cfg.get("d_ff", 256))
        e_layers = int(cfg.get("e_layers", 2))
        factor = int(cfg.get("factor", 1))
        activation = str(cfg.get("activation", "gelu"))
        self.output_attention = bool(cfg.get("output_attention", False))

        embed_type = str(cfg.get("embed_type", "fixed"))
        freq = str(cfg.get("freq", "h"))  # không dùng nếu x_mark=None

        # DataEmbedding expects x: [B, T, C]
        self.enc_embedding = DataEmbedding(
            c_in=self.enc_in,
            d_model=d_model,
            embed_type=embed_type,
            freq=freq,
            dropout=dropout,
        )

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

        self.act = F.gelu if activation.lower() == "gelu" else F.relu
        self.drop = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model, self.num_class)

    def _to_BTC(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Transformer expects 3D tensor, got {tuple(x.shape)}")

        # [B, T, C]
        if x.shape[-1] == self.enc_in:
            return x

        # [B, C, T] -> [B, T, C]
        if x.shape[1] == self.enc_in:
            return x.transpose(1, 2).contiguous()

        raise ValueError(f"Channel mismatch: enc_in={self.enc_in}, got {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_tc = self._to_BTC(x)                      # [B, T, C]
        enc_out = self.enc_embedding(x_tc, None)    # [B, T, d_model]
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        out = self.act(enc_out)
        out = self.drop(out)
        out = out.mean(dim=1)                       # [B, d_model]
        return self.projection(out)                 # [B, num_class]
