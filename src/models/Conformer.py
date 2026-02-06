# src/models/Conformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers.Transformer_EncDec import Encoder, EncoderLayer
from src.models.layers.SelfAttention_Family import FullAttention, AttentionLayer
from src.models.layers.Embed import ShallowNetEmbedding


class Model(nn.Module):
    """
    EEG-Conformer (supervised-only, pipeline-compatible)
    Input from pipeline: x = [B, C, T]
    Output: logits [B, num_class]
    """
    def __init__(self, cfg, enc_in: int, seq_len: int):
        super().__init__()
        self.enc_in = int(enc_in)
        self.seq_len = int(seq_len)

        d_model = int(cfg.get("d_model", 128))
        dropout = float(cfg.get("dropout", 0.1))
        n_heads = int(cfg.get("n_heads", 4))
        d_ff = int(cfg.get("d_ff", 256))
        e_layers = int(cfg.get("e_layers", 2))
        factor = int(cfg.get("factor", 1))
        activation = str(cfg.get("activation", "gelu"))
        self.output_attention = bool(cfg.get("output_attention", False))
        self.num_class = int(cfg["num_class"])

        # Embedding: expects [B, T, C]
        self.enc_embedding = ShallowNetEmbedding(self.enc_in, d_model, dropout)

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

        self.act = F.gelu if activation.lower() == "gelu" else F.relu
        self.dropout = nn.Dropout(dropout)

        # robust: infer in_features automatically
        self.projection = nn.Linear(d_model, self.num_class)


    def forward(self, x):
        # x: [B, C, T] -> [B, T, C]
        if x.dim() != 3:
            raise ValueError(f"Conformer expects 3D tensor, got {tuple(x.shape)}")

        if x.shape[-1] == self.enc_in:
            # already [B, T, C]
            x_tc = x
        elif x.shape[1] == self.enc_in:
            # [B, C, T] -> [B, T, C]
            x_tc = x.transpose(1, 2).contiguous()
        else:
            raise ValueError(
                f"Channel mismatch: enc_in={self.enc_in}, got input shape={tuple(x.shape)}"
            )

        enc_out = self.enc_embedding(x_tc)   # expects [B, T, C]
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        out = self.act(enc_out)
        out = self.dropout(out)
        out = out.mean(dim=1)          # [B, d_model]  (mean over tokens)
        return self.projection(out) 