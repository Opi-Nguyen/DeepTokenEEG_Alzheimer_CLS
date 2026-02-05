# src/models/BIOT.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers.Transformer_EncDec import Encoder, EncoderLayer
from src.models.layers.SelfAttention_Family import FullAttention, AttentionLayer
from src.models.layers.Embed import BIOTEmbedding


class Model(nn.Module):
    """
    BIOT (supervised-only, pipeline-compatible)
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

        patch_len = int(cfg.get("patch_len", 16))
        stride = int(cfg.get("stride", patch_len))
        augmentations = list(cfg.get("biot_augmentations", ["mask", "channel"]))

        self.enc_embedding = BIOTEmbedding(
            enc_in=self.enc_in,
            seq_len=self.seq_len,
            d_model=d_model,
            patch_len=patch_len,
            stride=stride,
            augmentation=augmentations,
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
        self.dropout = nn.Dropout(dropout)

        # robust: auto infer in_features
        self.projection = nn.Linear(d_model, self.num_class)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"BIOT expects 3D tensor, got shape={tuple(x.shape)}")

        enc_out = self.enc_embedding(x)       # embedding tự xử lý [B,C,T] hoặc [B,T,C]
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        out = self.act(enc_out)
        out = self.dropout(out)
        out = out.mean(dim=1)                # [B, d_model]
        return self.projection(out) 
