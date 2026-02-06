# src/models/LEAD.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers.ADformer_EncDec import Encoder, EncoderLayer
from src.models.layers.SelfAttention_Family import ADformerLayer
from src.models.layers.Embed import TokenChannelEmbedding


class Model(nn.Module):
    """
    LEAD (supervised-only, pipeline-compatible)

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

        self.num_class = int(cfg["num_class"])
        d_model = int(cfg.get("d_model", 128))
        dropout = float(cfg.get("dropout", 0.1))
        n_heads = int(cfg.get("n_heads", 4))
        d_ff = int(cfg.get("d_ff", 256))
        e_layers = int(cfg.get("e_layers", 2))
        activation = str(cfg.get("activation", "gelu"))
        output_attention = bool(cfg.get("output_attention", False))
        no_inter_attn = bool(cfg.get("no_inter_attn", False))

        no_temporal_block = bool(cfg.get("no_temporal_block", False))
        no_channel_block = bool(cfg.get("no_channel_block", False))
        if no_temporal_block and no_channel_block:
            raise ValueError("At least one of the two blocks should be enabled (temporal/channel).")

        def _parse_int_list(v):
            if v is None:
                return []
            if isinstance(v, (list, tuple)):
                return [int(x) for x in v]
            if isinstance(v, str):
                v = v.strip()
                if not v:
                    return []
                return [int(x.strip()) for x in v.split(",") if x.strip()]
            return [int(v)]

        patch_len_list = [] if no_temporal_block else _parse_int_list(cfg.get("patch_len_list", "16"))
        up_dim_list = [] if no_channel_block else _parse_int_list(cfg.get("up_dim_list", "16"))

        # stride_list default = patch_len_list (giống code gốc)
        stride_list = _parse_int_list(cfg.get("stride_list", patch_len_list))
        if len(stride_list) != len(patch_len_list):
            stride_list = patch_len_list[:]  # fallback

        # augmentations: supervised thì default "none" (đỡ random augmentation mạnh)
        aug = cfg.get("augmentations", "none")
        if isinstance(aug, str):
            augmentations = [x.strip() for x in aug.split(",") if x.strip()]
        else:
            augmentations = list(aug) if aug is not None else ["none"]
        if len(augmentations) == 0:
            augmentations = ["none"]

        # Embedding (TokenChannelEmbedding expects x: [B, seq_len, enc_in] == [B, T, C])
        self.enc_embedding = TokenChannelEmbedding(
            enc_in=self.enc_in,
            seq_len=self.seq_len,
            d_model=d_model,
            patch_len_list=patch_len_list,
            up_dim_list=up_dim_list,
            stride_list=stride_list,
            dropout=dropout,
            augmentation=augmentations,
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ADformerLayer(
                        len(patch_len_list),
                        len(up_dim_list),
                        d_model,
                        n_heads,
                        dropout,
                        output_attention,
                        no_inter_attn,
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

        self.act = F.gelu
        self.drop = nn.Dropout(dropout)

        # dùng LazyLinear cho chắc (vì dim flatten phụ thuộc embedding/encoder)
        self.classifier = nn.LazyLinear(self.num_class)

    def _to_BTC(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return x_tc: [B, T, C] with T == self.seq_len and C == self.enc_in
        """
        if x.dim() != 3:
            raise ValueError(f"LEAD expects 3D tensor, got {tuple(x.shape)}")

        C = self.enc_in

        # normalize to [B, C, T] so we can pad/crop on last dim
        if x.shape[1] == C:          # [B, C, T]
            x_ct = x
        elif x.shape[-1] == C:       # [B, T, C] -> [B, C, T]
            x_ct = x.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"Channel mismatch: enc_in={C}, got {tuple(x.shape)}")

        # enforce T
        T = int(x_ct.shape[-1])
        if T < self.seq_len:
            x_ct = F.pad(x_ct, (0, self.seq_len - T), mode="replicate")
        elif T > self.seq_len:
            x_ct = x_ct[..., : self.seq_len].contiguous()

        # back to [B, T, C]
        return x_ct.transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_tc = self._to_BTC(x)  # [B, T, C]

        enc_out_t, enc_out_c = self.enc_embedding(x_tc)
        enc_out_t, enc_out_c, _, _ = self.encoder(enc_out_t, enc_out_c, attn_mask=None)

        if enc_out_t is None:
            enc_out = enc_out_c
        elif enc_out_c is None:
            enc_out = enc_out_t
        else:
            enc_out = torch.cat((enc_out_t, enc_out_c), dim=1)

        out = self.act(enc_out)
        out = self.drop(out)
        out = out.reshape(out.shape[0], -1)   # [B, *]
        return self.classifier(out)           # [B, num_class]
