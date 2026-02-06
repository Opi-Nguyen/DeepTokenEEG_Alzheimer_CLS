# src/models/build_model.py
from __future__ import annotations
from typing import Any, Dict
import torch.nn as nn

def build_model(cfg: Dict[str, Any], enc_in: int, seq_len: int) -> nn.Module:
    name = str(cfg["model_name"])

    if name == "DeepTokenEEG":
        # model cũ của bạn (Tokenizer+ResBlock) expects input [B, T, C]
        from src.models.DeepTokenEEG import Model as Core
        core = Core(
            enc_in=int(enc_in),
            num_class=int(cfg["num_class"]),
            d_model=int(cfg["d_model"]),
            dropout=float(cfg["dropout"]),
            n_blocks=int(cfg["resnet_n_blocks"]),
            dilations=list(cfg["resnet_dilations"]),
        )

        class _Wrapper(nn.Module):
            def __init__(self, m: nn.Module):
                super().__init__()
                self.m = m
            def forward(self, x):
                # pipeline gives [B, C, T] -> convert to [B, T, C]
                if x.dim() == 3 and x.shape[1] == enc_in:
                    x = x.permute(0, 2, 1).contiguous()
                return self.m(x)

        return _Wrapper(core)

    if name == "Conformer":
        from src.models.Conformer import Model
        return Model(cfg=cfg, enc_in=int(enc_in), seq_len=int(seq_len))

    if name == "BIOT":
        from src.models.BIOT import Model
        return Model(cfg=cfg, enc_in=int(enc_in), seq_len=int(seq_len))

    if name == "Transformer":
        from src.models.Transformer import Model
        return Model(cfg=cfg, enc_in=int(enc_in), seq_len=int(seq_len))
    
    if name == "TimesNet":
        from src.models.TimesNet import Model
        return Model(cfg=cfg, enc_in=int(enc_in), seq_len=int(seq_len))
    
    if name == "EEG2Rep":
        from src.models.EEG2Rep import Model
        return Model(cfg=cfg, enc_in=int(enc_in), seq_len=int(seq_len))
    
    if name == "LEAD":
        from src.models.LEAD import Model
        return Model(cfg=cfg, enc_in=int(enc_in), seq_len=int(seq_len))

    raise ValueError(f"Unknown model_name={name}")
