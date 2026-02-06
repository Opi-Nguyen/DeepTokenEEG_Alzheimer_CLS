# src/models/TimesNetCLS.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers.Embed import DataEmbedding


def _fft_for_period(x: torch.Tensor, k: int = 3) -> Tuple[List[int], torch.Tensor]:
    """
    x: [B, T, D]
    return:
      periods: list[int] length k
      weights: [k] (softmax-able)
    """
    B, T, D = x.shape
    # rfft over time
    xf = torch.fft.rfft(x, dim=1)  # [B, F, D]
    amp = xf.abs().mean(dim=0).mean(dim=-1)  # [F]
    amp[0] = 0  # drop DC

    k = min(k, amp.shape[0])
    top = torch.topk(amp, k=k, dim=0)
    freq = top.indices  # [k]

    periods: List[int] = []
    for f in freq.tolist():
        if f <= 0:
            p = T
        else:
            p = max(1, T // f)
        periods.append(p)

    weights = top.values  # [k]
    return periods, weights


class InceptionBlockV1(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_kernels: int = 6):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_kernels):
            k = 2 * i + 1
            p = i
            self.convs.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=(k, k), padding=(p, p))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        outs = [conv(x) for conv in self.convs]
        y = torch.stack(outs, dim=-1).mean(dim=-1)
        return y


class TimesBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, top_k: int, num_kernels: int, dropout: float):
        super().__init__()
        self.top_k = int(top_k)
        self.conv = nn.Sequential(
            InceptionBlockV1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            InceptionBlockV1(d_ff, d_model, num_kernels=num_kernels),
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        """
        B, T, D = x.shape
        periods, weights = _fft_for_period(x, k=self.top_k)
        weights = F.softmax(weights, dim=0)  # [k]

        res = []
        for p in periods:
            # pad T to multiple of p
            if T % p != 0:
                pad_len = p - (T % p)
                x_pad = F.pad(x, (0, 0, 0, pad_len))  # pad time on the right
                Tp = T + pad_len
            else:
                x_pad = x
                Tp = T

            # reshape -> 2D: [B, D, Tp//p, p]
            x_2d = x_pad.reshape(B, Tp // p, p, D).permute(0, 3, 1, 2).contiguous()
            y_2d = self.conv(x_2d)  # [B, D, Tp//p, p]
            y = y_2d.permute(0, 2, 3, 1).reshape(B, Tp, D)[:, :T, :].contiguous()
            res.append(y)

        y = torch.stack(res, dim=-1)  # [B, T, D, k]
        w = weights.view(1, 1, 1, -1)
        y = (y * w).sum(dim=-1)       # [B, T, D]

        y = self.drop(y)
        return self.norm(x + y)


class Model(nn.Module):
    """
    TimesNet-CLS (supervised-only, pipeline-compatible)
    Input : x [B, C, T] or [B, T, C]
    Output: logits [B, num_class]
    """
    def __init__(self, cfg: Dict[str, Any], enc_in: int, seq_len: int):
        super().__init__()
        self.enc_in = int(enc_in)
        self.seq_len = int(seq_len)
        self.num_class = int(cfg["num_class"])

        d_model = int(cfg.get("d_model", 128))
        d_ff = int(cfg.get("d_ff", 256))
        e_layers = int(cfg.get("e_layers", 2))
        dropout = float(cfg.get("dropout", 0.1))
        top_k = int(cfg.get("top_k", 3))
        num_kernels = int(cfg.get("num_kernels", 6))

        # embedding expects x [B, T, C]
        self.enc_embedding = DataEmbedding(
            c_in=self.enc_in,
            d_model=d_model,
            embed_type=str(cfg.get("embed_type", "fixed")),
            freq=str(cfg.get("freq", "h")),
            dropout=dropout,
        )

        self.blocks = nn.ModuleList([
            TimesBlock(d_model=d_model, d_ff=d_ff, top_k=top_k, num_kernels=num_kernels, dropout=dropout)
            for _ in range(e_layers)
        ])

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model, self.num_class)

    def _to_BTC(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"TimesNet expects 3D tensor, got {tuple(x.shape)}")

        # [B, T, C]
        if x.shape[-1] == self.enc_in:
            return x

        # [B, C, T] -> [B, T, C]
        if x.shape[1] == self.enc_in:
            return x.transpose(1, 2).contiguous()

        raise ValueError(f"Channel mismatch: enc_in={self.enc_in}, got {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_tc = self._to_BTC(x)                 # [B, T, C]
        # enforce T == seq_len (để ổn định)
        T = x_tc.shape[1]
        if T < self.seq_len:
            x_tc = F.pad(x_tc, (0, 0, 0, self.seq_len - T), mode="replicate")
        elif T > self.seq_len:
            x_tc = x_tc[:, : self.seq_len, :].contiguous()

        enc = self.enc_embedding(x_tc, None)   # [B, T, d_model]
        for blk in self.blocks:
            enc = blk(enc)

        out = self.act(enc)
        out = self.drop(out)
        out = out.mean(dim=1)                  # [B, d_model]
        return self.projection(out)            # [B, num_class]
