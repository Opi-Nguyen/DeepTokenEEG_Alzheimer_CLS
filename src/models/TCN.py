# src/models/TCN.py
from __future__ import annotations
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Chomp1d(nn.Module):
    """Chomp padding ở cuối để giữ causal-ish output length."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class _TemporalBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        use_weight_norm: bool = False,
    ):
        super().__init__()
        k = int(kernel_size)
        d = int(dilation)
        # padding để output length không giảm (sau đó chomp lại)
        pad = (k - 1) * d

        conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, dilation=d)
        conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=pad, dilation=d)
        if use_weight_norm:
            conv1 = nn.utils.weight_norm(conv1)
            conv2 = nn.utils.weight_norm(conv2)

        self.net = nn.Sequential(
            conv1,
            _Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            conv2,
            _Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
        self.out_act = nn.ReLU()

        # init ổn định
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.out_act(y + res)


class _TemporalConvNet(nn.Module):
    def __init__(
        self,
        in_ch: int,
        channels: List[int],
        kernel_size: int,
        dropout: float,
        dilations: Union[List[int], None] = None,
        use_weight_norm: bool = False,
    ):
        super().__init__()
        if dilations is None:
            dilations = [2**i for i in range(len(channels))]
        if len(dilations) != len(channels):
            raise ValueError(f"len(dilations) must equal len(channels). Got {len(dilations)} vs {len(channels)}")

        layers = []
        c_prev = int(in_ch)
        for c_out, d in zip(channels, dilations):
            layers.append(
                _TemporalBlock(
                    in_ch=c_prev,
                    out_ch=int(c_out),
                    kernel_size=int(kernel_size),
                    dilation=int(d),
                    dropout=float(dropout),
                    use_weight_norm=use_weight_norm,
                )
            )
            c_prev = int(c_out)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Model(nn.Module):
    """
    TCN (supervised-only, pipeline-compatible)

    Input:
      - x: [B, C, T] (pipeline) hoặc [B, T, C]
    Output:
      - logits: [B, num_class]
    """
    def __init__(self, cfg: Dict[str, Any], enc_in: int, seq_len: int):
        super().__init__()
        if not hasattr(cfg, "get"):
            cfg = vars(cfg)

        self.enc_in = int(enc_in)
        self.seq_len = int(seq_len)
        self.num_class = int(cfg["num_class"])

        dropout = float(cfg.get("dropout", 0.1))

        # cấu hình TCN
        # tcn_channels: list hoặc string "64,128,128"
        ch = cfg.get("tcn_channels", "64,128,128")
        if isinstance(ch, str):
            channels = [int(x.strip()) for x in ch.split(",") if x.strip()]
        else:
            channels = [int(x) for x in ch]
        if len(channels) == 0:
            channels = [128, 128]

        kernel_size = int(cfg.get("tcn_kernel_size", 3))
        use_weight_norm = bool(cfg.get("tcn_weight_norm", False))

        dils = cfg.get("tcn_dilations", None)
        if isinstance(dils, str):
            dilations = [int(x.strip()) for x in dils.split(",") if x.strip()]
        elif isinstance(dils, (list, tuple)):
            dilations = [int(x) for x in dils]
        else:
            dilations = None  # auto 2**i

        self.tcn = _TemporalConvNet(
            in_ch=self.enc_in,
            channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
            dilations=dilations,
            use_weight_norm=use_weight_norm,
        )

        self.pool = str(cfg.get("tcn_pool", "mean")).lower()  # "mean" | "last" | "max"
        self.classifier = nn.Linear(channels[-1], self.num_class)

    def _to_BCT(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"TCN expects 3D tensor, got {tuple(x.shape)}")

        C = self.enc_in

        # normalize to [B, C, T]
        if x.shape[1] == C:
            x_ct = x
        elif x.shape[-1] == C:
            x_ct = x.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"Channel mismatch: enc_in={C}, got {tuple(x.shape)}")

        # enforce T == seq_len (để so sánh fair giữa model)
        T = int(x_ct.shape[-1])
        if T < self.seq_len:
            x_ct = F.pad(x_ct, (0, self.seq_len - T), mode="replicate")
        elif T > self.seq_len:
            x_ct = x_ct[..., : self.seq_len].contiguous()

        return x_ct

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ct = self._to_BCT(x)          # [B, C, T]
        h = self.tcn(x_ct)              # [B, hidden, T]

        if self.pool == "last":
            feat = h[:, :, -1]          # [B, hidden]
        elif self.pool == "max":
            feat = torch.max(h, dim=-1).values
        else:
            feat = torch.mean(h, dim=-1)

        return self.classifier(feat)    # [B, num_class]


