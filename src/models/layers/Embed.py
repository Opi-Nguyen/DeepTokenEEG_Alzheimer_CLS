import copy
import math
import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.utils import weight_norm

from src.models.layers.Augmentation import get_augmentation
from src.utils.uea import bandpass_filter_func


# 2-d relative coordinates for 19 channels. We define position from left to right, top to bottom.
# Note that channels T3, T4, T5, T6 in old system are the same channels as T7, T8, P7, P8 in new system.
# 'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3/T7', 'C3', 'Cz', 'C4', 'T4/T8',
# 'T5/P7', 'P3', 'Pz', 'P4', 'T6/P8', 'O1', 'O2'

CHANNEL_RELATIVE_COORDINATES = {
    "Fp1": (2, 1), "Fp2": (4, 1),
    "F7": (1, 2), "F3": (2, 2), "Fz": (3, 2), "F4": (4, 2), "F8": (5, 2),
    "T3": (1, 3), "C3": (2, 3), "Cz": (3, 3), "C4": (4, 3), "T4": (5, 3),
    "T5": (1, 4), "P3": (2, 4), "Pz": (3, 4), "P4": (4, 4), "T6": (5, 4),
    "O1": (2, 5), "O2": (4, 5),
}


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class ChannelPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(ChannelPositionalEmbedding, self).__init__()
        if (d_model // 2) % 2 != 0:
            raise ValueError("d_model must be an even number for 2-D channel positional embedding.")
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, (d_model // 2)).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, (d_model // 2), 2).float() * -(math.log(10000.0) / (d_model // 2))
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        coordinates = torch.tensor(list(CHANNEL_RELATIVE_COORDINATES.values())).to(x.device)
        x_axis = self.pe[:, coordinates[:, 0].long()]
        y_axis = self.pe[:, coordinates[:, 1].long()]
        return torch.cat([x_axis, y_axis], dim=-1)


class TokenEmbedding(nn.Module):  # (batch_size, seq_len, enc_in)
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="fixed", freq="h"):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = (
            self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        )
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="timeF", freq="h"):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = (
                self.value_embedding(x)
                + self.temporal_embedding(x_mark)
                + self.position_embedding(x)
            )
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)  # c_in is seq_length here
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)  # (batch_size, enc_in, seq_length)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)  # (batch_size, enc_in, d_model)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ShallowNetEmbedding(nn.Module):
    """
    Robust ShallowNetEmbedding for EEG.
    Accept x:
      - [B, C, T]
      - [B, T, C]
      - [B, 1, C, T]
    Internally standardize -> [B, 1, C, T]
    """
    def __init__(
        self,
        c_in: int,
        d_model: int,
        dropout: float,
        temporal_kernel: int = 25,
        pool_kernel: int = 4,
        pool_stride: int = 2,
    ):
        super().__init__()
        self.c_in = int(c_in)
        self.temporal_kernel = int(temporal_kernel)
        self.pool_kernel = int(pool_kernel)
        self.pool_stride = int(pool_stride)

        # "same-ish" padding on time dimension so W doesn't collapse
        pad_w = self.temporal_kernel // 2

        self.conv_time = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=(1, self.temporal_kernel),
            stride=(1, 1),
            padding=(0, pad_w),
            bias=True,
        )
        self.conv_chan = nn.Conv2d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=(self.c_in, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )

        self.bn = nn.BatchNorm2d(d_model)
        self.act = nn.ELU()
        self.drop = nn.Dropout(dropout)

        self.projection = nn.Conv2d(d_model, d_model, kernel_size=(1, 1), stride=(1, 1))

        self._printed = False  # debug once if you want

    def _to_B1CT(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            # [B,1,C,T]
            if x.shape[1] != 1 or x.shape[2] != self.c_in:
                raise ValueError(f"Expected [B,1,{self.c_in},T], got {tuple(x.shape)}")
            return x

        if x.dim() != 3:
            raise ValueError(f"Expected 3D/4D input, got dim={x.dim()} shape={tuple(x.shape)}")

        # [B,T,C]
        if x.shape[-1] == self.c_in:
            return x.permute(0, 2, 1).unsqueeze(1)  # -> [B,1,C,T]
        # [B,C,T]
        if x.shape[1] == self.c_in:
            return x.unsqueeze(1)                   # -> [B,1,C,T]

        raise ValueError(f"Channel mismatch: c_in={self.c_in}, got input shape={tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = self._to_B1CT(x)  # [B,1,C,T]

        # Optional debug (bật nếu cần):
        # if not self._printed:
        #     print("DEBUG ShallowNetEmbedding input:", tuple(x2.shape))
        #     self._printed = True

        x2 = self.conv_time(x2)   # [B,d_model,C,W]
        x2 = self.conv_chan(x2)   # [B,d_model,1,W]
        x2 = self.bn(x2)
        x2 = self.act(x2)

        # Safe pooling on time dimension (W)
        W = int(x2.shape[-1])
        if W >= self.pool_kernel:
            x2 = F.avg_pool2d(x2, kernel_size=(1, self.pool_kernel), stride=(1, self.pool_stride))
        else:
            # fallback: keep at least 1 token
            x2 = F.adaptive_avg_pool2d(x2, output_size=(1, 1))

        x2 = self.drop(x2)
        x2 = self.projection(x2)

        # [B, d_model, 1, W'] -> [B, W', d_model]
        x2 = rearrange(x2, "b d h w -> b (h w) d")
        return x2



class EEG2RepEmbedding(nn.Module):
    def __init__(self, c_in, d_model, pooling_size):
        super().__init__()

        k = 7
        # Embedding Layer -----------------------------------------------------------
        self.depthwise_conv = nn.Conv2d(in_channels=1, out_channels=d_model, kernel_size=(c_in, 1))
        self.spatial_padding = nn.ReflectionPad2d((int(np.floor((k - 1) / 2)), int(np.ceil((k - 1) / 2)), 0, 0))
        self.spatialwise_conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, k))
        self.spatialwise_conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, k))
        self.SiLU = nn.SiLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, pooling_size), stride=(1, pooling_size))

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(1)  # Shape becomes (B, 1, C, T)
        x = self.depthwise_conv(x)  # (B, d_model, 1 , T)
        x = x.transpose(1, 2)  # (B, 1, d_model, T)
        x = self.spatial_padding(x)
        x = self.spatialwise_conv1(x)  # (B, 1, d_model, T)
        x = self.SiLU(x)
        x = self.maxpool(x)  # (B, 1, d_model, T // pooling_size)
        x = self.spatial_padding(x)
        x = self.spatialwise_conv2(x)
        x = x.squeeze(1)  # (B, d_model, T // pooling_size)
        x = x.transpose(1, 2)  # (B, T // pooling_size, d_model)
        x = self.SiLU(x)

        return x


class CrossChannelTokenEmbedding(nn.Module):  # (batch_size, 1, enc_in, seq_len)
    def __init__(self, c_in, l_patch, d_model, stride=None):
        super().__init__()
        if stride is None:
            stride = l_patch
        self.tokenConv = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=(c_in, l_patch),
            stride=(1, stride),
            padding=0,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x)
        return x  # (batch_size, d_model, enc_in, patch_num)


class UpDimensionChannelEmbedding(nn.Module):  # B x C x T
    def __init__(self, c_in, t_in, u_dim, d_model):
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.u_dim = u_dim
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=u_dim,
            kernel_size=3,
            padding=padding,
            bias=False,
        )
        self.fc = nn.Linear(t_in, d_model)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x)  # B x u_dim x T
        x = self.fc(x)  # B x u_dim x d_model
        return x


class TokenChannelEmbedding(nn.Module):
    def __init__(
        self,
        enc_in,
        seq_len,
        d_model,
        patch_len_list,
        up_dim_list,
        stride_list,
        dropout,
        augmentation=["none"],
    ):
        super().__init__()
        self.patch_len_list = patch_len_list
        self.up_dim_list = up_dim_list
        self.stride_list = stride_list
        self.enc_in = enc_in
        self.paddings = [nn.ReplicationPad1d((0, stride)) for stride in stride_list]

        linear_layers_t = [
            CrossChannelTokenEmbedding(
                c_in=enc_in,
                l_patch=patch_len,
                d_model=d_model,
            )
            for patch_len in patch_len_list
        ]
        linear_layers_c = [
            UpDimensionChannelEmbedding(
                c_in=enc_in,
                t_in=seq_len,
                u_dim=u_dim,
                d_model=d_model,
            )  # c_in is seq_length here
            for u_dim in up_dim_list
        ]
        self.value_embeddings_t = nn.ModuleList(linear_layers_t)
        self.value_embeddings_c = nn.ModuleList(linear_layers_c)
        self.position_embedding_t = PositionalEmbedding(d_model=d_model)
        self.position_embedding_c = PositionalEmbedding(d_model=seq_len)
        self.dropout = nn.Dropout(dropout)
        self.augmentation = nn.ModuleList(
            [get_augmentation(aug) for aug in augmentation]
        )

        self.learnable_embeddings_t = nn.ParameterList(
            [nn.Parameter(torch.randn(1, d_model)) for _ in self.patch_len_list]
        )
        self.learnable_embeddings_c = nn.ParameterList(
            [nn.Parameter(torch.randn(1, d_model)) for _ in self.up_dim_list]
        )

    def forward(self, x):  # (batch_size, seq_len, enc_in)
        x = x.permute(0, 2, 1)  # (batch_size, enc_in, seq_len)

        x_list_t = []
        x_list_c = []
        for padding, value_embedding_t in zip(self.paddings, self.value_embeddings_t):
            x_copy = x.clone()
            # per granularity augmentation
            aug_idx = random.randint(0, len(self.augmentation) - 1)
            x_new_t = self.augmentation[aug_idx](x_copy)
            # temporal dimension
            x_new_t = padding(x_new_t).unsqueeze(1)  # (batch_size, 1, enc_in, seq_len+stride)
            x_new_t = value_embedding_t(x_new_t)  # (batch_size, d_model, 1, patch_num)
            x_new_t = x_new_t.squeeze(2).transpose(1, 2)  # (batch_size, patch_num, d_model)
            x_list_t.append(x_new_t)

        for value_embedding_c in self.value_embeddings_c:
            x_copy = x.clone()
            # per granularity augmentation
            aug_idx = random.randint(0, len(self.augmentation) - 1)
            x_new_c = self.augmentation[aug_idx](x_copy)
            # add positional embedding to tag each channel
            x_new_c = x_new_c + self.position_embedding_c(x_new_c)
            # channel dimension
            x_new_c = value_embedding_c(x_new_c)  # (batch_size, enc_in, d_model)
            x_list_c.append(x_new_c)

        x_t = [
            x + cxt + self.position_embedding_t(x)
            for x, cxt in zip(x_list_t, self.learnable_embeddings_t)
        ]  # (batch_size, patch_num_1, d_model), (batch_size, patch_num_2, d_model), ...
        x_c = [
            x + cxt
            for x, cxt in zip(x_list_c, self.learnable_embeddings_c)
        ]  # (batch_size, enc_in, d_model), (batch_size, enc_in, d_model), ...
        return x_t, x_c


class BIOTEmbedding(nn.Module):
    def __init__(
        self,
        enc_in,
        seq_len,
        d_model,
        patch_len,
        stride,
        augmentation=["none"],
    ):
        super().__init__()
        self.enc_in = int(enc_in)
        self.seq_len = int(seq_len)
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding = nn.ReplicationPad1d((0, stride))

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.channel_embedding = nn.Parameter(torch.randn(1, enc_in, seq_len) * 0.02)
        patch_num = int((seq_len - patch_len) / stride + 2)
        self.segment_embedding = nn.Parameter(torch.randn(1, patch_num * enc_in, d_model) * 0.02)

        self.augmentation = nn.ModuleList(
            [get_augmentation(aug) for aug in augmentation]
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        """
        Accept:
        - x: [B, T, C]
        - x: [B, C, T]
        Standardize -> x_ct: [B, C, T] with T == self.seq_len
        """
        if x.dim() != 3:
            raise ValueError(f"BIOTEmbedding expects 3D input, got {tuple(x.shape)}")

        C = int(self.enc_in)
        T0 = int(self.seq_len)

        # --- normalize to [B, C, T] ---
        if x.shape[1] == C:
            # already [B, C, T]
            x_ct = x
        elif x.shape[-1] == C:
            # [B, T, C] -> [B, C, T]
            x_ct = x.permute(0, 2, 1).contiguous()
        else:
            raise ValueError(f"Channel mismatch: enc_in={C}, got {tuple(x.shape)}")

        # --- enforce T == self.seq_len (segment_embedding/channel_embedding depend on this) ---
        T = int(x_ct.shape[-1])
        if T < T0:
            # pad right to T0 (replicate last value)
            x_ct = F.pad(x_ct, (0, T0 - T), mode="replicate")
        elif T > T0:
            x_ct = x_ct[..., :T0].contiguous()

        # --- augmentation: ONLY in training ---
        if self.training and len(self.augmentation) > 0:
            aug_idx = random.randint(0, len(self.augmentation) - 1)
            x_ct = self.augmentation[aug_idx](x_ct)

        # --- add channel embedding (broadcast) ---
        x_ct = x_ct + self.channel_embedding  # [B, C, T0]

        # keep original patching pipeline (expects [B,C,T])
        x_ct = self.padding(x_ct)  # (0, stride)
        x_ct = x_ct.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B,C,patch_num,patch_len]
        x_ct = rearrange(x_ct, "b e n l -> (b e) n l")  # [B*C, patch_num, patch_len]

        x_ct = self.value_embedding(x_ct) + self.position_embedding(x_ct)  # [B*C, patch_num, d_model]
        x_ct = rearrange(x_ct, "(b e) n d -> b (e n) d", e=C)              # [B, C*patch_num, d_model]
        return x_ct + self.segment_embedding