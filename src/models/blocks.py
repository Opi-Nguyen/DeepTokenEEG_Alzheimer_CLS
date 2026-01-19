import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, causal=False):
        super(ResidualBlock1D, self).__init__()
        padding = (3 - 1) * dilation if causal else dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x
        out = self.conv1(x)
        out = self.relu(self.bn1(out))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :, :x.size(2)]

class Tokenizer(nn.Module):
    def __init__(self, enc_in, d_model, method='conv', kernel_size=7):
        super().__init__()
        if method == 'conv':
            self.tokenizer = nn.Sequential(
                nn.Conv1d(enc_in, d_model, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                nn.BatchNorm1d(d_model),
                nn.ReLU()
            )
        self.method = method

    def forward(self, x):
        return self.tokenizer(x.permute(0, 2, 1))
