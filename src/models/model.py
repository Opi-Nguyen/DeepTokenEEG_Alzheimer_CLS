import torch
import torch.nn as nn
from .blocks import ResidualBlock1D, Tokenizer

class Model(nn.Module):
    def __init__(self, enc_in: int, num_class: int, d_model: int, dropout: float,
                 n_blocks: int, dilations):
        super().__init__()
        self.tokenizer = Tokenizer(enc_in, d_model, method="conv", kernel_size=7)

        # pick dilations for ablation
        dilations = dilations[:n_blocks]
        self.res_blocks = nn.ModuleList([
            ResidualBlock1D(d_model, d_model, dilation=d) for d in dilations
        ])
        self.cross_res = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_class)

    def forward_backbone(self, x):
        # x: [B, T, C] -> tokenizer expects [B, T, C] then permute inside Tokenizer
        x = self.tokenizer(x)              # [B, d_model, T]
        residual = x
        for block in self.res_blocks:
            x = block(x)
            x = x + self.cross_res(residual)
            residual = x
        feat = torch.mean(x, dim=2)        # [B, d_model]
        return feat, x                     # return both (pre-classifier vec, tokenizer/resnet map)

    def forward(self, x):
        feat, _map = self.forward_backbone(x)
        return self.classifier(self.dropout(feat))
