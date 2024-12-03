import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ConvolutionalBlock1d(nn.Module):
    """Convolutional block with batch normalization and GELU activation"""

    def __init__(self, in_C, C, W, *, D=1, groups=1, checkpoint_bn=False):
        "A simple convolutional block with batch normalization and GELU activation"

        super().__init__()
        self.bn = nn.BatchNorm1d(in_C)
        self.checkpoint_bn = checkpoint_bn
        self.conv = nn.Conv1d(in_C, C, kernel_size=W, padding=(W // 2) * D, dilation=D, groups=groups)

    def forward(self, x):
        fun = lambda x: F.gelu(self.bn(x))
        x = checkpoint(fun, x, use_reentrant=True) if self.checkpoint_bn else fun(x)
        return self.conv(x)


class DoubleConvolutionalDownsampleBlock(nn.Module):
    """Downsample block with two convolutional layers, second with stride 2"""

    def __init__(self, in_C, C, W, D=1, groups=1, checkpoint_bn=True):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_C)
        self.bn2 = nn.BatchNorm1d(C)
        self.do_check = checkpoint_bn
        self.conv = nn.Conv1d(in_C, C, kernel_size=W, padding=(W // 2) * D, dilation=D, groups=groups)
        self.conv2 = nn.Conv1d(C, C, kernel_size=4, stride=2, padding=1, groups=1)

    def forward(self, x):
        fun = lambda x: F.gelu(self.bn(x))
        fun2 = lambda x: F.gelu(self.bn2(x))

        x = checkpoint(fun, x, use_reentrant=True) if self.do_check else fun(x)
        x = self.conv(x)
        x = checkpoint(fun2, x, use_reentrant=True) if self.do_check else fun2(x)
        x = self.conv2(x)
        return x


# This code is adapted from Llama 2 materials provided by Meta Platforms, Inc.
# Licensed under the LLAMA 2 Community License, Copyright (c) Meta Platforms, Inc. All Rights Reserved.
# Changes made: used SDPA, added dropout, and changed the FF block, plus heavy overal rewriting
def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    "Apply the rotary embedding to the KQ tensors"
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # [B, N, H, C/H] -> [B, N, H, C/(2*H)]
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))  # dtype: complex64
    if not freqs_cis.shape == (xq_.shape[1], xq_.shape[-1]):
        raise ValueError(f"freqs_cis mismatch: {freqs_cis.shape}, {(xq_.shape[1], xq_.shape[-1])}")
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)  # [B, N, H, C/(2*H)] -> [B, N, H, C/H]
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)  # dtype: float32
    return xq_out.type_as(xq), xk_out.type_as(xk)  # dtype: float32/(b)float16 as autocast wishes


def precompute_freqs_cis(dim: int, N: int, theta: float = 40000.0) -> torch.Tensor:
    "Slightly longer theta because llama3 has it and because we may use it up to 16k context size"
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))  # [dim//2]
    t = torch.arange(N, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # [N, dim//2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # dtype: complex64
    return freqs_cis  # [N, dim//2] where dim is d_model//n_heads


# Adopted from the post by a github user loubbrad https://github.com/pytorch/pytorch/issues/97899
class FusedEncoderBlock(nn.Module):  # also from llama
    """Transformer encoder block using F.scaled_dot_product_attention()"""

    def __init__(self, d_model: int, n_bins: int, n_heads: int = 8, drop_p: float = 0.2, ff_mult: int = 4):
        super().__init__()
        self.drop_p = drop_p
        self.n_heads = n_heads
        self.n_bins = n_bins
        self.register_buffer("freqs_cis", precompute_freqs_cis(d_model // self.n_heads, self.n_bins))
        self.d_head = d_model // n_heads

        # Attention
        self.q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.att_proj_linear = nn.Linear(in_features=d_model, out_features=d_model)
        self.resid_dropout = nn.Dropout(drop_p)

        # FF Layer
        self.ff_dropout = nn.Dropout(drop_p)
        self.ff_linear_1 = nn.Linear(in_features=d_model, out_features=d_model * ff_mult, bias=False)
        self.ff_linear_2 = nn.Linear(in_features=d_model * ff_mult, out_features=d_model, bias=False)

        # Pre layer norms
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._att_block(self.norm1(x), self.freqs_cis)
        return x + self.ff_dropout(self.ff_linear_2(nn.GELU(self.ff_linear_1(x))))

    def _att_block(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.q(x), self.k(x), self.v(x)

        # Reshape for rotary embeddings
        xq = xq.view(batch_size, seq_len, self.n_heads, self.d_head)
        xk = xk.view(batch_size, seq_len, self.n_heads, self.d_head)
        xv = xv.view(batch_size, seq_len, self.n_heads, self.d_head)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Reshape for attention calculation: (b_sz, n_head, s_len, d_head)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        att_dropout = self.drop_p if self.training else 0  # Required as we are not using a nn.Dropout layer
        att = F.scaled_dot_product_attention(query=xq, key=xk, value=xv, dropout_p=att_dropout, is_causal=False)

        out = att.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.n_heads * self.d_head)
        return self.resid_dropout(self.att_proj_linear(out))


class TransformerTower(nn.Module):
    """A stack of FusedEncoderBlocks with permite, since main architecture is convolutional (channel first)"""

    def __init__(self, n_layers: int, d_model: int, n_bins: int, n_heads: int, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([FusedEncoderBlock(d_model, n_bins, n_heads, **kwargs) for _ in range(n_layers)])

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for layer in self.layers:
            x = layer(x)
        x = x.permute(0, 2, 1)
        return x
