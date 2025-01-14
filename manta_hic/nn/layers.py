import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# -------- Convolutional blocks ---------


class ConvolutionalBlock1d(nn.Module):
    """
    A simple 1D convolutional block with group normalization, optional checkpointing, and GELU activation.

    Parameters
    ----------
    in_C : int
        Input channel dimension; must be divisible by 32 for the default group norm grouping.
    C : int
        Output channel dimension.
    W : int
        Kernel size of the 1D convolution.
    D : int, optional
        Dilation factor for the convolution (default=1).
    groups : int, optional
        Number of groups for grouped convolution (default=1).
    checkpoint_gn : bool, optional
        If True, applies checkpointing to the group norm for memory savings (default=False).
    """

    def __init__(self, in_C, C, W, *, D=1, groups=1, checkpoint_gn=False):
        super().__init__()
        self.groupnorm = nn.GroupNorm(num_groups=in_C // 32, num_channels=in_C)
        self.checkpoint_gn = checkpoint_gn
        self.conv = nn.Conv1d(in_C, C, kernel_size=W, padding=(W // 2) * D, dilation=D, groups=groups)

    def forward(self, x):
        fun = lambda x: F.gelu(self.groupnorm(x))
        x = checkpoint(fun, x, use_reentrant=True) if self.checkpoint_gn else fun(x)
        return self.conv(x)


class DoubleConvolutionalDownsampleBlock(nn.Module):
    """Downsample block with two convolutional layers, second with stride 2"""

    def __init__(self, in_C, C, W, D=1, groups=1, checkpoint_gn=True):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=in_C // 32, num_channels=in_C)
        self.gn2 = nn.GroupNorm(num_groups=C // 32, num_channels=C)
        self.do_check = checkpoint_gn
        self.conv = nn.Conv1d(in_C, C, kernel_size=W, padding=(W // 2) * D, dilation=D, groups=groups)
        self.conv2 = nn.Conv1d(C, C, kernel_size=4, stride=2, padding=1, groups=1)

    def forward(self, x):
        fun = lambda x: F.gelu(self.gn(x))
        fun2 = lambda x: F.gelu(self.gn2(x))

        x = checkpoint(fun, x, use_reentrant=True) if self.do_check else fun(x)
        x = self.conv(x)
        x = checkpoint(fun2, x, use_reentrant=True) if self.do_check else fun2(x)
        x = self.conv2(x)
        return x


class ConvolutionalBlock2d(nn.Module):
    """Convolutional block with group normalization and GELU activation."""

    def __init__(self, in_channels, out_channels, kernel_size, *, dilation=1, groups=1):
        super(ConvolutionalBlock2d, self).__init__()
        self.gn = nn.GroupNorm(in_channels // 8, in_channels)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size // 2) * dilation,
            dilation=dilation,
            groups=groups,
        )

    def forward(self, x):
        """Forward pass: [B, in_channels, H, W] -> [B, out_channels, H, W]"""
        return self.conv(F.gelu(self.gn(x)))


class ResidualDilatedBlock(nn.Module):
    """Residual block with two dilated ConvolutionalBlock2d layers and dropout."""

    def __init__(self, channels, kernel_size, dilation, dropout=0.2):
        super(ResidualDilatedBlock, self).__init__()
        self.conv1 = ConvolutionalBlock2d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = ConvolutionalBlock2d(channels, channels, kernel_size, dilation=dilation, groups=4)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass: [B, channels, H, W] -> [B, channels, H, W]
        """
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x + residual


class FibonacciResidualTower(nn.Module):
    """
    A stack of ResidualDilatedBlock modules with Fibonacci-like dilation rates (1,2,3,5,8, ...).

    The successive residual blocks each use two dilated ConvolutionalBlock2d layers plus dropout,
    returning the same shape [B, channels, H, W] after each block.

    Parameters
    ----------
    channels : int
        Number of input (and output) channels for each residual block.
    num_layers : int
        Number of Fibonacci-dilated blocks.
    kernel_size : int
        Kernel size for each ConvolutionalBlock2d.
    dropout : float, optional
        Dropout probability (default=0.2).
    """

    def __init__(self, channels, num_layers, kernel_size, dropout=0.2):
        super(FibonacciResidualTower, self).__init__()

        def fib(n):
            a, b = 1, 2
            for _ in range(n):
                yield a
                a, b = b, a + b

        self.dilation_rates = list(fib(num_layers))
        self.layers = nn.ModuleList(
            [
                ResidualDilatedBlock(channels, kernel_size, dilation=rate, dropout=dropout)
                for rate in self.dilation_rates
            ]
        )

    def forward(self, x):
        """
        Forward pass through all ResidualDilatedBlocks.
        [B, channels, H, W] -> [B, channels, H, W]
        """
        for layer in self.layers:
            x = layer(x)
        return x


# --------------------Hi-C specific modules (features to 2D, distance matrices, symmetrize) -----------------


class FeaturesTo2D(nn.Module):
    """
    Convert 1D features into a 2D representation by broadcast addition of two channel-halves,
    plus distance and upper/lower-triangle signals.

    Specifically, we split x into two halves along the channel dimension, then create a 2D grid:
    two_d[b, c, i, j] = x[b, c0, i] + x[b, c1, j], along with distance-based features and a
    triangular mask to distinguish upper/lower halves.

    Parameters
    ----------
    x : torch.Tensor
        1D features of shape [B, C, W].
    dist_mat : torch.Tensor
        Distance matrix of shape [1, H, W] or [1, 1, W, W] to be broadcast.

    Returns
    -------
    torch.Tensor
        2D features of shape [B, out_channels, W, W].
    """

    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(FeaturesTo2D, self).__init__()
        # We keep a relatively wide convolution here to encode distance, upper/lower half info, etc.
        self.conv = ConvolutionalBlock2d(in_channels, out_channels, kernel_size)

    def forward(self, x, dist_mat):
        """
        x: [B, C, W]  (1D features)
        dist_mat: [1, H, W]  (distance matrix, broadcast later)
        Returns: [B, out_channels, H, W]
        """
        # Split x into two halves and create a 2D feature via broadcast add
        # That is: X[:, :in_c, :] + X[:, in_c:, :]^T
        in_c = x.shape[1] // 2
        half_1 = x[:, :in_c, :]  # [B, in_c, W]
        half_2 = x[:, in_c:, :]  # [B, in_c, W]

        # Create an asymmetric matrix: half_1[:, :, i] + half_2[:, :, j]
        # Distinguish upper/lower triangular parts
        two_d = half_1.unsqueeze(-1) + half_2.unsqueeze(2)  # [B, in_c, W, W]

        # Distance features
        dist = dist_mat.unsqueeze(0).repeat(x.size(0), 1, 1, 1)  # [B, 1, W, W]

        # A +/-1 triangular mask so the network can learn upper vs. lower differently
        triu = torch.triu(torch.ones_like(dist), diagonal=1).to(x.device) * 2 - 1
        # Combine all features
        x2d = torch.cat((two_d, triu, dist, dist**2, dist * triu), dim=1)  # wide intermediate
        x2d = self.conv(x2d)
        return x2d


class Symmetrize(nn.Module):
    """
    Symmetrize a [B, C, H, W] tensor by averaging it with its transpose across the last two dims.
    """

    def forward(self, x):
        x_t = torch.transpose(x, -2, -1)
        return (x + x_t) / 2


# ------------------ Transformers ------------------------------


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


def precompute_freqs_cis(dim: int, N: int, theta: float = 10000.0) -> torch.Tensor:
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
        x = x + self._ff_block(self.norm2(x))
        return x

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff_dropout(self.ff_linear_2(F.gelu(self.ff_linear_1(x))))

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
    """A stack of FusedEncoderBlocks with permute, since main architecture is convolutional (channel first)"""

    def __init__(self, n_layers: int, d_model: int, n_bins: int, n_heads: int, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([FusedEncoderBlock(d_model, n_bins, n_heads, **kwargs) for _ in range(n_layers)])
        self.norm = nn.RMSNorm(d_model)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        return x
