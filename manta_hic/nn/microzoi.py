"""
A "micro Borzoi" model for predicting gene expression from DNA sequences. 

Main differences from Borzoi are: 
* Model operates at a single 256bp resolution 
* There is no U-net 
* Using modern transformer architecture with rotary embeddings and SDPA
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from manta_hic.nn.layers import (
    ConvolutionalBlock1d,
    DoubleConvolutionalDownsampleBlock,
    TransformerTower,
)


class MicroBorzoi(nn.Module):
    def __init__(
        self,
        seq_conv_width=15,
        tower_mults=[8, 10, 12, 14, 16, 16, 16, 16],
        groups=[1, 1, 1, 1, 1, 1, 1],
        conv_block="double",
        width=[3, 3, 3, 3, 3, 3, 3],
        base_channels=64,
        n_heads=16,
        transf_layers=12,
        seq_length=2**19 + 2**18,
        last_channels_mult=32,
        output_channels_human=7611,
        output_channels_mouse=2608,
        attn_dropout=0.3,
        conv_dropout=0.2,
        num_bn_checkpoints=1,
        checkpoint_first=False,
    ):
        super().__init__()
        # Hard-coding those as Borzoi training data never changes
        self.nbins = seq_length // 256
        self.crop = (seq_length // 32 - 6144) // 8 // 2

        self.conv_block_type = "single"
        self.check_first = checkpoint_first
        self.channels_1d = [base_channels * i for i in tower_mults]
        ch_1d = self.channels_1d
        working_channels = self.working_channels = ch_1d[-1]
        self.last_chanels = base_channels * last_channels_mult
        self.nheads = n_heads
        self.output_channels_human = output_channels_human
        self.output_channels_mouse = output_channels_mouse

        # utility layers
        self.dropout = nn.Dropout1d(p=conv_dropout)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Convolutional layers
        self.conv0 = nn.Conv1d(4, ch_1d[0], kernel_size=seq_conv_width, padding=10)

        self.conv_blocks = nn.ModuleList()
        if conv_block == "single":
            convBlock = ConvolutionalBlock1d
        elif conv_block == "double":
            convBlock = DoubleConvolutionalDownsampleBlock
        for ind, (c_st, c_end, gr, W) in enumerate(zip(self.channels_1d[:-1], self.channels_1d[1:], groups, width)):
            do_checkpoint = ind < num_bn_checkpoints
            self.conv_blocks.append(convBlock(c_st, c_end, W=W, groups=gr, checkpoint_bn=do_checkpoint))

        self.mha_tower = TransformerTower(
            n_layers=transf_layers,
            d_model=working_channels,
            n_bins=self.nbins,
            n_heads=n_heads,
            drop_p=attn_dropout,
        )

        self.output_conv_block = ConvolutionalBlock1d(working_channels, self.last_chanels, W=1, D=1, groups=1)
        self.output_conv_block2 = ConvolutionalBlock1d(self.last_chanels, self.last_chanels, W=3, D=1, groups=8)

        self.final_conv_human = nn.Conv1d(self.last_chanels, self.output_channels_human, kernel_size=1)
        self.final_conv_mouse = nn.Conv1d(self.last_chanels, self.output_channels_mouse, kernel_size=1)

    def forward(self, x, genome="hg38", offset=0):

        # First convolutional layer (possibly checkpointed - it's the biggest)
        first = lambda x: self.maxpool(self.conv0(x))
        x = checkpoint(first, x, use_reentrant=True) if self.check_first else first(x)

        # Convolutional blocks - double block has stride 2 so no need for maxpool
        for conv_block in self.conv_blocks:
            if self.conv_block_type == "single":
                x = self.maxpool(conv_block(x))
            else:
                x = conv_block(x)

        x = self.mha_tower(x)

        # Crop after transformers - the rest is just local convolutions
        x = x[:, :, self.crop + offset : x.shape[2] - self.crop + offset]
        x = self.output_conv_block(x)  # groups=1, w=1
        x = self.dropout(x)
        x = self.output_conv_block2(x)  # groups=8, w=3
        x = self.dropout(x)
        x = F.gelu(x)

        if genome == "hg38":
            x = self.final_conv_human(x)
        elif genome == "mm10":
            x = self.final_conv_mouse(x)
        else:
            raise ValueError("Target can be hg38 or mm10")
        x = F.softplus(x)
        return x


def borzoi_loss(output, target, total_weight=0.2):
    "Multinomial-Poisson loss like in Borzoi"
    epsilon = 1e-7
    total_weight = 0.2

    seq_len = output.shape[2]

    s_targ = target.sum(dim=2) + 1e-4  # (batch_size, num_targets)
    s_pred = output.sum(dim=2) + 1e-4

    targ = target + epsilon  # (batch_size, num_targets, seq_len)
    pred = output + epsilon

    poisson_loss = (s_pred - s_targ * torch.log(s_pred + epsilon)).mean() / seq_len  # sum(L) -> mean

    p_pred = pred / s_pred.unsqueeze(2)  # probability of prediction
    multinomial_loss = (-targ * torch.log(p_pred)).mean()  # mean(batch, track)

    return multinomial_loss + total_weight * poisson_loss


def corr(target, output, return_individual=False):
    "Fast correlation between target and output"
    t_mean, p_mean = target - target.mean(dim=-1, keepdim=True), output - output.mean(dim=-1, keepdim=True)
    cr = (t_mean * p_mean).sum(dim=-1) / (1e-5 + (t_mean.pow(2).sum(dim=-1) * p_mean.pow(2).sum(dim=-1)).sqrt())

    cr_mean = cr.mean().item()
    if return_individual:
        cr = cr.detach().cpu().numpy()
        return cr_mean, cr
    return cr_mean
