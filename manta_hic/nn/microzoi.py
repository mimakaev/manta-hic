"""
A "micro Borzoi" model for predicting gene expression from DNA sequences.

Main differences from Borzoi are:
* Model operates at a single 256bp resolution
* There is no U-net
* Using modern transformer architecture with rotary embeddings and SDPA
"""

import h5py
import hdf5plugin
import numpy as np
import polars as pl
import pysam
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from manta_hic.nn.layers import ConvolutionalBlock1d, TransformerTower
from manta_hic.ops.seq_ops import make_seq_1hot
from manta_hic.training_meta import get_seqs_targs


class MicroBorzoi(nn.Module):
    def __init__(
        self,
        seq_conv_width=15,
        tower_mults=[8, 10, 12, 14, 16, 16, 16, 16],
        groups=[1, 1, 1, 1, 1, 1, 1],
        conv_block="single",
        width=[3, 3, 3, 3, 3, 3, 3],
        base_channels=64,
        n_heads=16,
        ff_mult=4,
        transf_layers=12,
        seq_length=2**19 + 2**18,
        last_channels_mult=32,
        output_channels_human=7611,
        output_channels_mouse=2608,
        attn_dropout=0.3,
        conv_dropout=0.2,
        num_bn_checkpoints=1,
        checkpoint_first=False,
        return_type="default",
    ):
        super().__init__()
        # Hard-coding those as Borzoi training data never changes
        self.nbins = seq_length // 256
        self.crop = (seq_length // 32 - 6144) // 8 // 2

        self.conv_block_type = conv_block
        self.check_first = checkpoint_first
        self.channels_1d = [base_channels * i for i in tower_mults]
        ch_1d = self.channels_1d
        working_channels = self.working_channels = ch_1d[-1]
        self.last_chanels = base_channels * last_channels_mult
        self.nheads = n_heads
        self.output_channels_human = output_channels_human
        self.output_channels_mouse = output_channels_mouse
        self.return_type = return_type

        # utility layers
        self.dropout = nn.Dropout1d(p=conv_dropout)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Convolutional layers
        self.conv0 = nn.Conv1d(4, ch_1d[0], kernel_size=seq_conv_width, padding=10)

        self.conv_blocks = nn.ModuleList()
        if conv_block == "single":
            convBlock = ConvolutionalBlock1d
        else:
            raise ValueError("Only single blocks are supported")

        for ind, (c_st, c_end, gr, W) in enumerate(zip(self.channels_1d[:-1], self.channels_1d[1:], groups, width)):
            do_checkpoint = ind < num_bn_checkpoints
            self.conv_blocks.append(convBlock(c_st, c_end, W=W, groups=gr, checkpoint_bn=do_checkpoint))

        self.mha_tower = TransformerTower(
            n_layers=transf_layers,
            d_model=working_channels,
            n_bins=self.nbins,
            n_heads=n_heads,
            drop_p=attn_dropout,
            ff_mult=ff_mult,
        )
        if self.return_type in ["default"]:
            self.output_conv_block = ConvolutionalBlock1d(working_channels, self.last_chanels, W=1, D=1, groups=1)
            self.output_conv_block2 = ConvolutionalBlock1d(self.last_chanels, self.last_chanels, W=3, D=1, groups=8)

            self.final_conv_human = nn.Conv1d(self.last_chanels, self.output_channels_human, kernel_size=1)
            self.final_conv_mouse = nn.Conv1d(self.last_chanels, self.output_channels_mouse, kernel_size=1)

    def forward(self, x, genome="hg38", offset=0, crop_mha=0):

        # First convolutional layer (possibly checkpointed - it's the biggest)
        first = lambda x: self.maxpool(self.conv0(x))
        x = checkpoint(first, x, use_reentrant=True) if self.check_first else first(x)

        # Convolutional blocks - double block has stride 2 so no need for maxpool
        for conv_block in self.conv_blocks:
            x = self.maxpool(conv_block(x))

        x = self.mha_tower(x)

        if self.return_type == "mha":
            return x[:, :, crop_mha : x.shape[2] - crop_mha]  # crop on device to make it faster

        # Crop after transformers - the rest is just local convolutions
        crop_st, crop_end = self.crop + offset, x.shape[2] - self.crop + offset
        x = x[:, :, crop_st:crop_end]

        x = self.output_conv_block(x)  # groups=1, w=1
        x = self.dropout(x)
        x = self.output_conv_block2(x)  # groups=8, w=3
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


def dataGenerator(
    *,
    base,
    genomes,
    seq_length,
    fasta_path,
    resolution,  # end storage parameters
    batch_size=1,  # start training/validation parameters
    val_fold=3,
    test_fold=4,
    max_shift=0,
    mode="val",
):
    """Ranks and world size aware data generator that would alternate hg38 and mm10 blocks"""

    fastas = {name: pysam.FastaFile(fasta_path.format(name)) for name in genomes}

    dfs = []

    val_fold = int(val_fold)
    test_fold = int(test_fold)
    omit_folds = [val_fold, test_fold]
    for gen in genomes:
        seqdf, _ = get_seqs_targs(gen)
        # add index and genome, convert fold to int, add seq start and end
        seqdf = seqdf.with_columns(
            pl.int_range(pl.len()).over("fold").alias("index"),
            pl.col("fold").str.replace("fold", "").cast(int),
            ((pl.col("start") + pl.col("end")) // 2 - seq_length // 2).alias("seq_start"),
            ((pl.col("start") + pl.col("end")) // 2 + seq_length // 2).alias("seq_end"),
        )
        # add file path per fold and keys
        seqdf = seqdf.with_columns(
            (pl.lit(base + "/" + gen + "/fold") + pl.col("fold").cast(str) + pl.lit(".h5")).alias("file"),
            (pl.lit("sample") + pl.col("index").cast(str)).alias("key"),
        )

        if mode == "train":
            seqdf = seqdf.filter(~pl.col("fold").cast(int).is_in(omit_folds))
        elif mode == "val":
            seqdf = seqdf.filter(pl.col("fold").cast(int) == val_fold)
        elif mode == "test":
            seqdf = seqdf.filter(pl.col("fold").cast(int) == test_fold)
        elif mode != "all":
            raise ValueError("Mode should be train, val, test, or all")
        dfs.append(seqdf[: len(seqdf) // batch_size * batch_size])  # truncate to batch size

    # concat and shuffle in blocks of batch_size - lol polars is good!
    seqdf = pl.concat(dfs).sort((pl.int_range(pl.len()) // batch_size).hash(np.random.randint(0, 1e9)))

    for batch in seqdf.iter_slices(batch_size):
        if mode == "train":
            shift_bins = np.random.randint(-max_shift, max_shift)
        else:
            shift_bins = 0
        shift_bp = shift_bins * resolution

        global_meta = {"shift_bins": shift_bins, "genome": batch["genome"][0]}
        batch_arrs = [[], []]
        metadatas = []
        for row in batch.iter_rows(named=True):
            with h5py.File(row["file"], "r") as myfile:
                d = myfile[row["key"] + "/data"][:]
                batch_arrs[1].append(d)

                fa = fastas[row["genome"]]
                seq_hot = make_seq_1hot(fa, row["chrom"], row["seq_start"] - shift_bp, row["seq_end"] - shift_bp)
                batch_arrs[0].append(seq_hot)

                metadatas.append(row)
        yield (batch_arrs, global_meta, metadatas)
