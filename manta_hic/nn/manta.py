import datetime as dt
import queue
import threading
from contextlib import nullcontext

import h5py
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from manta_hic.nn.layers import (
    ConvolutionalBlock1d,
    ConvolutionalBlock2d,
    FeaturesTo2D,
    FibonacciResidualTower,
    Symmetrize,
    TransformerTower,
)
from manta_hic.ops.hic_ops import (
    coarsegrained_hic_corrs,
    create_expected_matrix,
    hic_hierarchical_loss,
)
from manta_hic.ops.tensor_ops import list_to_tensor_batch
from manta_hic.training_meta import assign_fold_type


def fetch_activations(genome, chrom, start, end, reverse, cache_file="microzoi_cache.h5py"):
    """
    Fetches the cached activations for a given region.

    Parameters
    ----------
    genome : str
        Genome assembly (e.g. hg38).
    chrom : str
        Chromosome name.
    start : int
        Start coordinate in base pairs.
    end : int
        End coordinate in base pairs.
    reverse : bool
        Whether to fetch the reverse strand.
    cache_file : str
        Path to the cache file.
    """
    CHUNK_BP = 2**19
    BIN_BP = 256
    CHUNK_BINS = CHUNK_BP // BIN_BP

    if start % BIN_BP != 0 or end % BIN_BP != 0:
        raise ValueError("Start/end must be aligned to 256bp.")
    if end <= start:
        raise ValueError("End must be > start.")

    start_bin, end_bin = start // BIN_BP, end // BIN_BP
    num_bins = end_bin - start_bin
    chunk_start, chunk_end = start_bin // CHUNK_BINS, (end_bin - 1) // CHUNK_BINS
    rng = range(chunk_start, chunk_end + 1) if not reverse else range(chunk_end, chunk_start - 1, -1)
    suffix = "normal" if not reverse else "reverse"

    acts = []
    with h5py.File(cache_file, "r") as f:
        for cidx in rng:
            cstart = cidx * CHUNK_BP
            cend = cstart + CHUNK_BP
            cstart_bin, cend_bin = cstart // BIN_BP, cend // BIN_BP

            sbin = max(start_bin, cstart_bin)
            ebin = min(end_bin, cend_bin)
            sl = sbin - cstart_bin
            el = ebin - cstart_bin
            if reverse:
                sl, el = CHUNK_BINS - el, CHUNK_BINS - sl

            dsname = f"{genome}/{chrom}/{cstart}_{cend}_{suffix}"
            if dsname not in f:
                raise KeyError(f"{dsname} not found in cache.")
            acts.append(f[dsname][:, sl:el])

    if not acts:
        raise ValueError("No activation slices fetched. Check coordinates.")
    out = np.concatenate(acts, axis=1).astype(np.float16)
    if out.shape[1] != num_bins:
        raise ValueError(f"Expected {num_bins} bins, got {out.shape[1]}")
    return out


class HiCDataset:
    """
    A dataset class for handling Hi-C data.
    Parameters
    ----------
    filename : str
        Path to the HDF5 file containing Hi-C data.
    nbins : int
        Number of bins to use for Hi-C data slices.
    hic_res : int
        Resolution of Hi-C data in base pairs.
    pad : int
        Padding to add around Hi-C data slices.
    test_fold : str, optional
        Name of the fold to use for testing data (default is "fold3").
    val_fold : str, optional
        Name of the fold to use for validation data (default is "fold4").
    training : bool, optional
        Whether to use the dataset for training (default is True).
    random : bool, optional
        Whether to use random starting positions for Hi-C data slices (default is True).
    cache_file : str, optional
        Path to the cache file for storing intermediate results (default is "../data_ssd/microzoi_cache.h5py").
    Attributes
    ----------
    filename : str
        Path to the HDF5 file containing Hi-C data.
    nbins : int
        Number of bins to use for Hi-C data slices.
    hic_res : int
        Resolution of Hi-C data in base pairs.
    pad : int
        Padding to add around Hi-C data slices.
    random : bool
        Whether to use random starting positions for Hi-C data slices.
    cache_file : str
        Path to the cache file for storing intermediate results.
    df : polars.DataFrame
        DataFrame containing metadata and fold information for the dataset.
    M : int
        Number of bins in the Hi-C data.
    Methods
    -------
    __len__()
        Returns the number of samples in the dataset.
    __getitem__(idx, stochastic_reverse=True)
        Retrieves a sample from the dataset at the specified index.
    get_single(idx)
        Retrieves a sample from the dataset at the specified index without random shifts or reverse.
    """

    def __init__(
        self,
        filename,
        nbins,
        hic_res,
        pad,
        test_fold="fold3",
        val_fold="fold4",
        training=True,
        random=True,
        cache_file="../data_ssd/microzoi_cache.h5py",
    ):
        self.filename = filename
        self.nbins = nbins
        self.hic_res = hic_res
        self.pad = pad
        self.random = random
        self.cache_file = cache_file

        with h5py.File(filename, "r") as f:
            df = pl.DataFrame({"chrom": f["chrom"][:], "start": f["start"][:], "end": f["end"][:]})
            self.M = f["hic"].shape[-1]
        df = df.with_columns(pl.col("chrom").cast(str))
        # verify that end-start = hic_res * nbins
        # assert (df["end"] - df["start"] == hic_res * nbins).all()

        df = assign_fold_type(df, test_fold=test_fold, val_fold=val_fold, genome="hg38")
        df = df.with_columns(pl.int_range(pl.len()).alias("index"))
        if training:
            df = df.filter(pl.col("fold_type") == "train")
        else:
            df = df.filter(pl.col("fold_type") == "test")
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx, stochastic_reverse=True):
        row = self.df.row(idx, named=True)
        orig_idx = row["index"]
        start = np.random.randint(0, self.M - self.nbins) if self.random else 0
        start_bp = row["start"] + start * self.hic_res - self.pad * self.hic_res
        end_bp = start_bp + self.nbins * self.hic_res + 2 * self.pad * self.hic_res
        use_reverse = stochastic_reverse and np.random.rand() > 0.5

        with h5py.File(self.filename, "r") as f:
            hic_slice = f["hic"][orig_idx, :, start : start + self.nbins, start : start + self.nbins]
            weightmat = f["weights"][orig_idx, :, start : start + self.nbins]
            exp = f["exp"][orig_idx]
        if use_reverse:
            hic_slice = hic_slice[:, ::-1, ::-1].copy()
            weightmat = weightmat[:, ::-1].copy()

        interms = fetch_activations("hg38", row["chrom"], start_bp, end_bp, use_reverse, self.cache_file)
        return interms, hic_slice, weightmat, exp

    def get_single(self, idx):
        # direct snippet fetch, no random shifts or reverse
        return self.__getitem__(idx, stochastic_reverse=False)


class ThreadedDataLoader:
    """
    A data loader that loads data in a separate thread and uses a queue to store batches.
    Parameters
    ----------
    dataset : Dataset
        The dataset from which to load the data.
    batch_size : int, optional
        The number of samples per batch to load (default is 1).
    shuffle : bool, optional
        Whether to shuffle the data before loading (default is True).
    fraction : float, optional
        The fraction of the dataset to load (default is 1.0). 0.25 is recommended as default tile have 25% overlap.
    queue_size : int, optional
        The maximum size of the queue to store batches (default is 3).
    Methods
    -------
    __iter__()
        Returns an iterator that yields batches of data.
    get_item(idx)
        Returns a single item from the dataset at the specified index.
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, fraction=1.0, queue_size=3):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.fraction = fraction
        self.q = queue.Queue(maxsize=queue_size)

    def _loader_thread(self):
        need = int(len(self.dataset) * self.fraction)
        indices = np.random.choice(len(self.dataset), need, replace=False)
        if not self.shuffle:
            indices = np.sort(indices)
        for i in range(0, len(indices), self.batch_size):
            batch_idx = indices[i : i + self.batch_size]
            batch = [self.dataset[j] for j in batch_idx]
            if len(batch) < self.batch_size:
                break
            # transpose so that interms, hic_slice, weightmat, exp become separate lists
            batch_t = list(map(list, zip(*batch)))
            self.q.put(tuple(batch_t))
        self.q.put(None)

    def __iter__(self):
        thread = threading.Thread(target=self._loader_thread)
        thread.start()
        while True:
            data = self.q.get()
            if data is None:
                break
            yield data
        thread.join()

    def get_item(self, idx):
        return ([i] for i in self.dataset.get_single(idx))


def run_epoch(model, dataloader, device, is_train=True, optimizer=None, scaler=None):
    corrs = []

    model.train() if is_train else model.eval()

    for batch in dataloader:
        t0 = dt.datetime.now()

        batch = [list_to_tensor_batch(i, device) for i in batch]
        acts, target, weight, exp = batch
        if is_train:
            acts.requires_grad = True

        target, weightmat = create_expected_matrix(target, weight, exp)

        if is_train:
            optimizer.zero_grad()

        with autocast(device_type="cuda"), torch.no_grad() if not is_train else nullcontext():
            output = model(acts)
            if is_train:
                loss = hic_hierarchical_loss(output, target, weightmat)
            corr = [i.detach().cpu().numpy() for i in coarsegrained_hic_corrs(output, target, weight, exp)]
            corr = np.array(corr)

        corrs.append(corr)

        # Backprop/update only if training
        if is_train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Simple progress print
        duration = (dt.datetime.now() - t0).total_seconds()
        cr = ", ".join([f"{i.mean():.4f}" for i in corr])
        print(f"[{'Train' if is_train else 'Val'}] spearm/pears/msd = {cr}, duration={duration:.3f} s   ", end="\r")

    return np.array(corrs)


def calculate_distance_matrix(n_bins):
    """
    Compute a log10-based distance matrix, centered and scaled,
    then return as a [1, H, W] tensor for broadcast.
    """
    i, j = np.indices((n_bins, n_bins))
    dist_mat = np.log10(np.abs(i - j) + 3)
    dist_mat = (dist_mat - np.mean(dist_mat)) / np.std(dist_mat)
    dist_mat = torch.from_numpy(dist_mat).float().unsqueeze(0)
    return dist_mat


class Manta2(nn.Module):
    """
    Main model class that combines:
      1) A 1D convolutional backbone (with pooling and a Transformer tower).
      2) Two parallel 2D branches (direct and tower) converting the final 1D features to 2D.
      3) A final merge via additional 2D convolutions and symmetrization.

    We assume the input length (the 1D sequence dimension) is:
        2^(tower_height + 1) * (n_bins + 2 * bins_pad)
    so that repeated pooling steps ultimately arrive at (n_bins + 2 * bins_pad) for the 2D branches.
    After the final 2D operations, the output is cropped (on the 1D side) to n_bins and reshaped to
    [B, output_channels, n_bins, n_bins].

    Parameters
    ----------
    n_bins : int
        Number of 1D bins in the final 2D output (excluding padding).
    bins_pad : int
        Amount of padding on each side of the input that will be cropped before 2D conversion.
    input_channels : int
        Number of input channels. This is convolved to channels_1d initially.
    channels_1d : int
        Channel dimension for the 1D backbone.
    tower_height : int
        Number of 1D conv+pool blocks (in total) before the final pool that leads to n_bins + 2*bins_pad.
        Effectively, this sets the number of half-pool operations. A value of H requires an input length
        2^(H+1) * (n_bins + 2*bins_pad). Cannot be zero.
    transformer_layers : int
        Number of transformer layers in the TransformerTower.
    transformer_dropout : float
        Dropout rate in the transformer.
    transformer_n_heads : int
        Number of attention heads in the transformer.
    direct_2d_input_channels : int
        Intermediate channel width before final 2D conv for the "direct" branch.
    direct_2d_input_width : int
        Kernel size for the direct branch 2D conv.
    direct_2d_channels : int
        Final 2D channel dimension for the direct branch (before merging).
    tower_2d_input_channels : int
        Intermediate channel width for the "tower" branch before the residual tower.
    tower_2d_input_width : int
        Kernel size for the tower branch 2D conv.
    tower_2d_channels : int
        Channel dimension for the tower branch 2D representation.
    tower_2d_width : int
        Kernel size for the residual blocks in the tower.
    tower_2d_dropout : float
        Dropout rate for the residual blocks in the tower.
    tower_2d_height : int
        Number of residual dilated blocks in the tower (Fibonacci dilation).
    final_channels : int
        Channel dimension after joining the two 2D branches.
    output_channels : int
        Number of output channels in the final 2D convolution.
    """

    def __init__(
        self,
        *,
        n_bins=1024,
        bins_pad=64,
        input_channels=1024,
        channels_1d=1024,
        tower_height=2,
        transformer_layers=10,
        transformer_dropout=0.4,
        transformer_n_heads=16,
        direct_2d_input_channels=128,
        direct_2d_input_width=5,
        direct_2d_channels=64,
        tower_2d_input_channels=256,
        tower_2d_input_width=11,
        tower_2d_channels=64,
        tower_2d_width=9,
        tower_2d_dropout=0.2,
        tower_2d_height=8,
        final_channels=64,
        output_channels=2,
    ):
        super(Manta2, self).__init__()
        self.n_bins = n_bins
        self.bins_pad = bins_pad
        self.channels_1d = channels_1d

        # Precompute distance matrices for full (n_bins x n_bins) and half ((n_bins//2) x (n_bins//2))
        dist_mat_full = calculate_distance_matrix(n_bins)
        dist_mat_half = calculate_distance_matrix(n_bins // 2)
        self.register_buffer("dist_mat_full", dist_mat_full, persistent=False)
        self.register_buffer("dist_mat_half", dist_mat_half, persistent=False)

        # 1D backbone
        self.first_conv_1d = nn.Conv1d(input_channels, channels_1d, kernel_size=5, padding=2)
        self.maxpool1d = nn.MaxPool1d(kernel_size=2, stride=2)

        # Multiple conv blocks + pooling
        self.conv_blocks_1d = nn.ModuleList()
        for _ in range(tower_height):
            self.conv_blocks_1d.append(ConvolutionalBlock1d(channels_1d, channels_1d, 3, groups=1))

        # Transformer tower (assume it has RMSNorm inside or appended)
        self.mha_tower = TransformerTower(
            n_layers=transformer_layers,
            d_model=channels_1d,
            n_bins=2 * (n_bins + 2 * bins_pad),
            n_heads=transformer_n_heads,
            drop_p=transformer_dropout,
        )

        # Direct 2D branch
        # Reserve some channels for the extra distance/upper-lower features in FeaturesTo2D
        self.conv_direct_1d = ConvolutionalBlock1d(channels_1d, 2 * direct_2d_input_channels - 8, 1)
        self.features_to_2d_direct = FeaturesTo2D(
            direct_2d_input_channels, direct_2d_channels, kernel_size=direct_2d_input_width
        )

        # Tower 2D branch
        self.conv_tower_1d = ConvolutionalBlock1d(channels_1d, 2 * tower_2d_input_channels - 8, 1)
        self.maxpool1d_tower = nn.MaxPool1d(kernel_size=2, stride=2)
        self.features_to_2d_tower = FeaturesTo2D(
            tower_2d_input_channels, tower_2d_channels, kernel_size=tower_2d_input_width
        )

        # Residual dilated tower
        self.residual_dilated_tower = FibonacciResidualTower(
            tower_2d_channels, tower_2d_height, tower_2d_width, dropout=tower_2d_dropout
        )
        self.batchnorm_tower = nn.BatchNorm2d(tower_2d_channels, momentum=0.01)

        # 2D deconv + groupnorm
        self.deconv = nn.ConvTranspose2d(tower_2d_channels, tower_2d_channels, kernel_size=2, stride=2, groups=4)
        self.gn_deconv = nn.GroupNorm(tower_2d_channels // 8, tower_2d_channels)

        # Final join
        self.join_conv = ConvolutionalBlock2d(direct_2d_channels + tower_2d_channels, final_channels, kernel_size=5)
        self.join_conv_2 = ConvolutionalBlock2d(final_channels, final_channels // 2, kernel_size=5, groups=4)
        self.final_conv = nn.Conv2d(final_channels // 2, output_channels, kernel_size=5, padding=2)

        # Symmetrization helper
        self.symm = Symmetrize()

    def forward(self, x, symmetrize=True):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape: [B, input_channels, 2^(tower_height+1) * (n_bins + 2 * bins_pad)]
        symmetrize : bool
            If True, symmetrize the final output.

        Returns
        -------
        torch.Tensor
            Shape: [B, output_channels, n_bins, n_bins]
        """

        # 1) First 1D conv + pool
        x = self.first_conv_1d(x)  # [B, channels_1d, ...]
        x = self.maxpool1d(x)  # [B, channels_1d, half of previous]

        # 2) A few conv blocks, each with an extra pool
        for block in self.conv_blocks_1d[:-1]:
            x = block(x)  # [B, channels_1d, ...]
            x = self.maxpool1d(x)  # [B, channels_1d, half of previous]

        # 3) Transformer tower at 2 * (n_bins + 2 * bins_pad)
        x = self.mha_tower(x)  # [B, channels_1d, 2 * (n_bins + 2 * bins_pad)]

        # 4) Final conv block + pool => dimension is now (n_bins + 2 * bins_pad)
        x = self.conv_blocks_1d[-1](x)  # [B, channels_1d, 2 * (n_bins + 2 * bins_pad)]
        x = self.maxpool1d(x)  # [B, channels_1d, n_bins + 2 * bins_pad]

        # 5) Direct 2D branch
        x_direct = self.conv_direct_1d(x)  # [B, 2*direct_2d_input_channels - 8, n_bins + 2 * bins_pad]
        # Crop out bins_pad on each side, leaving [B, 2*direct_2d_input_channels - 8, n_bins]
        x_direct = x_direct[:, :, self.bins_pad : -self.bins_pad]
        # Convert to 2D using dist_mat_full
        x_direct = self.features_to_2d_direct(x_direct, self.dist_mat_full)  # [B, direct_2d_channels, n_bins, n_bins]

        # 6) Tower 2D branch
        x_tower = self.conv_tower_1d(x)  # [B, 2*tower_2d_input_channels - 8, n_bins + 2 * bins_pad]
        x_tower = x_tower[:, :, self.bins_pad : -self.bins_pad]  # [B, 2*tower_2d_input_channels - 8, n_bins]
        x_tower = self.maxpool1d_tower(x_tower)  # [B, tower_2d_input_channels, n_bins//2]
        x_tower = self.features_to_2d_tower(x_tower, self.dist_mat_half)  # [B, tower_2d_channels, n_bins//2, n_bins//2]

        # 7) Residual tower in 2D
        x_tower = self.residual_dilated_tower(x_tower)  # [B, tower_2d_channels, n_bins//2, n_bins//2]
        x_tower = self.batchnorm_tower(x_tower)  # [B, tower_2d_channels, n_bins//2, n_bins//2]
        x_tower = F.gelu(x_tower)
        x_tower = self.deconv(x_tower)  # [B, tower_2d_channels, n_bins, n_bins]
        x_tower = F.gelu(self.gn_deconv(x_tower))

        # 8) Join the two 2D branches
        x_2d = torch.cat((x_direct, x_tower), dim=1)  # [B, direct_2d_channels + tower_2d_channels, n_bins, n_bins]

        # 9) A couple of 2D conv blocks
        x_2d = self.join_conv(x_2d)  # [B, final_channels, n_bins, n_bins]
        x_2d = self.join_conv_2(x_2d)  # [B, final_channels//2, n_bins, n_bins]

        # 10) Final conv, optional symmetrization, then softplus
        x_2d = self.final_conv(x_2d)  # [B, output_channels, n_bins, n_bins]
        if symmetrize:
            x_2d = self.symm(x_2d)
        x_2d = F.softplus(x_2d)

        return x_2d
