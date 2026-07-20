"""
Manta -- predicts a 2D Hi-C contact map from precomputed MicroZoi activations.

The MicroZoi activation cache lives elsewhere: the read/fetch side (what training and inference consume) is
``nn/fetchers.py``; the offline cache builder is ``nn/fill_cache.py``.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from manta_hic.nn.layers import (
    ConvolutionalBlock1d,
    ConvolutionalBlock2d,
    FeaturesTo2D,
    FibonacciResidualTower,
    Symmetrize,
    TransformerTower,
)


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


class Manta(nn.Module):
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
        Number of 1D conv+pool blocks (in total). Rescales the input by a factor 2^(H+1).
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
    checkpoint_first : bool
        If True, checkpoint the first 1D convolutions.
    conv_blocks_checkpoint : int
        Number of 1D conv blocks to checkpoint GN part (starting from the first one).

    Notes
    -----
    We perform the transformer tower at the hic_resolution/2. This is because the Hi-C map is generally not more than
    1024x1024, and transformers can totally handle 2000-3000 bins with not much overhead. The hope is that at lower
    resolution transformers will be able to perform more "compute".

    The first convolution and maxpool is technically not the part of the "tower" because it has a fixed and special
    input dimension, and because our convolutional blocks start with GN+GELU, and we can't start with a nonlinearity
    directly following the activations from the previous network, Microzoi.

    To allow for the resolution of 512bp, we have a special case - if the tower height is 0, the maxpool is not
    applied after the first convolution and the tower height is set to 1. So we convolve from input_channels to
    channels_1d, immediately do a transformer tower, and convolve/maxpool down to 512bp resolution.

    We have checkpointing logic as follows. The first convolution is checkpointed as a whole convolution, as it's
    output is the largest activation in the network. The rest of the convolutions have an option to checkpoint only
    their groupnorm part, as it is "cheaper" than the convolution itself. It is possible to add more checkpointing
    logic in the future, specifically for "Akita-like" networks, and be checkpointing whole convolutions.
    """

    def __init__(
        self,
        *,
        n_bins=1024,
        bins_pad=128,
        input_channels=1024 + 8,
        channels_1d=512,
        tower_height=2,
        transformer_layers=8,
        transformer_dropout=0.4,
        transformer_n_heads=8,
        direct_2d_input_channels=64,
        direct_2d_channels=48,
        tower_2d_input_channels=96,
        tower_2d_channels=48,
        tower_2d_width=5,
        tower_2d_dropout=0.2,
        tower_2d_height=9,
        final_channels=32,
        output_channels=2,
        checkpoint_first=False,
        conv_blocks_checkpoint=0,
    ):
        super(Manta, self).__init__()
        self.n_bins = n_bins
        self.bins_pad = bins_pad
        self.channels_1d = channels_1d
        self.checkpoint_first = checkpoint_first
        self.conv_blocks_checkpoint = conv_blocks_checkpoint

        if final_channels < 2 * output_channels:
            raise ValueError("Final channels must be at least 2 times the output channels.")

        # Distance matrices are computed on the fly per actual map size and cached (keyed by side length), so
        # one model runs at variable window size (e.g. 512 for sweeps, 1024 for production). The normalization
        # is over the NxN matrix, so a slice of a bigger dist_mat != dist_mat of a smaller size -- must recompute
        # per size (cheap, cached). The n_bins passed here is the *max* size (sizes freqs_cis in the transformer).
        self._dist_cache: dict[int, torch.Tensor] = {}

        # 1D backbone
        self.first_conv_1d = nn.Conv1d(input_channels, channels_1d, kernel_size=3, padding=1)
        self.maxpool1d = nn.MaxPool1d(kernel_size=2, stride=2)
        self.tower_height = tower_height

        # Multiple conv blocks + pooling
        # We have to have at least one block, so if tower_height is zero
        # we instead don't do maxpool after the "initial" convolution.
        self.conv_blocks_1d = nn.ModuleList()
        for i in range(max(tower_height, 1)):
            do_checkpoint = i < self.conv_blocks_checkpoint and self.training
            cblock = ConvolutionalBlock1d(channels_1d, channels_1d, 3, groups=1, checkpoint_gn=do_checkpoint)
            self.conv_blocks_1d.append(cblock)

        # Transformer tower (assume it has RMSNorm inside or appended)
        mha_bins = (2 if self.tower_height > -1 else 1) * (n_bins + 2 * bins_pad)  # for 256bp resolution model
        self.mha_tower = TransformerTower(
            n_layers=transformer_layers,
            d_model=channels_1d,
            n_bins=mha_bins,
            n_heads=transformer_n_heads,
            drop_p=transformer_dropout,
        )

        # Direct 2D branch
        # Reserve some channels for the extra distance/upper-lower features in FeaturesTo2D
        self.conv_direct_1d = ConvolutionalBlock1d(channels_1d, 2 * direct_2d_input_channels - 8, 1)
        self.features_to_2d_direct = FeaturesTo2D(direct_2d_input_channels, direct_2d_channels, kernel_size=3)

        # Tower 2D branch
        self.conv_tower_1d = ConvolutionalBlock1d(channels_1d, 2 * tower_2d_input_channels - 8, 1)
        self.maxpool1d_tower = nn.MaxPool1d(kernel_size=2, stride=2)
        self.features_to_2d_tower = FeaturesTo2D(tower_2d_input_channels, tower_2d_channels, kernel_size=3)

        # Residual dilated tower
        self.residual_dilated_tower = FibonacciResidualTower(
            tower_2d_channels, tower_2d_height, tower_2d_width, dropout=tower_2d_dropout
        )
        self.batchnorm_tower = nn.BatchNorm2d(tower_2d_channels, momentum=0.01)

        # 2D deconv + groupnorm
        self.deconv = nn.ConvTranspose2d(tower_2d_channels, tower_2d_channels, kernel_size=2, stride=2, groups=4)
        self.gn_deconv = nn.GroupNorm(tower_2d_channels // 8, tower_2d_channels)

        # Final join
        self.join_conv = ConvolutionalBlock2d(direct_2d_channels + tower_2d_channels, final_channels, kernel_size=3)
        self.join_conv_2 = ConvolutionalBlock2d(final_channels, final_channels // 2, kernel_size=5, groups=4)
        self.final_conv = nn.Conv2d(final_channels // 2, output_channels, kernel_size=1, padding=0)

        # Symmetrization helper
        self.symm = Symmetrize()

    def _dist(self, n):
        """Distance matrix for an ``n x n`` map, cached per size and on the model's device."""
        cached = self._dist_cache.get(n)
        dev = self.first_conv_1d.weight.device
        if cached is None or cached.device != dev:
            cached = calculate_distance_matrix(n).to(dev)
            self._dist_cache[n] = cached
        return cached

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

        # 1) First 1D conv + pool [B, channels_1d, ...]
        do_checkpoint = self.checkpoint_first and self.training
        x = checkpoint(self.first_conv_1d, x, use_reentrant=True) if do_checkpoint else self.first_conv_1d(x)
        if self.tower_height > 0:  # if we want 512bp resolution, we simply don't maxpool here and keep 1 convolution
            x = self.maxpool1d(x)  # [B, channels_1d, half of previous]

        # 2) A few conv blocks, each with an extra pool
        for block in self.conv_blocks_1d[:-1]:
            x = block(x)  # [B, channels_1d, ...]
            x = self.maxpool1d(x)  # [B, channels_1d, half of previous]

        # 3) Transformer tower at 2 * (n_bins + 2 * bins_pad)
        x = self.mha_tower(x)  # [B, channels_1d, 2 * (n_bins + 2 * bins_pad)]

        # 4) Final conv block + pool => dimension is now (n_bins + 2 * bins_pad)
        x = self.conv_blocks_1d[-1](x)  # [B, channels_1d, 2 * (n_bins + 2 * bins_pad)]

        if self.tower_height > -1:  # shortcut for 256bp resolution models
            x = self.maxpool1d(x)  # [B, channels_1d, n_bins + 2 * bins_pad]

        # 5) Direct 2D branch
        x_direct = self.conv_direct_1d(x)  # [B, 2*direct_2d_input_channels - 8, n_bins + 2 * bins_pad]
        # Crop out bins_pad on each side, leaving [B, 2*direct_2d_input_channels - 8, n_bins]
        x_direct = x_direct[:, :, self.bins_pad : -self.bins_pad]
        # Convert to 2D using a distance matrix sized to the actual (possibly variable) map
        x_direct = self.features_to_2d_direct(x_direct, self._dist(x_direct.shape[-1]))  # [B, direct_2d_ch, N, N]

        # 6) Tower 2D branch
        x_tower = self.conv_tower_1d(x)  # [B, 2*tower_2d_input_channels - 8, n_bins + 2 * bins_pad]
        x_tower = x_tower[:, :, self.bins_pad : -self.bins_pad]  # [B, 2*tower_2d_input_channels - 8, n_bins]
        x_tower = self.maxpool1d_tower(x_tower)  # [B, tower_2d_input_channels, n_bins//2]
        x_tower = self.features_to_2d_tower(x_tower, self._dist(x_tower.shape[-1]))  # [B, tower_2d_ch, N/2, N/2]

        # 7) Residual tower in 2D
        x_tower = self.residual_dilated_tower(x_tower)  # [B, tower_2d_channels, n_bins//2, n_bins//2]
        # This is an oversight, this batchnorm should be gone, but we are sticking to it due to the pre-trained models.
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


def save_manta_checkpoint(
    model, path, *, channel_names=None, model_params=None, genome=None, history=None, train_meta=None
):
    """
    Save a Manta model as a **self-describing** checkpoint: ``{"state_dict": ..., "config": {...}}``.

    The config carries everything needed to reload and use the model without inferring anything from tensor
    shapes -- crucially the ``resolution`` (which the state dict can't disambiguate below 1024 bp), plus
    ``n_bins`` / ``bins_pad`` / ``tower_height`` / ``output_channels`` and, optionally, ``genome`` (the model
    is genome-specific, so this lets inference reject a mismatched cache/target), ``channel_names`` (so channel
    labels no longer depend on the target file) and non-default ``model_params``. ``MantaInference`` reads this
    format directly.

    ``history`` (optional) is the per-epoch training log -- a list of small dicts of *mean* metrics (train/val
    loss and mean correlations). Stored right in the checkpoint so the model file IS its own training record; no
    side-car folders. ``train_meta`` (optional) is a dict of run-level settings (lr, batch size, dtypes, ...).
    Both are plain JSON-able Python, negligible in size even at hundreds of epochs.
    """
    state = model.state_dict()
    config = {
        "resolution": 2 ** (int(model.tower_height) + 9),
        "n_bins": int(model.n_bins),
        "bins_pad": int(model.bins_pad),
        "tower_height": int(model.tower_height),
        "output_channels": int(state["final_conv.weight"].shape[0]),
    }
    if genome is not None:
        config["genome"] = str(genome)
    if channel_names is not None:
        config["channel_names"] = list(channel_names)
    if model_params:
        config["model_params"] = dict(model_params)
    if history is not None:
        config["history"] = list(history)
    if train_meta is not None:
        config["train_meta"] = dict(train_meta)
    torch.save({"state_dict": state, "config": config}, path)
