import datetime as dt
import io
import json
import queue
import threading
from contextlib import nullcontext
from typing import Optional

import h5py
import hdf5plugin
import numpy as np
import polars as pl
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
from manta_hic.nn.microzoi import MicroBorzoi
from manta_hic.ops.hic_ops import (
    coarsegrained_hic_corrs,
    create_expected_matrix,
    hic_hierarchical_loss,
)
from manta_hic.ops.seq_ops import make_seq_1hot, open_fasta_chromsizes
from manta_hic.ops.tensor_ops import list_to_tensor_batch
from manta_hic.training_meta import assign_fold_type

MICROZOI_RECEPTIVE_FIELD = 2**19 + 2**18
BIN_BP = 256
CACHE_OVERHANG_BP = 2**22


def fetch_tile_microzoi_activations(
    model,
    fasta_open,
    chrom,
    start_bp,
    end_bp,
    mutate: Optional[list[tuple[str, int, str] | tuple[str, int, int]]] = None,
    reverse=False,
    start_offset_bins=0,
    shift_bp=0,
    crop_mha_bins=512,
    batch_size=4,
    require_grad=False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Fetch MicroZoi model activations across a genomic region, optionally reversed.

    Parameters
    ----------
    model : nn.Module
        The MicroZoi model (or compatible) that accepts a one-hot encoded sequence
        of length `MICROZOI_RECEPTIVE_FIELD` and returns activations.
    fasta_open : PySAM.FastaFile or compatible
        Opened FASTA file for the reference genome.
    chrom : str
        Chromosome name or identifier for the region of interest.
    start_bp : int
        Start coordinate (inclusive) in base pairs of the region.
    end_bp : int
        End coordinate (exclusive) in base pairs of the region.
    mutate : list of tuples, optional
        List of tuple ("replace", position, sequence) or ("invert"/"scramble", position, position2). Default is None.
    reverse : bool, optional
        If True, fetch and process the region in reverse orientation.
        Defaults to True.
    start_offset_bins : int, optional
        Number of bins to discard from the beginning of the first tile
        in forward orientation, or from the end in reverse orientation.
        It serves to provide alternative alignment of tiles to diversify inputs for training.
        Defaults to 0.
    shift_bp : int, optional
        Shift in base pairs to apply when fetching the sequence. Does not affect
        alignment of the output activations, only the underlying input to the model.
        Serves to further diversify inputs.
        Defaults to 0.
    crop_mha_bins : int, optional
        Number of bins to crop from each side within the model, as those are unreliable.
        Defaults to 512.
    batch_size : int, optional
        Number of tiles to process in a single batch when calling `model`.
        Defaults to 4.
    require_grad : bool, optional
        If True, returns the activations and the input sequence as a tensor with
        `requires_grad=True`. Defaults to False.


    Returns
    -------
    torch.Tensor
        A tensor of shape `[channels, num_bins_total]`, where
        `num_bins_total = (end_bp - start_bp) // BIN_BP`. The activations
        are concatenated across all tiles, with offsets removed so that
        the final coverage spans exactly the requested region.

    torch.Tensor, torch.Tensor, start_bp, end_bp
        If require_grad is set to true, returns activations and input sequence together
        with the start and end of the sequence.

    Raises
    ------
    AssertionError
        If `(end_bp - start_bp)` is not divisible by `BIN_BP`, or if
        `start_offset_bins < 0`, or if `crop_mha_bins` is large enough
        to make the tile size zero or negative.

    Notes
    -----
    - Each tile is fetched with length `MICROZOI_RECEPTIVE_FIELD` bp, but
      only the central `(MICROZOI_RECEPTIVE_FIELD - 2 * crop_mha_bins * BIN_BP)`
      portion is used to form the output. Tiles are stepped by the usable
      portion to maintain correct coverage and avoid off-by-one errors.
    - When `reverse=True`, the sequence is fetched in reverse‐complement
      order, and the final activations are returned in reverse order.
    """

    assert (end_bp - start_bp) % BIN_BP == 0, "End_bp - start_bp is not divisible by bin_bp"
    assert start_offset_bins >= 0, "start_offset_bins should be a positive number"

    num_bins_total = (end_bp - start_bp) // BIN_BP
    tile_size_bins = MICROZOI_RECEPTIVE_FIELD // BIN_BP - 2 * crop_mha_bins
    full_tile_bp = MICROZOI_RECEPTIVE_FIELD  # 786432
    usable_tile_bp = tile_step_bp = tile_size_bins * BIN_BP

    assert usable_tile_bp > 0, "Crop_mha_bins is too large"

    # determining the start/end of the tiles we need to fetch
    tile_activations_start = start_bp - start_offset_bins * BIN_BP
    num_tiles = (num_bins_total + start_offset_bins + tile_size_bins - 1) // tile_size_bins
    tile_activations_end = tile_activations_start + num_tiles * tile_size_bins * BIN_BP
    end_offset_bins = num_tiles * tile_size_bins - num_bins_total - start_offset_bins

    # sequence to fetch
    seq_start = tile_activations_start - crop_mha_bins * BIN_BP + shift_bp
    seq_end = tile_activations_end + crop_mha_bins * BIN_BP + shift_bp

    # fetch the sequence and calculate sequence tiles - if reverse, tiles are naturally in reverse order
    # if reverse, sequence is reversed - it also handles negatives and overhangs
    seq = make_seq_1hot(fasta_open, chrom, seq_start, seq_end, reverse, mutate=mutate)
    if require_grad:
        seq = torch.from_numpy(seq)
        seq.requires_grad = True
    tiles = [seq[i : i + full_tile_bp] for i in range(0, len(seq) - (full_tile_bp - tile_step_bp), tile_step_bp)]
    assert len(tiles) == num_tiles
    assert all(len(tile) == full_tile_bp for tile in tiles)

    # convert tiles to batches up to batch_size long, and then to lists and then list_to_tensor_batch
    batches = [tiles[i : i + batch_size] for i in range(0, len(tiles), batch_size)]
    # avoid last batch of size 1 - move one element from previous batch to last - this is speedup
    if len(batches[-1]) == 1 and len(batches) > 1 and batch_size > 2:
        batches[-1] = [batches[-2].pop()] + batches[-1]
    device = next(model.parameters()).device
    batches = [list_to_tensor_batch(batch, device) for batch in batches]

    # fetch activations (genome argument actually irrelevant for MHA) and cat them
    activations = []
    for batch in batches:

        def fun(x):
            return model(x.permute(0, 2, 1), genome="hg38", offset=0, crop_mha=crop_mha_bins)  # [B, C, N]

        if require_grad:
            batch_activations = checkpoint(fun, batch)
        else:
            batch_activations = fun(batch)
        # add 8 channels of x=-1...1 linear function and 3 powers of it to the activations, [B, C+8, N]
        linear = torch.linspace(-1, 1, batch_activations.shape[2], device=device).unsqueeze(0).unsqueeze(0)
        linear = linear.repeat(batch_activations.shape[0], 1, 1)
        linear = torch.cat([linear.pow(i) for i in range(8)], dim=1)
        batch_activations = torch.cat([batch_activations, linear], dim=1)
        batch_activations = batch_activations.permute(1, 0, 2).reshape(batch_activations.shape[1], -1)  # [C, B * N]
        activations.append(batch_activations)
    activations = torch.cat(activations, dim=1)  # [C, num_bins_total]

    # If reverse, we need to crop the correct amount of bins from the start,end
    M = activations.shape[1]
    if reverse:
        activations = activations[:, end_offset_bins : M - start_offset_bins]
    else:
        activations = activations[:, start_offset_bins : M - end_offset_bins]

    assert activations.shape[1] == num_bins_total, f"Activations shape is {activations.shape}, not {num_bins_total}"
    if require_grad:
        return activations, seq, seq_start, seq_end
    return activations


class MicrozoiStochasticActivationFetcher(object):
    """
    A class that fetches MicroZoi model activations across a genomic region, optionally reversed.

    For each fetching, the following things are randomized:
    * Stochastic shift in basepairs is applied up to `max_shift_bp` in both directions.
    * The region is cropped to a random size between `crop_mha_range[0]` and `crop_mha_range[1]` bins.
    * The offset in bins is randomized up to the size of the tile
    """

    def __init__(self, model, fasta_open, max_shift_bp=128, crop_mha_range=(512, 1024), batch_size=4, n_runs=1):
        self.model = model
        self.fasta_open = fasta_open
        self.max_shift_bp = max_shift_bp
        self.crop_mha_range = crop_mha_range
        self.batch_size = batch_size
        self.n_runs = n_runs
        self.model.eval()

    def fetch(self, chrom, start_bp, end_bp, reverse=False, mutate=None):
        """
        Fetch MicroZoi model activations across a genomic region, with all randomizations applied.
        """
        results = []
        for _ in range(self.n_runs):
            shift_bp = np.random.randint(-self.max_shift_bp, self.max_shift_bp + 1)
            crop_mha_bins = np.random.randint(*self.crop_mha_range)
            offset_bins = np.random.randint(0, MICROZOI_RECEPTIVE_FIELD // BIN_BP - 2 * crop_mha_bins)
            res = fetch_tile_microzoi_activations(
                self.model,
                self.fasta_open,
                chrom,
                start_bp,
                end_bp,
                reverse=reverse,
                start_offset_bins=offset_bins,
                shift_bp=shift_bp,
                crop_mha_bins=crop_mha_bins,
                batch_size=self.batch_size,
                mutate=mutate,
            )
            results.append(res)
        if self.n_runs > 1:
            # average the results
            res = torch.stack(results).mean(dim=0)
        return res


def populate_microzoi_cache(
    cache_path,
    modfile,
    fasta,
    chroms=("#", "chrX"),
    params_file=None,
    N_runs=16,
    crop_mha_range=(640, 1024),
    max_shift_bp=128,
    batch_size=4,
    n_channels=1024 + 8,
    device="cuda:0",
):
    """
    Populate a single HDF5 file with MicroZoi activations for N_runs of random parameters. Instead of storing
    per-block, we create one dataset per chromosome & orientation, of shape [n_channels, total_bins], and fill it in
    chunks. Each run has its own group.

    For each (run, chrom, orientation), we:
      - Round the chromsize down to a multiple of BIN_BP
      - We'll store from [-CACHE_OVERHANG_BP, chrom_len_rounded + CACHE_OVERHANG_BP)
      - That region, in bins, is total_bins
      - We create a dataset [n_channels, total_bins], then fill it in increments of up to e.g. 2**23 basepairs if needed
        (but we do all chunking ourselves).
      - We do no partial writing at the end; any leftover is just appended.
        The final shape is always exact.
      - We check that the returned activation shape for every piece is [n_channels, chunk_bins]
      - If the activation shape's channels differ from n_channels, we raise an error.

    We also store the entire model with torch.save() in a single "model_blob" dataset.
    The random parameters for each run are stored in run-group attrs.

    Parameters
    ----------
    cache_path : str
        Path to the HDF5 cache file to create.
    modfile : str
        Path to the model file (e.g. "model.pt").
    fasta: str or pysam.FastaFile
        Path to the FASTA file or an open handle.
    chroms: list of str
        List of chromosome names to populate the cache for.
    params_file : str
        Path to the JSON file with model parameters. Defaults to None (default parameters).
    N_runs : int
        Number of runs (distinct random shifts, offsets, crop sizes) to store.
    crop_mha_range : tuple of int
        (min_bins, max_bins). We create a linspace of length N_runs from this range for crop_mha_bins.
    max_shift_bp : int
        Maximum shift in basepairs (±).
    batch_size : int
        Batch size for fetch_tile_microzoi_activations.
    n_channels : int
        Expected number of channels returned by the model. Default is 1024+8=1032.
    device : str
        Torch device to load model and do computations.

    Notes
    -----
    Crop_mha range starts at 640 bins. The reason for that is that 640 bins allows for creating a mutation that
    affects only one tile. Since 768 is the quarter of the receptive field, we would crop two quarters from each side,
    and overhangs of neighboring tiles would "meet" in the center of the current tile. A smaller crop_mha of 640
    leaves 256 bins in the center of the tile, which is where a mutation should be placed to affect only one tile. We
    are including 640 bins in here so that the model would know about this crop_mha size and would do mutational
    screens well.

    """

    # Load the microzoi model

    fasta_open, chromsizes = open_fasta_chromsizes(fasta, chroms)

    if params_file is None:
        params = {"model": {}}  # default parameters
    else:
        params = json.load(open(params_file, "r"))

    base_model = MicroBorzoi(return_type="mha", **params["model"]).to(device)
    sd = torch.load(modfile, map_location=device, weights_only=True)
    base_model.load_state_dict(sd, strict=False)
    base_model.eval()

    # We'll store crop values via a linspace
    crop_values = np.round(np.linspace(crop_mha_range[0], crop_mha_range[1], N_runs)).astype(int)

    with h5py.File(cache_path, "w") as f:
        # Record some basic attributes
        f.attrs["CACHE_OVERHANG_BP"] = CACHE_OVERHANG_BP
        f.attrs["N_runs"] = N_runs
        f.attrs["BIN_BP"] = BIN_BP
        f.attrs["max_shift_bp"] = max_shift_bp
        f.attrs["crop_mha_range"] = crop_mha_range
        f.attrs["model_params"] = json.dumps(params)

        # Save the entire model as a single blob
        with io.BytesIO() as buffer:
            torch.save(base_model.state_dict(), buffer)
            buffer.seek(0)
            model_bytes = np.frombuffer(buffer.read(), dtype=np.uint8)
        f.create_dataset("model_blob", data=model_bytes)

        for run_idx in range(N_runs):
            run_group = f.create_group(f"run{run_idx}")

            crop_mha_bins = crop_values[run_idx]
            shift_bp = np.random.randint(-max_shift_bp, max_shift_bp + 1)
            max_offset = (MICROZOI_RECEPTIVE_FIELD // BIN_BP) - (2 * crop_mha_bins)
            offset_bins = np.random.randint(0, max_offset)

            run_group.attrs["crop_mha_bins"] = crop_mha_bins
            run_group.attrs["shift_bp"] = shift_bp
            run_group.attrs["offset_bins"] = offset_bins

            for chrom, chrom_len in chromsizes.items():
                print(f"Populating {chrom} ({chrom_len}) for run {run_idx}...")
                # Round chromosome length down to multiple of BIN_BP
                chrom_len_rounded = (chrom_len // BIN_BP) * BIN_BP
                start_of_chrom = -CACHE_OVERHANG_BP
                end_of_chrom = chrom_len_rounded + CACHE_OVERHANG_BP
                total_bp = end_of_chrom - start_of_chrom
                total_bins = total_bp // BIN_BP

                for reverse_bool in [False, True]:
                    orientation = "reverse" if reverse_bool else "forward"
                    ds_name = f"{chrom}_{orientation}"

                    # Create dataset of shape [n_channels, total_bins]
                    # We'll fill it in chunks of up to e.g. 2**23 basepairs if we like,
                    # but let's do a loop in e.g. 2**23 sized increments if needed.
                    dset = run_group.create_dataset(
                        ds_name,
                        shape=(n_channels, total_bins),
                        dtype=np.float16,
                        compression=hdf5plugin.Zstd(clevel=9),
                        chunks=(n_channels, min(1024, total_bins)),
                    )

                    block_bp = 2**23
                    num_blocks = (total_bp + block_bp - 1) // block_bp

                    # We'll accumulate an offset in bins for writing
                    write_bin_offset = 0

                    def write_item(dset, idx, idx2, data):
                        dset[:, idx:idx2] = data

                    last_thread = None

                    for block_idx in range(num_blocks):
                        block_start_bp = start_of_chrom + block_idx * block_bp
                        block_end_bp = min(start_of_chrom + (block_idx + 1) * block_bp, end_of_chrom)
                        # fetch
                        with torch.no_grad(), torch.autocast(device):
                            activ = fetch_tile_microzoi_activations(
                                model=base_model,
                                fasta_open=fasta_open,
                                chrom=chrom,
                                start_bp=block_start_bp,
                                end_bp=block_end_bp,
                                reverse=reverse_bool,
                                start_offset_bins=offset_bins,
                                shift_bp=shift_bp,
                                crop_mha_bins=crop_mha_bins,
                                batch_size=batch_size,
                            )
                        torch.clip_(activ, -25000, 25000)  # clip to float16 range minus a bit

                        if (~torch.isfinite(activ)).sum() > 0:
                            activ[~torch.isfinite(activ)] = 0
                            assert (~torch.isfinite(activ)).sum() == 0
                            print("Non-finite activations detected!!!")

                        arr = activ.cpu().numpy().astype(np.float16)
                        assert (~np.isfinite(arr)).sum() == 0

                        if reverse_bool:  # reverse the array - we are saving in forward orientation
                            arr = arr[:, ::-1]

                        # Check channels
                        if arr.shape[0] != n_channels:
                            raise ValueError(f"Expected {n_channels} channels, got {arr.shape[0]}")

                        block_bins = arr.shape[1]
                        end_bin_offset = write_bin_offset + block_bins

                        if last_thread is not None:
                            last_thread.join()

                        # Write to the HDF5 dataset
                        # dset[:, write_bin_offset:end_bin_offset] = arr
                        last_thread = threading.Thread(
                            target=write_item, args=(dset, write_bin_offset, end_bin_offset, arr)
                        )
                        last_thread.start()
                        write_bin_offset = end_bin_offset

                    if write_bin_offset != total_bins:
                        raise RuntimeError("Didn't fill the entire dataset - mismatch between chunking and total_bins.")
                    last_thread.join()


def create_microzoi_model_from_cache(cache_path, device="cuda", return_type="mha", **kwargs):
    """
    Create a MicroZoi (MicroBorzoi) model from the 'model_blob' recorded in the HDF5 cache file.
    This re-creates the exact Torch model used to produce the cached activations.

    Parameters
    ----------
    cache_path : str
        Path to the HDF5 file created by populate_microzoi_cache.
    device : str
        Torch device.
    kwargs : dict
        Additional keyword arguments to pass to the MicroBorzoi constructor.

    Returns
    -------
    model : MicroBorzoi
        The MicroZoi model, loaded from the blob, set to eval mode.
    """
    with h5py.File(cache_path, "r") as f:
        model_bytes = f["model_blob"][:].tobytes()
        params = json.loads(f.attrs["model_params"])

    # combine model parameters
    mod_args = params["model"]
    mod_args.update(kwargs)
    mod_args.update({"return_type": return_type})

    model = MicroBorzoi(**mod_args).to(device)
    model.load_state_dict(torch.load(io.BytesIO(model_bytes), map_location=device, weights_only=True))
    model.eval()
    return model


class CachedStochasticActivationFetcher(object):
    """
    Drop-in replacement for MicroziStochasticActivationFetcher,
    but uses a precomputed HDF5 cache of Microzoi activations.

    Implementation:
      - We open the cache file in `fetch()`, read the top-level or run-level metadata,
        pick a random run, then just do a simple slice from the dataset for (chrom, orientation).
      - The user must ensure that [start_bp, end_bp) is fully contained within
        [-CACHE_OVERHANG_BP, chrom_len_rounded + CACHE_OVERHANG_BP), and that the
        region is aligned to BIN_BP. If not, we raise an error.
    """

    def __init__(self, cache_path, fasta_open=None, batch_size=4):
        """
        Parameters
        ----------
        cache_path : str
            Path to the HDF5 file with cached activations.
        device : str
            Torch device for returning the final tensor.
        """
        self.cache_path = cache_path
        self.fasta_open = fasta_open
        self.batch_size = batch_size
        with h5py.File(self.cache_path, "r") as f:
            self.N_runs = f.attrs["N_runs"]
            self.cache_overhang_bp = f.attrs["CACHE_OVERHANG_BP"]
            self.bin_bp = f.attrs["BIN_BP"]

    def fetch(self, chrom, start_bp, end_bp, reverse=False, return_full=False, run_idx=None):
        """
        Fetch cached activations for region [start_bp, end_bp).
        start_bp, end_bp must be multiples of bin_bp, and must lie entirely
        within the stored dataset region.

        If return_full, returns the slice, together with shift_bins, crop_mha, and shift_bp.
        """
        if (start_bp % self.bin_bp) != 0 or (end_bp % self.bin_bp) != 0:
            raise ValueError("start_bp and end_bp must be multiples of BIN_BP")
        if run_idx is None:
            run_idx = np.random.randint(self.N_runs)

        orientation = "reverse" if reverse else "forward"
        with h5py.File(self.cache_path, "r") as f:
            run_group = f[f"run{run_idx}"]
            ds_name = f"{chrom}_{orientation}"
            if ds_name not in run_group:
                raise KeyError(f"Dataset not found: run{run_idx}/{ds_name}")
            dset = run_group[ds_name]

            # The dataset covers [-cache_overhang_bp, chrom_len_rounded + cache_overhang_bp)
            # in basepairs. So the total length in bins is dset.shape[1].
            # The start of the region in basepairs is "start_of_chrom" = -cache_overhang_bp.
            # Let's figure out the slice offset in bins:
            total_bins = dset.shape[1]
            region_start_bp = -self.cache_overhang_bp
            region_end_bp = region_start_bp + total_bins * self.bin_bp

            if start_bp < region_start_bp or end_bp > region_end_bp:
                raise ValueError(
                    f"Requested region [{start_bp}, {end_bp}) is outside stored range "
                    f"[{region_start_bp}, {region_end_bp})."
                )

            offset_start_bin = (start_bp - region_start_bp) // self.bin_bp
            offset_end_bin = (end_bp - region_start_bp) // self.bin_bp
            if offset_start_bin < 0 or offset_end_bin > total_bins:
                raise ValueError(f"Slice in bins [{offset_start_bin}, {offset_end_bin}) is out of [0, {total_bins}).")

            arr = dset[:, offset_start_bin:offset_end_bin]
            if reverse:  # reverse the array - we are saving in forward orientation
                arr = arr[:, ::-1].copy()

        if return_full:
            with h5py.File(self.cache_path, "r") as f:
                run_group = f[f"run{run_idx}"]
                crop_mha_bins = run_group.attrs["crop_mha_bins"]
                shift_bp = run_group.attrs["shift_bp"]
                offset_bins = run_group.attrs["offset_bins"]
            return arr, offset_bins, crop_mha_bins, shift_bp
        return arr

    def _fetch_microzoi_model(self, device):
        """
        Fetch the microzoi model. Only create model the first time this is called
        """
        if not hasattr(self, "_model"):
            with h5py.File(self.cache_path, "r") as f:
                model_bytes = f["model_blob"][:].tobytes()
                params = json.loads(f.attrs["model_params"])
            self._model = MicroBorzoi(return_type="mha", **params["model"]).to(device)
            self._model.load_state_dict(torch.load(io.BytesIO(model_bytes), map_location=device, weights_only=True))
            self._model.eval()
        return self._model

    def fetch_matched_pairs(self, chrom, start_bp, end_bp, device, reverse=False, mutates=[]):
        """
        Fetch a pair of activations for the same region, but with different mutations applied.

        Fetches a whole region from cache, and then "patches" it with microzoi activations calculated for the
        regions that needed to be recalculated due to overlap with mutations.

        Note thta we should use the same random run for both of the activations, and we need to patch
        both the original and the mutated activation.

        returns 2*len(mutates) activations, each of shape [channels, bins]

        """

        assert all([len(i) == 1 for i in mutates]), "Only single mutations are supported"
        assert all([i[0][1] == mutates[0][0][1] for i in mutates]), "All mutations should be on the same position"

        N_pairs = len(mutates)

        # position of mutation in the sequence to fetch
        mut_pos = mutates[0][0][1]
        mut_pos_bins = (mut_pos - start_bp) // BIN_BP

        # calculating microzoi window for the tile to patch
        window_start = mut_pos - MICROZOI_RECEPTIVE_FIELD // 2
        window_end = window_start + MICROZOI_RECEPTIVE_FIELD

        all_acts = []
        all_seqs = []
        for pair in range(N_pairs):
            acts, offset_bins, crop_mha_bins, shift_bp = self.fetch(
                chrom, start_bp, end_bp, reverse=reverse, return_full=True
            )
            all_acts.append(acts)

            # adjust window to use shift_bp
            window_start += shift_bp
            window_end += shift_bp

            # fetch sequences for the window
            seq = make_seq_1hot(self.fasta_open, chrom, window_start, window_end, reverse)
            seq_mutate = make_seq_1hot(self.fasta_open, chrom, window_start, window_end, reverse, mutate=mutates[pair])
            all_seqs.append(seq)
            all_seqs.append(seq_mutate)

        # fetch the model
        model = self._fetch_microzoi_model(device)

        # split seqs into batches
        batches = [all_seqs[i : i + self.batch_size] for i in range(0, len(all_seqs), self.batch_size)]
        batches = [list_to_tensor_batch(batch, device) for batch in batches]

        # fetch activations to patch
        all_patches = []
        for batch in batches:
            with torch.no_grad(), torch.autocast(device):
                patch = model(batch.permute(0, 2, 1), genome="hg38", offset=0, crop_mha=768)
                # add linear function and its powers to the activations
                linear = torch.linspace(-1, 1, patch.shape[2], device=device).unsqueeze(0).unsqueeze(0)
                linear = linear.repeat(patch.shape[0], 1, 1)
                linear = torch.cat([linear.pow(i) for i in range(8)], dim=1)
                patch = torch.cat([patch, linear], dim=1)
            for i in patch.detach().cpu().numpy():
                all_patches.append(i)

        # patch the activations
        tile_size_bins = MICROZOI_RECEPTIVE_FIELD // BIN_BP - 2 * 768
        insert_start_bins = mut_pos_bins - tile_size_bins // 2
        insert_end_bins = insert_start_bins + tile_size_bins

        final_acts = []  # activations to retur
        for pair in range(N_pairs):
            acts = all_acts[pair].copy()
            patch_wt = all_patches[2 * pair]
            patch_mut = all_patches[2 * pair + 1]

            # insert the patch
            acts[:, insert_start_bins:insert_end_bins] = patch_wt
            acts_mut = acts.copy()
            acts_mut[:, insert_start_bins:insert_end_bins] = patch_mut

            final_acts.append(acts)
            final_acts.append(acts_mut)

        return final_acts


class HybridCachedStochasticFetcher:
    """
    A combined fetcher that, on each fetch, does one of the following:
      1) With probability prob_stochastic, calls a fallback (stochastic) fetcher,
         which we automatically build from the HDF5 cache attributes if not provided.
      2) With probability prob_mean, picks a distinct subset of runs (2..max_mean_runs, no replacement) from the cache,
         and returns their elementwise mean.
      3) Otherwise, returns a single-run cached activation.

    The fallback fetcher is auto-initialized from:
        - The microzoi model saved in the cache (via model_blob)
        - The max_shift_bp from cache attrs
        - The crop_mha_range from cache attrs
        - The user-provided fasta_open and batch_size
        etc.

    Parameters
    ----------
    cache_path : str
        Path to the HDF5 file with cached Microzoi activations (and attributes).
    fasta_open : pysam.FastaFile or similar
        FASTA handle for fallback fetcher (model inference).
    prob_mean : float
        Probability of returning the mean over multiple runs from the cache.
    max_mean_runs : int
        Maximum number of runs to average over. We pick a random n in [2, max_mean_runs],
        distinct runs for that average.
    """

    def __init__(
        self,
        cache_path,
        fasta_open,
        prob_mean=0.1,
        min_mean_runs=2,
        max_mean_runs=4,
    ):
        self.cache_path = cache_path
        self.fasta_open = fasta_open
        self.prob_mean = prob_mean
        self.min_mean_runs = min_mean_runs
        self.max_mean_runs = max_mean_runs

        # Open cache, read the relevant attributes, and build the fallback fetcher automatically
        with h5py.File(self.cache_path, "r") as f:
            self.N_runs = f.attrs["N_runs"]

    def fetch(self, chrom, start_bp, end_bp, reverse=False):
        """
        Fetch activations for [start_bp, end_bp). With probability prob_stochastic, we use the fallback fetcher;
        with probability prob_mean, we average multiple runs from the cache; else, pick a single run from the cache.

        Parameters
        ----------
        chrom : str
            Chromosome name.
        start_bp : int
            Start coordinate (multiple of bin_bp).
        end_bp : int
            End coordinate (multiple of bin_bp).
        reverse : bool, optional
            If True, return reversed orientation.

        Returns
        -------
        torch.Tensor
            Activation of shape [channels, bins].
        """
        r = np.random.rand()

        if r < self.prob_mean:
            # Multi-run average
            max_distinct = min(self.max_mean_runs, self.N_runs)
            min_distinct = self.min_mean_runs
            if max_distinct < 2:
                # fallback to single-run if not enough runs in the cache
                return self._fetch_cached([0], chrom, start_bp, end_bp, reverse)
            n_runs_to_avg = np.random.randint(min_distinct, max_distinct + 1)
            run_indices = np.random.choice(self.N_runs, size=n_runs_to_avg, replace=False)
            return self._fetch_cached(run_indices, chrom, start_bp, end_bp, reverse)

        else:
            # Single-run from the cache
            run_idx = np.random.randint(self.N_runs)
            return self._fetch_cached([run_idx], chrom, start_bp, end_bp, reverse)

    def _fetch_cached(self, run_indices, chrom, start_bp, end_bp, reverse):
        """
        Fetch (and possibly average) cached activations from the given run indices.

        Parameters
        ----------
        run_indices : list of int
            Run indices to average over. If length==1, returns that run's data.
        chrom, start_bp, end_bp, reverse : as above

        Returns
        -------
        torch.Tensor
            Activation of shape [channels, bins].
        """
        orientation = "reverse" if reverse else "forward"

        arr_accum = None
        with h5py.File(self.cache_path, "r") as f:
            region_start_bp = -CACHE_OVERHANG_BP
            for i, run_idx in enumerate(run_indices):
                run_group = f[f"run{run_idx}"]
                ds_name = f"{chrom}_{orientation}"
                if ds_name not in run_group:
                    raise KeyError(f"Dataset not found: run{run_idx}/{ds_name}")
                dset = run_group[ds_name]

                total_bins = dset.shape[1]
                region_end_bp = region_start_bp + total_bins * BIN_BP
                if start_bp < region_start_bp or end_bp > region_end_bp:
                    raise ValueError(
                        f"Requested region [{start_bp}, {end_bp}) is outside stored range "
                        f"[{region_start_bp}, {region_end_bp})."
                    )
                offset_start_bin = (start_bp - region_start_bp) // BIN_BP
                offset_end_bin = (end_bp - region_start_bp) // BIN_BP

                arr = dset[:, offset_start_bin:offset_end_bin]
                # Because you stored each orientation in forward orientation,
                # but labeled it "chrom_reverse" or "chrom_forward",
                # we keep the final flip if reverse to replicate the original fetcher logic.
                if reverse:
                    arr = arr[:, ::-1]

                arr = arr.astype(np.float32)
                if arr_accum is None:
                    arr_accum = arr
                else:
                    arr_accum += arr
        if len(run_indices) > 1:
            arr_accum /= float(len(run_indices))
        return arr_accum


class SequenceFetcher(object):
    """
    A class that fetches sequences from a FASTA file, rather than intermediate activations.

    It can be used for "one shot" models that do not use transfer learning, like Akita.
    Accepts optional "max_shift_bp" argument to apply a random shift in basepairs.
    """

    def __init__(self, fasta_open, max_shift_bp=128):
        self.fasta_open = fasta_open
        self.max_shift_bp = max_shift_bp

    def fetch(self, chrom, start_bp, end_bp, reverse=True):
        """
        Fetch a sequence from a FASTA file, optionally reversed.
        """

        shift_bp = np.random.randint(-self.max_shift_bp, self.max_shift_bp + 1)
        return make_seq_1hot(self.fasta_open, chrom, start_bp + shift_bp, end_bp + shift_bp, reverse).T


class DummyFetcher(object):
    """A dummy fetcher that returns None for all fetches."""

    def fetch(self, *args, **kwargs):
        return None


class HiCDataset:
    """
    A dataset class for handling Hi-C data.
    Parameters
    ----------
    filename : str
        Path to the HDF5 file containing Hi-C data.
    fetcher : CachedStochasticActivationFetcher or other fetcher defined above
        A fetcher object that can retrieve activations for a given region.
    n_bins : int
        Number of bins to use for Hi-C data slices.
    bins_pad : int
        Padding to add around Hi-C data slices.
    genome : str
        Genome name for the dataset - used to assign fold types.
    test_fold : str, optional
        Name of the fold to use for testing data (default is "fold3").
    val_fold : str, optional
        Name of the fold to use for validation data (default is "fold4").
    fold_types_use : Iterable[str] | None, optional
        Which fold-types to use for the dataset (default is ["train"]). None uses everything.
    random : bool, optional
        Whether to use random starting positions for Hi-C data slices (default is True).

    Methods
    -------
    __len__()
        Returns the number of samples in the dataset.
    __getitem__(idx, stochastic_reverse=True)
        Retrieves a sample from the dataset at the specified index.
    get_single(idx)
        Retrieves a sample from the dataset at the specified index without random shifts or reverse.

    Notes
    -----

    The Hi-C dataset class holds Hi-C data in a slightly bigger squares than the requested nbins.
    The squares are tiled in a way that any map n_bins by n_bins can be extracted from the dataset.
    (specifically, we save 1.25*n_bins by 1.25*n_bins squares, and save them every 0.25*n_bins).

    The datasets holds 3 arrays that define a (chrom, start, end) dataframe.
    It also saves:

    *  the Hi-C data in a 3D array (index, 1.25*n_bins, 1.25*n_bins)
    *  the weights in a 2D array (index, 1.25*n_bins)
    *  the expected values in a 2D array (index, 1.25*n_bins)

    The expected value was calculated from the entire crhomosomal arm.
    """

    def __init__(
        self,
        filename,
        fetcher,
        n_bins,
        bins_pad,
        genome,
        test_fold="fold3",
        val_fold="fold4",
        fold_types_use=("train",),
        stochastic_offset=True,
        stochastic_reverse=True,
    ):
        self.filename = filename
        self.n_bins = n_bins
        self.bins_pad = bins_pad
        self.stochastic_offset = stochastic_offset
        self.stochastic_reverse = stochastic_reverse
        self.fetcher = fetcher

        allowed_fold_types = ["train", "val", "test", "discard"]
        if fold_types_use is not None:
            assert all(fold_type in allowed_fold_types for fold_type in fold_types_use), "Invalid fold type"

        with h5py.File(filename, "r") as f:
            df = pl.DataFrame({"chrom": f["chrom"][:], "start": f["start"][:], "end": f["end"][:]})
            self.M = f["hic"].shape[-1]
            self.n_channels = f["hic"].shape[1]
            self.hic_res = (f["end"][0] - f["start"][0]) // self.M  # infer Hi-C resolution from the matrix size

        df = df.with_columns(pl.col("chrom").cast(str), pl.int_range(pl.len()).alias("index"))
        df = assign_fold_type(df, test_fold=test_fold, val_fold=val_fold, genome=genome)

        if fold_types_use:
            df = df.filter(pl.col("fold_type").is_in(fold_types_use))

        self.df = df

    def get_slice_by_index(self, idx, offset_bins=0, reverse=False):
        row = self.df.row(idx, named=True)
        orig_idx = row["index"]
        map_start_bp = row["start"] + offset_bins * self.hic_res
        map_end_bp = map_start_bp + self.n_bins * self.hic_res
        fetch_start_bp = map_start_bp - self.bins_pad * self.hic_res
        fetch_end_bp = map_end_bp + self.bins_pad * self.hic_res

        with h5py.File(self.filename, "r") as f:
            offset_slice = slice(offset_bins, offset_bins + self.n_bins)
            hic_slice = f["hic"][orig_idx, :, offset_slice, offset_slice]
            weight_slice = f["weights"][orig_idx, :, offset_slice]
            exp = f["exp"][orig_idx]
        if reverse:
            hic_slice = hic_slice[:, ::-1, ::-1].copy()
            weight_slice = weight_slice[:, ::-1].copy()

        activations = self.fetcher.fetch(row["chrom"], fetch_start_bp, fetch_end_bp, reverse=reverse)

        result = {
            "acts": activations,
            "hic_slice": hic_slice,
            "weight_slice": weight_slice,
            "exp": exp,
            "chrom": row["chrom"],
            "fetch_start_bp": fetch_start_bp,
            "fetch_end_bp": fetch_end_bp,
            "map_start_bp": map_start_bp,
            "map_end_bp": map_end_bp,
            "reverse": reverse,
        }

        return result

    def get_slice_by_coords(self, chrom, start_bp, reverse=False):
        """
        Finds which record does the given coordinates belong to and returns the slice.
        If record is not found (usually if it crosses the centromere) raise an error.
        """
        end_bp = start_bp + self.n_bins * self.hic_res
        df = self.df.with_row_count(name="new_index").filter(
            (pl.col("chrom") == chrom) & (pl.col("start") <= start_bp) & (pl.col("end") >= end_bp)
        )
        if len(df) == 0:
            raise ValueError("Region not found in the dataset")
        row = df.row(0, named=True)
        if (start_bp - row["start"]) % self.hic_res != 0:
            raise ValueError("Start bp is not multiple of hic resolution")
        offset = (start_bp - row["start"]) // self.hic_res
        return self.get_slice_by_index(row["new_index"], offset_bins=offset, reverse=reverse)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        offset_bins = np.random.randint(0, self.M - self.n_bins) if self.stochastic_offset else 0
        use_reverse = self.stochastic_reverse and np.random.rand() > 0.5
        return self.get_slice_by_index(idx, offset_bins=offset_bins, reverse=use_reverse)


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
            self.q.put(tuple(batch))
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

        acts = list_to_tensor_batch([i["acts"] for i in batch], device)
        target = list_to_tensor_batch([i["hic_slice"] for i in batch], device)
        weight = list_to_tensor_batch([i["weight_slice"] for i in batch], device)
        exp = list_to_tensor_batch([i["exp"] for i in batch], device)
        target, weightmat = create_expected_matrix(target, weight, exp)

        # calculating activations

        with torch.autocast("cuda"):

            if is_train:
                acts.requires_grad = True
                optimizer.zero_grad()

            with torch.no_grad() if not is_train else nullcontext():
                output = model(acts)
                if is_train:
                    loss = hic_hierarchical_loss(output, target, weightmat)
                corr = [i for i in coarsegrained_hic_corrs(output, target, weight, exp, also_divide_by_mean=True)]
                corr = np.array([i.detach().cpu().numpy() for i in corr])

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
        super(Manta2, self).__init__()
        self.n_bins = n_bins
        self.bins_pad = bins_pad
        self.channels_1d = channels_1d
        self.checkpoint_first = checkpoint_first
        self.conv_blocks_checkpoint = conv_blocks_checkpoint

        if final_channels < 2 * output_channels:
            raise ValueError("Final channels must be at least 2 times the output channels.")

        # Precompute distance matrices for full (n_bins x n_bins) and half ((n_bins//2) x (n_bins//2))
        dist_mat_full = calculate_distance_matrix(n_bins)
        dist_mat_half = calculate_distance_matrix(n_bins // 2)
        self.register_buffer("dist_mat_full", dist_mat_full, persistent=False)
        self.register_buffer("dist_mat_half", dist_mat_half, persistent=False)

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
        # Convert to 2D using dist_mat_full
        x_direct = self.features_to_2d_direct(x_direct, self.dist_mat_full)  # [B, direct_2d_channels, n_bins, n_bins]

        # 6) Tower 2D branch
        x_tower = self.conv_tower_1d(x)  # [B, 2*tower_2d_input_channels - 8, n_bins + 2 * bins_pad]
        x_tower = x_tower[:, :, self.bins_pad : -self.bins_pad]  # [B, 2*tower_2d_input_channels - 8, n_bins]
        x_tower = self.maxpool1d_tower(x_tower)  # [B, tower_2d_input_channels, n_bins//2]
        x_tower = self.features_to_2d_tower(x_tower, self.dist_mat_half)  # [B, tower_2d_channels, n_bins//2, n_bins//2]

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
