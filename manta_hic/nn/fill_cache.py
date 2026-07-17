"""
Offline builder for the MicroZoi activation cache (CLI: ``manta_hic io fill-cache``). Runs MicroZoi over each
chromosome in ``N_runs`` stochastic tilings/orientations and writes the activations to a giant HDF5 that the
read side (``nn/fetchers.py``) slices at train/inference time.
"""

import io
import json
import threading
from typing import Optional

import h5py
import hdf5plugin
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from manta_hic.nn.fetchers import BIN_BP, CACHE_OVERHANG_BP, MICROZOI_RECEPTIVE_FIELD
from manta_hic.nn.microzoi import Microzoi
from manta_hic.ops.seq_ops import make_seq_1hot, open_fasta_chromsizes
from manta_hic.ops.tensor_ops import list_to_tensor_batch, round_mantissa, torch_device_type

# Activation-cache codec: Blosc-zstd + BIT shuffle (was plain Zstd level 9). Decompresses multi-threaded --
# with BLOSC_NTHREADS=4 (set in the package __init__) a fetch reads ~30% faster than the old single-threaded
# zstd, the training/mutation IO hot path. On float16 activations bitshuffle beats byteshuffle: ~30% faster
# reads and ~5x faster writes at ~the same size. clevel=5 reads faster than 9. Same codec as the bands.
CACHE_COMPRESSION = hdf5plugin.Blosc(cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.BITSHUFFLE)

# Lossy pre-compression: keep only this many of float16's 10 mantissa bits (round-to-nearest) before writing.
# 4 bits shrinks the cache a further ~1.7x with the predicted Hi-C map still correlated 0.999998 with the
# full-precision map (the model sees f16/bf16 and averages runs). Set to 10 (or None) to store full precision.
CACHE_ROUND_BITS = 4

# Cache HDF5 chunk = (n_channels, CACHE_CHUNK_BINS). A fetch reads all channels over a large (10k-40k-bin)
# window, so the chunk wants to be big enough for Blosc to thread. 4096 reads the common ~10k window as fast
# as 1024, reads big 40k windows ~15% faster, and compresses ~50% faster (multi-threaded) at a hair smaller
# size; 8192 starts to over-read the 10k window. (512/1024 are too small -- fewer blocks per chunk.)
CACHE_CHUNK_BINS = 4096


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
        List of tuple ("replace", position, sequence) or ("invert"/"shuffle[k]", position, position2). Default is None.
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


def populate_microzoi_cache(
    cache_path,
    modfile,
    fasta,
    genome,
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
    genome : str
        Genome the FASTA is for (e.g. "hg38"/"mm10"). Stored as a file attr so a cache carries its genome by
        construction; training/inference reject a cache whose genome disagrees with the target/model.
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

    base_model = Microzoi(return_type="mha", **params["model"]).to(device)
    sd = torch.load(modfile, map_location=device, weights_only=True)
    base_model.load_state_dict(sd, strict=False)
    base_model.eval()

    # We'll store crop values via a linspace
    crop_values = np.round(np.linspace(crop_mha_range[0], crop_mha_range[1], N_runs)).astype(int)

    with h5py.File(cache_path, "w") as f:
        # Record some basic attributes
        f.attrs["genome"] = genome
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
                        chunks=(n_channels, min(CACHE_CHUNK_BINS, total_bins)),
                        **CACHE_COMPRESSION,
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
                        with torch.no_grad(), torch.autocast(torch_device_type(device)):
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
                        if CACHE_ROUND_BITS is not None:  # lossy: zero low mantissa bits for ~1.7x compression
                            arr = round_mantissa(arr, CACHE_ROUND_BITS)

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
