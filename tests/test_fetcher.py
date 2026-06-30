"""
Tests for ``CachedStochasticActivationFetcher`` (cache reads, run-averaging, and mutation patching).

These run on CPU with no real model or 700 GB cache. The strategy is three-fold:

1. **Synthetic cache** -- a tiny HDF5 with the same layout ``populate_microzoi_cache`` produces, but filled
   with *known* values: channel 0 of every column holds that column's absolute genomic bin index and
   channel 1 holds the run index. So slicing, reverse, and run-averaging can be checked exactly.

2. **Fake position-encoding model** -- a deterministic ``nn.Module`` honouring MicroZoi's mha contract
   (input [B,4,RF] -> output [B,C,RF/256 - 2*crop], cropped) whose every output bin encodes the summed
   base code over that bin. A ``replace`` therefore changes exactly the output bins it overlaps, so patch
   placement can be verified to the bin.

3. **Differential test** -- ``_recompute_patches`` must be byte-identical to the trusted
   ``fetch_tile_microzoi_activations`` for the same (model, fasta), across orientations / tile offsets /
   shifts / mutations. This pins the recompute geometry without needing a position oracle.
"""

import h5py
import numpy as np
import pytest
import torch
from torch import nn

from manta_hic.nn.manta import (
    BIN_BP,
    MICROZOI_RECEPTIVE_FIELD,
    CachedStochasticActivationFetcher,
    fetch_tile_microzoi_activations,
)


def _np(t):
    """The fetcher returns torch tensors now; unwrap to numpy for value checks."""
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t


RF = MICROZOI_RECEPTIVE_FIELD
MODEL_CHANNELS = 24  # fetcher appends 8 positional channels -> cache stores MODEL_CHANNELS + 8
N_CHANNELS = MODEL_CHANNELS + 8
N_RUNS = 3
OVERHANG_BP = 262144  # small overhang so the synthetic cache stays tiny
CHROM = "chr1"
CHROM_LEN = 1_600_000


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #
class FakeFasta:
    """Deterministic genome: each absolute position maps to a fixed base via a well-mixed hash.

    A murmur-style finalizer (rather than a linear congruential step) is essential: a linear hash is
    periodic mod 4, which would make every 256 bp bin have identical base composition and let a balanced
    replacement slip past a sum-based model undetected.
    """

    _BASES = np.frombuffer(b"ACGT", dtype="S1")

    def fetch(self, chrom, start, end):
        h = np.arange(start, end, dtype=np.uint64)
        h = (h ^ (h >> np.uint64(33))) * np.uint64(0xFF51AFD7ED558CCD)
        h = (h ^ (h >> np.uint64(33))) * np.uint64(0xC4CEB9FE1A85EC53)
        h = h ^ (h >> np.uint64(33))
        codes = (h % np.uint64(4)).astype(np.int64)
        return self._BASES[codes].tobytes().decode("ascii")


class FakePosModel(nn.Module):
    """
    Mimics ``MicroBorzoi(return_type="mha").forward(x, genome, offset, crop_mha)``.

    Input x: one-hot [B, 4, L] (L a multiple of 256). Output [B, MODEL_CHANNELS, L//256 - 2*crop_mha],
    where each output bin = sum of base codes (A=0,C=1,G=2,T=3) over that bin's 256 bp (broadcast across
    channels). Deterministic and content-encoding, so a mutation changes exactly the bins it touches.
    """

    def __init__(self, channels=MODEL_CHANNELS):
        super().__init__()
        self.channels = channels
        self.dummy = nn.Parameter(torch.zeros(1))  # gives the module a device for fetch_tile

    def forward(self, x, genome="hg38", offset=0, crop_mha=0, crop_result=True):
        x = x.float()
        codes = torch.tensor([0.0, 1.0, 2.0, 3.0], device=x.device).view(1, 4, 1)
        base = (x * codes).sum(dim=1)  # [B, L]
        b, length = base.shape
        binsum = base.view(b, length // BIN_BP, BIN_BP).sum(dim=2)  # [B, L//256]
        out = binsum.unsqueeze(1).repeat(1, self.channels, 1)  # [B, C, L//256]
        if crop_mha:
            out = out[:, :, crop_mha : out.shape[2] - crop_mha]
        return out


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cache_path(tmp_path_factory):
    """Write a synthetic cache: channel 0 = absolute genomic bin index, channel 1 = run index."""
    path = tmp_path_factory.mktemp("cache") / "synthetic.h5"
    rounded = (CHROM_LEN // BIN_BP) * BIN_BP
    region_start_bin = -OVERHANG_BP // BIN_BP
    total_bins = (rounded + 2 * OVERHANG_BP) // BIN_BP
    bin_index = (region_start_bin + np.arange(total_bins)).astype(np.float16)  # absolute genomic bin per column

    with h5py.File(path, "w") as f:
        f.attrs["N_runs"] = N_RUNS
        f.attrs["CACHE_OVERHANG_BP"] = OVERHANG_BP
        f.attrs["BIN_BP"] = BIN_BP
        f.attrs["max_shift_bp"] = 128
        f.attrs["crop_mha_range"] = (640, 1024)
        f.attrs["model_params"] = '{"model": {}}'
        for r in range(N_RUNS):
            g = f.create_group(f"run{r}")
            g.attrs["crop_mha_bins"] = 768
            g.attrs["shift_bp"] = 0  # keep patches phase-aligned with the (synthetic) background
            g.attrs["offset_bins"] = 0
            for orient in ("forward", "reverse"):  # both stored forward-oriented, as populate_cache does
                data = np.zeros((N_CHANNELS, total_bins), dtype=np.float16)
                data[0] = bin_index
                data[1] = r
                g.create_dataset(f"{CHROM}_{orient}", data=data)
    return str(path)


@pytest.fixture()
def fetcher(cache_path):
    fet = CachedStochasticActivationFetcher(cache_path, fasta_open=FakeFasta(), batch_size=2)
    fet._model = FakePosModel()  # bypass _fetch_microzoi_model (no real model blob needed)
    return fet


# --------------------------------------------------------------------------- #
# 1. Cache reads: slicing / reverse / averaging                               #
# --------------------------------------------------------------------------- #
def test_fetch_slice_alignment(fetcher):
    start, end = 0, 100 * BIN_BP
    for r in range(N_RUNS):
        arr = _np(fetcher.fetch(CHROM, start, end, run_idx=r))
        assert arr.shape == (N_CHANNELS, 100)
        # channel 0 must equal the absolute genomic bin index of each column
        np.testing.assert_array_equal(arr[0], np.arange(start // BIN_BP, end // BIN_BP))
        np.testing.assert_array_equal(arr[1], np.full(100, r))


def test_fetch_reverse_is_flipped(fetcher):
    start, end = 50 * BIN_BP, 150 * BIN_BP
    fwd = _np(fetcher.fetch(CHROM, start, end, run_idx=0))
    rev = _np(fetcher.fetch(CHROM, start, end, run_idx=0, reverse=True))
    np.testing.assert_array_equal(rev[0], fwd[0][::-1])
    np.testing.assert_array_equal(rev, fwd[:, ::-1])


def test_read_runs_averaging_math(fetcher):
    start, end = 10 * BIN_BP, 60 * BIN_BP
    single = fetcher._read_runs(CHROM, start, end, False, [1], "cpu")
    assert single.dtype == torch.float16
    avg = fetcher._read_runs(CHROM, start, end, False, [0, 1, 2], "cpu")
    assert avg.dtype == torch.float16  # averaging stays in float16 (matches what the model sees under autocast)
    np.testing.assert_allclose(_np(avg)[1], np.mean([0, 1, 2]))  # channel 1 = mean run index
    np.testing.assert_array_equal(_np(avg)[0], _np(single)[0])  # position channel unchanged by averaging


def test_fetch_n_runs_averages_all(fetcher):
    start, end = 0, 40 * BIN_BP
    arr = fetcher.fetch(CHROM, start, end, n_runs=N_RUNS)  # n_runs == N_runs -> all runs
    assert arr.dtype == torch.float16
    np.testing.assert_allclose(_np(arr)[1], np.mean(range(N_RUNS)))


def test_fetch_out_of_range_and_misaligned_raise(fetcher):
    with pytest.raises(ValueError):
        fetcher.fetch(CHROM, -(OVERHANG_BP + BIN_BP), 0, run_idx=0)  # before stored range
    with pytest.raises(ValueError):
        fetcher.fetch(CHROM, 100, 100 + BIN_BP, run_idx=0)  # not bin-aligned


# --------------------------------------------------------------------------- #
# 2. Recompute geometry: differential vs the trusted tiler                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("soff", [0, 5, 40])
@pytest.mark.parametrize("shift_bp", [0, 37])
@pytest.mark.parametrize("width_bins", [1536, 3200])  # one tile, several tiles
def test_recompute_patches_matches_tiler(fetcher, reverse, soff, shift_bp, width_bins):
    mid = 800_000 // BIN_BP * BIN_BP
    win_lo, win_hi = mid - (width_bins // 2) * BIN_BP, mid + (width_bins - width_bins // 2) * BIN_BP
    mut = [("replace", mid, "ACGTACGTAC" * 20)]  # 200 bp
    for mutate in (None, mut):
        (mine,) = fetcher._recompute_patches(
            fetcher._model, CHROM, win_lo, win_hi, [(mutate, shift_bp, soff)], reverse, 768, "cpu"
        )
        with torch.no_grad(), torch.autocast("cpu"):
            ref = fetch_tile_microzoi_activations(
                fetcher._model,
                fetcher.fasta_open,
                CHROM,
                win_lo,
                win_hi,
                mutate=mutate,
                reverse=reverse,
                start_offset_bins=soff,
                shift_bp=shift_bp,
                crop_mha_bins=768,
                batch_size=2,
            )
        assert tuple(mine.shape) == tuple(ref.shape)
        np.testing.assert_array_equal(_np(mine), _np(ref.half()))


# --------------------------------------------------------------------------- #
# 3. Matched pairs: invariant, alignment, averaging                            #
# --------------------------------------------------------------------------- #
# A Manta-style window wide enough to contain the RF/4 flank around the mutation.
WIN_START = 600_000 // BIN_BP * BIN_BP
WIN_END = 1_400_000 // BIN_BP * BIN_BP
MUT_POS = 1_000_000 // BIN_BP * BIN_BP


def _changed_columns(wt, mut):
    d = np.abs(_np(wt).astype(np.float32) - _np(mut).astype(np.float32)).sum(0)
    return np.flatnonzero(d > 0)


def _expected_mut_cols(pos, seq_len, reverse):
    """Result columns of the genomic bins overlapping a replacement of length seq_len at pos."""
    bins = range(pos // BIN_BP, (pos + seq_len - 1) // BIN_BP + 1)
    n_total = (WIN_END - WIN_START) // BIN_BP
    if reverse:
        return {n_total - 1 - (b - WIN_START // BIN_BP) for b in bins}
    return {b - WIN_START // BIN_BP for b in bins}


def test_matched_pair_invariant_single_run(fetcher):
    muts = [("replace", MUT_POS, "ACGT" * 50)]
    wt, mut = fetcher.fetch_matched_pair(CHROM, WIN_START, WIN_END, "cpu", muts, n_runs=1, run_idx=0)
    assert tuple(wt.shape) == tuple(mut.shape) == (N_CHANNELS, (WIN_END - WIN_START) // BIN_BP)
    # outside the recompute WINDOW (wider than the mutated region): wt is the untouched cached background.
    win_lo, win_hi = fetcher._mutation_window(muts, WIN_START, WIN_END, MICROZOI_RECEPTIVE_FIELD // 4)
    outside = np.ones(wt.shape[1], bool)
    outside[(win_lo - WIN_START) // BIN_BP : (win_hi - WIN_START) // BIN_BP] = False
    cache0 = _np(fetcher.fetch(CHROM, WIN_START, WIN_END, run_idx=0)).astype(np.float32)
    np.testing.assert_array_equal(_np(wt).astype(np.float32)[:, outside], cache0[:, outside])
    # mut == wt everywhere except the bins the replacement actually overlaps
    assert set(_changed_columns(wt, mut)).issubset(_expected_mut_cols(MUT_POS, 200, False))


@pytest.mark.parametrize("reverse", [False, True])
def test_matched_pair_patch_lands_at_mutation(fetcher, reverse):
    """Changed columns must map exactly to the genomic bins the replacement overlaps."""
    muts = [("replace", MUT_POS, "ACGT" * 50)]  # 200 bp
    wt, mut = fetcher.fetch_matched_pair(CHROM, WIN_START, WIN_END, "cpu", muts, n_runs=1, run_idx=0, reverse=reverse)
    changed = set(_changed_columns(wt, mut).tolist())
    assert changed, "mutation produced no change"
    assert changed.issubset(_expected_mut_cols(MUT_POS, 200, reverse))


@pytest.mark.parametrize("n_runs", [1, 2, 3])
@pytest.mark.parametrize("patch_off_bins", [0, 8])
def test_matched_pair_averaging_clean(fetcher, n_runs, patch_off_bins):
    muts = [("replace", MUT_POS, "ACGT" * 50)]
    wt, mut = fetcher.fetch_matched_pair(
        CHROM, WIN_START, WIN_END, "cpu", muts, n_runs=n_runs, patch_max_offset_bins=patch_off_bins
    )
    assert np.isfinite(_np(wt)).all() and np.isfinite(_np(mut)).all()
    # Under run averaging the shared backgrounds (and the unmutated annulus of the patch) cancel, so the
    # difference stays confined to the bins the replacement overlaps -- it does not smear across runs.
    assert set(_changed_columns(wt, mut)).issubset(_expected_mut_cols(MUT_POS, 200, False))


def test_fetch_activations_matches_matched_pair_mut(fetcher):
    """fetch_activations(mutated) equals the mutant half of the pair under identical sampling."""
    muts = [("replace", MUT_POS, "ACGT" * 50)]
    _, mut_pair = fetcher.fetch_matched_pair(CHROM, WIN_START, WIN_END, "cpu", muts, n_runs=1, run_idx=0)
    mut_solo = fetcher.fetch_activations(CHROM, WIN_START, WIN_END, "cpu", muts, n_runs=1, run_idx=0)
    np.testing.assert_array_equal(_np(mut_solo), _np(mut_pair))


# --------------------------------------------------------------------------- #
# 4. Validation / warnings                                                     #
# --------------------------------------------------------------------------- #
def test_insert_is_rejected(fetcher):
    with pytest.raises(ValueError, match="not supported"):
        fetcher.fetch_matched_pair(CHROM, WIN_START, WIN_END, "cpu", [("insert", MUT_POS, "ACGT")], n_runs=1)


def test_out_of_window_mutation_rejected(fetcher):
    with pytest.raises(ValueError, match="outside the window"):
        fetcher.fetch_matched_pair(CHROM, WIN_START, WIN_END, "cpu", [("replace", WIN_END + BIN_BP, "AC")], n_runs=1)


def test_matched_pair_requires_mutation(fetcher):
    with pytest.raises(ValueError, match="at least one mutation"):
        fetcher.fetch_matched_pair(CHROM, WIN_START, WIN_END, "cpu", [], n_runs=1)


def test_n_runs_none_warns(fetcher):
    with pytest.warns(UserWarning, match="n_runs not specified"):
        fetcher.fetch_activations(CHROM, 0, 40 * BIN_BP, n_runs=None)
