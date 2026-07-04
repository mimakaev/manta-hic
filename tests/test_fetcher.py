"""
Tests for ``CachedMicrozoiFetcher`` (cache reads, run-averaging, and mutation patching).

These run on CPU with no real model or 700 GB cache. The strategy is three-fold:

1. **Synthetic cache** -- a tiny HDF5 with the same layout ``populate_microzoi_cache`` produces, but filled
   with *known* values: channel 0 of every column holds that column's absolute genomic bin index and
   channel 1 holds the run index. So slicing, reverse, and run-averaging can be checked exactly.

2. **Fake position-encoding model** -- a deterministic ``nn.Module`` honouring MicroZoi's mha contract
   (input [B,4,RF] -> output [B,C,RF/256 - 2*crop], cropped) whose every output bin encodes the summed
   base code over that bin. A ``replace`` therefore changes exactly the output bins it overlaps, so patch
   placement can be verified to the bin.

3. **Differential test** -- the live ``_recompute_jobs`` must be byte-identical to the trusted
   ``fetch_tile_microzoi_activations`` for the same (model, fasta), across orientations / tile offsets /
   shifts / mutations. This pins the recompute geometry without needing a position oracle.
"""

import h5py
import numpy as np
import pytest
import torch
from torch import nn

from manta_hic.nn.fetchers import (
    BIN_BP,
    MICROZOI_RECEPTIVE_FIELD,
    CachedMicrozoiFetcher,
)
from manta_hic.nn.fill_cache import fetch_tile_microzoi_activations


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
    Mimics ``Microzoi(return_type="mha").forward(x, genome, offset, crop_mha)``.

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
    fet = CachedMicrozoiFetcher(cache_path, fasta_open=FakeFasta(), batch_size=2)
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
# 2. Recompute geometry: the shipping pooled path, byte-exact vs the tiler      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("width_bins", [1536, 3200])  # one tile, several tiles
def test_recompute_jobs_matches_tiler(fetcher, reverse, width_bins):
    """The live L1 recompute (``_recompute_jobs``, always tile-offset 0) must be byte-identical to the trusted
    ``fetch_tile_microzoi_activations`` at ``start_offset_bins=0, shift_bp=0`` -- and pooling a WT + mutant job
    in one batch must give each the same result as a standalone tiler call."""
    mid = 800_000 // BIN_BP * BIN_BP
    win_lo, win_hi = mid - (width_bins // 2) * BIN_BP, mid + (width_bins - width_bins // 2) * BIN_BP
    mut = (("replace", mid, "ACGTACGTAC" * 20),)  # 200 bp
    jobs = [(CHROM, win_lo, win_hi, None, reverse), (CHROM, win_lo, win_hi, mut, reverse)]  # WT + mutant, pooled
    patches = fetcher._recompute_jobs(fetcher._model, jobs, 768, "cpu")
    for job, mutate in ((jobs[0], None), (jobs[1], mut)):
        with torch.no_grad(), torch.autocast("cpu"):
            ref = fetch_tile_microzoi_activations(
                fetcher._model,
                fetcher.fasta_open,
                CHROM,
                win_lo,
                win_hi,
                mutate=mutate,
                reverse=reverse,
                start_offset_bins=0,
                shift_bp=0,
                crop_mha_bins=768,
                batch_size=2,
            )
        assert tuple(patches[job].shape) == tuple(ref.shape)
        np.testing.assert_array_equal(_np(patches[job]), _np(ref.half()))


# --------------------------------------------------------------------------- #
# 3. L1 matched pairs via fetch_activations_batch (spec -> activations)         #
# --------------------------------------------------------------------------- #
# resolution == BIN_BP keeps the map window on the cache bin grid; the window is wide enough to hold a tile.
RES, NB, PAD, MAP = BIN_BP, 2000, 256, 3000 * BIN_BP
MUT_POS = 1_000_000


def _changed_columns(wt, mut):
    d = np.abs(_np(wt).astype(np.float32) - _np(mut).astype(np.float32)).sum(0)
    return np.flatnonzero(d > 0)


def _run_pair(fetcher, mut_pos, reverse=False):
    from manta_hic.nn.specs import background_for, variant_specs

    mut = (("replace", mut_pos, "ACGT" * 50),)  # 200 bp
    bg = background_for(CHROM, MAP, run_idx=0, reverse=reverse, mutations_superset=mut)
    wt, m = fetcher.fetch_activations_batch(
        variant_specs(bg, {"wt": None, "mut": mut}), resolution=RES, n_bins=NB, bins_pad=PAD, device="cpu"
    )
    fs = MAP - PAD * RES
    return bg, wt, m, fs


def _tile_cols(bg, fs, fe, reverse):
    """[lo, hi) array columns the tile pattern occupies (accounting for the reverse flip)."""
    lo, hi = bg.tiles[0][0], bg.tiles[-1][1]
    return ((fe - hi) // BIN_BP, (fe - lo) // BIN_BP) if reverse else ((lo - fs) // BIN_BP, (hi - fs) // BIN_BP)


def test_l1_matched_pair_localized_and_clean(fetcher):
    bg, wt, mut, fs = _run_pair(fetcher, MUT_POS)
    assert tuple(wt.shape) == tuple(mut.shape) == (N_CHANNELS, NB + 2 * PAD)
    changed = _changed_columns(wt, mut)
    assert len(changed) > 0, "mutation produced no change"
    # the difference sits at the mutation's bin(s), inside the tile, and nowhere else
    mut_col = (MUT_POS - fs) // BIN_BP
    assert changed.min() >= mut_col - 1 and changed.max() <= mut_col + 1
    # WT recompute leaves the cached background untouched OUTSIDE the tiles (clean pair everywhere else)
    tlo, thi = _tile_cols(bg, fs, fs + (NB + 2 * PAD) * RES, reverse=False)
    cache = _np(fetcher.fetch(CHROM, fs, fs + (NB + 2 * PAD) * RES, run_idx=0)).astype(np.float32)
    outside = np.ones(wt.shape[1], bool)
    outside[tlo:thi] = False
    np.testing.assert_array_equal(_np(wt).astype(np.float32)[:, outside], cache[:, outside])


def test_l1_matched_pair_reverse_localized(fetcher):
    bg, wt, mut, fs = _run_pair(fetcher, MUT_POS, reverse=True)
    changed = _changed_columns(wt, mut)
    assert len(changed) > 0
    tlo, thi = _tile_cols(bg, fs, fs + (NB + 2 * PAD) * RES, reverse=True)
    assert changed.min() >= tlo and changed.max() < thi  # localized to the (flipped) tile region


def test_l1_dedup_one_cache_read(fetcher):
    from manta_hic.nn.specs import background_for, variant_specs

    reads = []
    orig = fetcher.fetch
    fetcher.fetch = lambda *a, **k: (reads.append(1), orig(*a, **k))[1]
    mut = (("replace", MUT_POS, "ACGT" * 50),)
    bg = background_for(CHROM, MAP, run_idx=0, mutations_superset=mut)
    specs = variant_specs(bg, {"wt": None, "mut": mut, "mut2": mut})  # 3 specs, same background
    fetcher.fetch_activations_batch(specs, resolution=RES, n_bins=NB, bins_pad=PAD, device="cpu")
    assert len(reads) == 1  # one shared cache read


# --------------------------------------------------------------------------- #
# 4. Validation (length-changing mutations rejected at spec-build time)         #
# --------------------------------------------------------------------------- #
def test_insert_is_rejected():
    from manta_hic.nn.specs import background_for

    with pytest.raises(ValueError):
        background_for(CHROM, MAP, run_idx=0, mutations_superset=[("insert", MUT_POS, "ACGT")])
