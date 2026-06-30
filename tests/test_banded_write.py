"""
Write-side tests for the banded store (``io/banded_write.py``).

The genome-dependent parts (arms / Borzoi folds / cooltools expected) are validated on real coolers in the
prototype run; here we test the core, genome-independent correctness on a synthetic cooler: building the
band straight from pixels must reconstruct the cooler's own matrix exactly.
"""

import os
import tempfile

import cooler
import numpy as np
import pandas as pd
import pytest

from manta_hic.io.banded import square_from_band
from manta_hic.io.banded_write import chrom_band_from_pixels, coolers_to_banded

rng = np.random.default_rng(0)


def _make_cooler(L, res=1000, n_diag=64):
    """Write a tiny single-chromosome cooler with banded random counts; return its path."""
    bins = pd.DataFrame({"chrom": ["chr1"] * L, "start": np.arange(L) * res, "end": (np.arange(L) + 1) * res})
    i, j = np.triu_indices(L)
    keep = (j - i) < n_diag
    px = pd.DataFrame({"bin1_id": i[keep], "bin2_id": j[keep], "count": rng.poisson(5, size=keep.sum())})
    path = tempfile.mktemp(suffix=".cool")
    cooler.create_cooler(path, bins, px, dtypes={"count": "int32"}, ordered=True)
    return path


@pytest.fixture()
def synthetic_cooler():
    """A tiny single-chromosome cooler with banded random counts (d < 64)."""
    L, n_diag = 300, 64
    path = _make_cooler(L, n_diag=n_diag)
    yield cooler.Cooler(path), L, n_diag
    os.remove(path)


def test_band_from_pixels_reconstructs_cooler(synthetic_cooler):
    clr, L, n_diag = synthetic_cooler
    band = chrom_band_from_pixels(clr, "chr1", n_diag=n_diag)
    assert band.shape == (L, n_diag) and band.dtype == np.int16
    dense = np.clip(np.nan_to_num(clr.matrix(balance=False).fetch("chr1")), 0, 32000).astype(np.int64)
    # every n_diag-sized window reconstructed from the band equals the cooler's own submatrix
    for a in (0, 17, 100, L - n_diag):
        rec = square_from_band(band[None], a, n_diag)[0].astype(np.int64)
        np.testing.assert_array_equal(rec, dense[a : a + n_diag, a : a + n_diag])


def test_band_clips_to_int16(synthetic_cooler):
    clr, L, n_diag = synthetic_cooler
    band = chrom_band_from_pixels(clr, "chr1", n_diag=n_diag)
    assert band.max() <= 32000 and band.min() >= 0


def test_dry_run_returns_summary_without_writing(synthetic_cooler):
    """dry_run validates/opens coolers and returns a summary, but writes no file and skips heavy compute."""
    clr, L, n_diag = synthetic_cooler
    out = tempfile.mktemp(suffix=".bhic.h5")
    # genome="hg38" arms won't contain "chr1" of a 300-bin toy, so it lands in chroms_dropped -- fine for dry-run.
    summary = coolers_to_banded(
        [clr.uri], ["a"], out, genome="hg38", resolution=1000, group_name="g", n_diag=n_diag, dry_run=True
    )
    assert not os.path.exists(out)  # nothing written
    assert summary["n_channels"] == 1 and summary["shortnames"] == ["a"]
    assert summary["resolution"] == 1000 and summary["n_arms"] > 0


def test_dry_run_rejects_bad_shortnames(synthetic_cooler):
    clr, L, n_diag = synthetic_cooler
    with pytest.raises(ValueError, match="shortnames"):
        coolers_to_banded([clr.uri], ["a", "b"], "x", genome="hg38", resolution=1000, group_name="g", dry_run=True)


def test_mismatched_bin_grid_rejected():
    """Coolers in a group with different bin grids must be rejected before any conversion work."""
    p1, p2 = _make_cooler(300), _make_cooler(280)  # different chrom length -> different grid
    out = tempfile.mktemp(suffix=".bhic.h5")
    try:
        with pytest.raises(ValueError, match="bin grid mismatch"):
            coolers_to_banded([p1, p2], ["a", "b"], out, genome="hg38", resolution=1000, group_name="g")
    finally:
        for p in (p1, p2, out):
            if os.path.exists(p):
                os.remove(p)
