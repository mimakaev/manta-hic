"""
Write-side tests for the banded store (``io/banded_write.py``).

The genome-dependent parts (arms / Borzoi folds / cooltools expected) are validated on real coolers in the
prototype run; here we test the core, genome-independent correctness on a synthetic cooler: building the
band straight from pixels must reconstruct the cooler's own matrix exactly.
"""

import cooler
import numpy as np
import pandas as pd
import pytest

import manta_hic.io.banded_write as bw
from manta_hic.io.banded import square_from_band
from manta_hic.io.banded_write import chrom_band_from_pixels, coolers_to_banded


def _make_cooler(path, L, res=1000, n_diag=64, seed=0):
    """Write a tiny single-chromosome cooler with banded random counts at ``path``; return its str path."""
    rng = np.random.default_rng(seed)  # local + seeded: deterministic, order-independent
    bins = pd.DataFrame({"chrom": ["chr1"] * L, "start": np.arange(L) * res, "end": (np.arange(L) + 1) * res})
    i, j = np.triu_indices(L)
    keep = (j - i) < n_diag
    px = pd.DataFrame({"bin1_id": i[keep], "bin2_id": j[keep], "count": rng.poisson(5, size=keep.sum())})
    cooler.create_cooler(str(path), bins, px, dtypes={"count": "int32"}, ordered=True)
    return str(path)


@pytest.fixture()
def synthetic_cooler(tmp_path):
    """A tiny single-chromosome cooler with banded random counts (d < 64). tmp_path is auto-cleaned."""
    L, n_diag = 300, 64
    path = _make_cooler(tmp_path / "c.cool", L, n_diag=n_diag)
    return cooler.Cooler(path), L, n_diag


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


def test_dry_run_returns_summary_without_writing(synthetic_cooler, tmp_path, monkeypatch):
    """dry_run validates/opens coolers and returns a summary, but writes no file and skips heavy compute."""
    # stub chromarms so the test is self-contained (no bioframe chromsizes/centromere fetch)
    monkeypatch.setattr(
        bw, "chromarms", lambda genome: pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [10**9], "name": ["p"]})
    )
    clr, L, n_diag = synthetic_cooler
    out = tmp_path / "out.bhic.h5"
    summary = coolers_to_banded(
        [clr.uri], ["a"], str(out), genome="hg38", resolution=1000, group_name="g", n_diag=n_diag, dry_run=True
    )
    assert not out.exists()  # nothing written
    assert summary["n_channels"] == 1 and summary["shortnames"] == ["a"]
    assert summary["resolution"] == 1000 and summary["n_arms"] > 0


def test_dry_run_rejects_bad_shortnames(synthetic_cooler, tmp_path):
    clr, L, n_diag = synthetic_cooler  # shortname count is checked before any arm/bioframe work
    with pytest.raises(ValueError, match="shortnames"):
        coolers_to_banded(
            [clr.uri], ["a", "b"], str(tmp_path / "x"), genome="hg38", resolution=1000, group_name="g", dry_run=True
        )


def test_assign_arm_fold_ids(monkeypatch):
    """arm_id from arms, fold_id from the (polars) Borzoi fold_df incl. 'foldN' parsing; excluded -> -1."""
    import polars as pl

    res, L = 100, 10
    bins = pd.DataFrame({"chrom": ["chrX"] * L, "start": np.arange(L) * res, "end": (np.arange(L) + 1) * res})
    arms = pd.DataFrame({"chrom": ["chrX", "chrX"], "start": [0, 500], "end": [500, 1000], "name": ["p", "q"]})
    fake = pl.DataFrame(
        {
            "genome": ["hg38"] * 2,
            "chrom": ["chrX"] * 2,
            "start": [0, 600],
            "end": [600, 1000],
            "fold": ["fold3", "fold5"],
        }
    )
    monkeypatch.setattr(bw, "fold_df", fake)
    arm_id, fold_id = bw.assign_arm_fold_ids(bins, arms, "hg38")
    mid = (bins["start"].values + bins["end"].values) // 2
    assert (arm_id[mid < 500] == 0).all() and (arm_id[mid >= 500] == 1).all()
    assert (fold_id[mid < 600] == 3).all() and (fold_id[mid >= 600] == 5).all()  # "fold3"/"fold5" -> 3/5
    # a chromosome absent from arms/folds stays excluded (-1)
    other = pd.DataFrame({"chrom": ["chrZ"], "start": [0], "end": [res]})
    a2, f2 = bw.assign_arm_fold_ids(other, arms, "hg38")
    assert a2[0] == -1 and f2[0] == -1


def test_mismatched_bin_grid_rejected(tmp_path):
    """Coolers in a group with different bin grids must be rejected before any conversion work."""
    p1 = _make_cooler(tmp_path / "a.cool", 300, seed=1)
    p2 = _make_cooler(tmp_path / "b.cool", 280, seed=2)  # different chrom length -> different grid
    with pytest.raises(ValueError, match="bin grid mismatch"):
        coolers_to_banded([p1, p2], ["a", "b"], str(tmp_path / "o.h5"), genome="hg38", resolution=1000, group_name="g")
