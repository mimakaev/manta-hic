"""
Tests for the banded ("turned") Hi-C storage (``manta_hic/io/banded.py``).

Three things are pinned down:
1. **Round-trip** -- a window reconstructed from the band equals the direct dense submatrix (symmetry).
2. **Eligibility** -- ``eligible_positions`` (prefix-sum inclusion criteria) matches a brute-force per-window
   check of the same criteria.
3. **Target parity** -- feeding the reconstructed ``(hic, weight, exp)`` to ``create_expected_matrix`` yields the
   identical expected matrix as feeding the dense inputs.

The read side is a single owner, :class:`BandedHicFile`; these tests build a one-chromosome in-memory file via
``from_arrays`` with ``resolution=1`` (so a global position, a local bin, and a base-pair coordinate coincide).
"""

import numpy as np
import pytest
import torch

from manta_hic.io.banded import BandedHicFile, band_from_dense, square_from_band
from manta_hic.ops.hic_ops import create_expected_matrix


def _symmetric_counts(L, C):
    """A small Hi-C-like symmetric count matrix [C, L, L] that decays with distance."""
    rng = np.random.default_rng(L * 100 + C)  # local + seeded: deterministic, order-independent
    out = np.zeros((C, L, L), dtype=np.float32)
    for c in range(C):
        a = rng.poisson(3.0, size=(L, L)).astype(np.float32)
        m = (a + a.T) / 2
        dist = np.abs(np.subtract.outer(np.arange(L), np.arange(L)))
        out[c] = m / (1 + dist)  # decay with genomic distance
    return out


# --------------------------------------------------------------------------- #
# 1. Round-trip                                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_diag,n", [(24, 16), (16, 16), (32, 8)])
def test_window_reconstruction_matches_dense(n_diag, n):
    L, C = 80, 2
    M = _symmetric_counts(L, C)
    band = band_from_dense(M, n_diag)
    assert band.shape == (C, L, n_diag)
    for a in (0, 5, 17, L - n):
        np.testing.assert_array_equal(square_from_band(band, a, n), M[:, a : a + n, a : a + n])


def test_band_rejects_out_of_range():
    band = band_from_dense(_symmetric_counts(40, 1), 16)
    with pytest.raises(ValueError):
        square_from_band(band, 30, 16)  # 30+16 > 40
    with pytest.raises(ValueError):
        square_from_band(band, 0, 20)  # 20 > n_diag=16


# --------------------------------------------------------------------------- #
# Fixture: a one-chromosome file with an excluded centromere, two arms, folds   #
# --------------------------------------------------------------------------- #
def _make_file(n_bins=200, C=2, n_diag=24):
    rng = np.random.default_rng(1234)  # local + seeded: deterministic, order-independent
    M = _symmetric_counts(n_bins, C)
    weights = rng.uniform(0.2, 1.0, size=(C, n_bins)).astype(np.float32)
    bad = rng.random((C, n_bins)) < 0.05
    # arm 0 = [0,90), excluded centromere [90,100), arm 1 = [100,200)
    arm_id = np.empty(n_bins, np.int32)
    arm_id[:90] = 0
    arm_id[90:100] = -1
    arm_id[100:] = 1
    weights[bad] = 0.0
    # Borzoi-like contiguous fold blocks
    fold_id = np.zeros(n_bins, np.int32)
    fold_id[60:130] = 1
    fold_id[130:] = 2
    exp_per_arm = rng.uniform(0.1, 1.0, size=(C, 2, n_diag)).astype(np.float32)
    exp_per_arm[:, :, :2] = 0.0  # first two diagonals zeroed
    band = band_from_dense(M, n_diag)
    bf = BandedHicFile.from_arrays(
        ["chr1"], [band], [weights], [bad], [arm_id], [fold_id], exp_per_arm, resolution=1, n_diag=n_diag
    )
    return bf, M


def _brute_eligible(bf, n, max_bad_fraction, fold, overlap_threshold=0.9):
    fset = None if fold is None else ({int(fold)} if np.isscalar(fold) else {int(x) for x in fold})
    out = []
    for a in range(bf.total_bins - n + 1):
        seg_arm = bf.arm_id[a : a + n]
        if seg_arm[0] == -1 or not np.all(seg_arm == seg_arm[0]):
            continue
        win_mean = bf.bad[:, a : a + n].mean(axis=1)
        if np.sqrt((win_mean**2).mean()) >= max_bad_fraction:
            continue
        if fset is not None:  # keep if >= overlap_threshold of the window's bins are in the fold set
            if np.isin(bf.fold_id[a : a + n], list(fset)).mean() < overlap_threshold:
                continue
        out.append(a)
    return np.array(out, dtype=np.int64)


# --------------------------------------------------------------------------- #
# 2. Eligibility == brute force, for every criterion                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [16, 32, 50])
@pytest.mark.parametrize("max_bad_fraction", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("fold", [None, 0, 1, 2])
def test_eligible_positions_match_brute_force(n, max_bad_fraction, fold):
    bf, _ = _make_file()
    got = bf.eligible_positions(n, max_bad_fraction=max_bad_fraction, fold=fold)
    want = _brute_eligible(bf, n, max_bad_fraction, fold)
    np.testing.assert_array_equal(got, want)


def test_eligible_start_fraction_matches_file():
    """The conversion's watch-metric helper must count exactly what eligible_positions(fold=None) returns."""
    from manta_hic.io.banded_write import eligible_start_fraction

    bf, _ = _make_file()
    for n, mf in [(16, 0.1), (16, 0.5), (32, 1.0)]:
        frac, n_elig, n_cand = eligible_start_fraction(bf.bad, bf.arm_id, n, max_bad_fraction=mf)
        assert n_elig == len(bf.eligible_positions(n, max_bad_fraction=mf, fold=None))
        assert n_cand >= n_elig and (frac == (n_elig / n_cand if n_cand else 0.0))


@pytest.mark.parametrize("n", [16, 32])
@pytest.mark.parametrize("max_bad_fraction", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("fold", [None, 0, 1, 2])
def test_is_eligible_matches_eligible_positions(n, max_bad_fraction, fold):
    """The O(1) coordinate check must agree with the vectorized position set for every bin (resolution=1)."""
    bf, _ = _make_file()
    starts = set(int(a) for a in bf.eligible_positions(n, max_bad_fraction=max_bad_fraction, fold=fold))
    for a in range(bf.total_bins - n + 1):
        assert bf.is_eligible("chr1", a, n, max_bad_fraction=max_bad_fraction, fold=fold) == (a in starts)


def test_is_eligible_rejects_out_of_bounds_and_nonpositive():
    bf, _ = _make_file()
    assert not bf.is_eligible("chr1", bf.total_bins - 5, 16)  # runs off the end
    assert not bf.is_eligible("chr1", 0, 0)
    assert not bf.is_eligible("chrZ", 0, 16)  # unknown chromosome


def test_fold_fraction_tolerates_boundaries():
    """A fold set keeps windows that are >= overlap_threshold in that set, so a superset {0,2} contains at least
    the union of single-fold {0} and {2} windows, and every kept window is really >= 90% inside {0,2}."""
    bf, _ = _make_file()
    n = 16
    only0 = set(int(a) for a in bf.eligible_positions(n, fold=0))
    only2 = set(int(a) for a in bf.eligible_positions(n, fold=2))
    both = set(int(a) for a in bf.eligible_positions(n, fold={0, 2}))
    assert (only0 | only2) <= both
    for a in both:
        assert np.isin(bf.fold_id[a : a + n], [0, 2]).mean() >= 0.9


def test_window_at_rejects_excluded_arm():
    bf, _ = _make_file()
    assert bf.arm_id[92] == -1  # inside the excluded centromere [90,100)
    with pytest.raises(ValueError, match="excluded region"):
        bf.window_at(92, 8)


def test_window_at_rejects_arm_crossing_and_nonpositive_n():
    bf, _ = _make_file()
    with pytest.raises(ValueError, match="crosses an arm boundary"):
        bf.window_at(85, 20)  # [85,105) spans arm0 -> excluded -> arm1
    with pytest.raises(ValueError, match="must be positive"):
        bf.window_at(10, 0)


def test_eligibility_never_crosses_centromere_or_arm():
    bf, _ = _make_file()
    n = 16
    for a in bf.eligible_positions(n, max_bad_fraction=1.0):
        seg = bf.arm_id[a : a + n]
        assert seg[0] != -1 and np.all(seg == seg[0])  # one arm, not excluded
    assert not any(a < 100 and a + n > 90 and a + n <= 100 for a in bf.eligible_positions(n, max_bad_fraction=1.0))


# --------------------------------------------------------------------------- #
# 3. Target parity: banded reconstruction -> create_expected_matrix == dense   #
# --------------------------------------------------------------------------- #
def test_create_expected_matrix_parity():
    bf, M = _make_file()
    n = 16
    a = int(bf.eligible_positions(n, max_bad_fraction=1.0)[3])  # some valid window
    arm = int(bf.arm_id[a])

    hic_b, weight_b, exp_b = bf.window_at(a, n)  # from the band
    hic_d = M[:, a : a + n, a : a + n]  # from the dense matrix
    weight_d = bf.weights[:, a : a + n]
    exp_d = bf.exp[:, arm]

    np.testing.assert_array_equal(hic_b, hic_d)
    np.testing.assert_array_equal(weight_b, weight_d)
    np.testing.assert_array_equal(exp_b, exp_d)

    def expected(hic, weight, exp):
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).float().unsqueeze(0)  # add batch dim
        snippet, expmat = create_expected_matrix(t(hic), t(weight), t(exp))
        return snippet.numpy(), expmat.numpy()

    snip_b, exp_mat_b = expected(hic_b, weight_b, exp_b)
    snip_d, exp_mat_d = expected(hic_d, weight_d, exp_d)
    np.testing.assert_array_equal(snip_b, snip_d)
    np.testing.assert_array_equal(exp_mat_b, exp_mat_d)
