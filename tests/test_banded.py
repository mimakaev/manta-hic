"""
Tests for the banded ("turned") Hi-C storage prototype (``manta_hic/io/banded.py``).

Three things are pinned down:
1. **Round-trip** -- a window reconstructed from the band equals the direct dense submatrix (symmetry).
2. **Eligibility** -- ``eligible_starts`` (prefix-sum inclusion criteria) matches a brute-force per-window
   check of the same criteria.
3. **Target parity** -- feeding the reconstructed ``(hic, weight, exp)`` to the existing
   ``create_expected_matrix`` yields the identical expected matrix as feeding the dense inputs, i.e. the
   banded store reproduces the current training target exactly.
"""

import numpy as np
import pytest
import torch

from manta_hic.io.banded import BandedHicStore, band_from_dense, square_from_band
from manta_hic.ops.hic_ops import create_expected_matrix

rng = np.random.default_rng(0)


def _symmetric_counts(L, C):
    """A small Hi-C-like symmetric count matrix [C, L, L] that decays with distance."""
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
# Fixture: a small chromosome with an excluded centromere, two arms, folds     #
# --------------------------------------------------------------------------- #
def _make_store(n_bins=200, C=2, n_diag=24):
    M = _symmetric_counts(n_bins, C)
    weights = rng.uniform(0.2, 1.0, size=(C, n_bins)).astype(np.float32)
    bad = rng.random((C, n_bins)) < 0.05
    # arm 0 = [0,90), excluded centromere [90,100), arm 1 = [100,200)
    arm_id = np.empty(n_bins, np.int32)
    arm_id[:90] = 0
    arm_id[90:100] = -1
    arm_id[100:] = 1
    weights[bad] = 0.0  # zero bad bins (bad is [C, n_bins], matching weights)
    # Borzoi-like contiguous fold blocks
    fold_id = np.zeros(n_bins, np.int32)
    fold_id[60:130] = 1
    fold_id[130:] = 2
    exp_per_arm = rng.uniform(0.1, 1.0, size=(C, 2, n_diag)).astype(np.float32)
    exp_per_arm[:, :, :2] = 0.0  # first two diagonals zeroed, as in cool_io
    store = BandedHicStore.from_dense(M, weights, bad, arm_id, fold_id, exp_per_arm, n_diag)
    return store, M


def _brute_eligible(store, n, min_fraction, fold):
    out = []
    for a in range(store.n_bins - n + 1):
        seg_arm = store.arm_id[a : a + n]
        if seg_arm[0] == -1 or not np.all(seg_arm == seg_arm[0]):
            continue
        if fold is not None and not np.all(store.fold_id[a : a + n] == fold):
            continue
        win_mean = store.bad[:, a : a + n].mean(axis=1)
        if np.sqrt((win_mean**2).mean()) >= min_fraction:
            continue
        out.append(a)
    return np.array(out, dtype=np.int64)


# --------------------------------------------------------------------------- #
# 2. Eligibility == brute force, for every criterion                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [16, 32, 50])
@pytest.mark.parametrize("min_fraction", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("fold", [None, 0, 1, 2])
def test_eligible_starts_matches_brute_force(n, min_fraction, fold):
    store, _ = _make_store()
    got = store.eligible_starts(n, min_fraction=min_fraction, fold=fold)
    want = _brute_eligible(store, n, min_fraction, fold)
    np.testing.assert_array_equal(got, want)


def test_eligible_start_fraction_matches_store():
    """The conversion's watch-metric helper must count exactly what eligible_starts(fold=None) returns."""
    from manta_hic.io.banded_write import eligible_start_fraction

    store, _ = _make_store()
    for n, mf in [(16, 0.1), (16, 0.5), (32, 1.0)]:
        frac, n_elig, n_cand = eligible_start_fraction(store.bad, store.arm_id, n, min_fraction=mf)
        assert n_elig == len(store.eligible_starts(n, min_fraction=mf, fold=None))
        assert n_cand >= n_elig and (frac == (n_elig / n_cand if n_cand else 0.0))


def test_get_window_rejects_excluded_arm():
    """get_window on a bin in an excluded region (arm_id=-1) must raise, not silently use the last arm."""
    store, _ = _make_store()
    n = 8
    assert store.arm_id[92] == -1  # inside the excluded centromere [90,100)
    with pytest.raises(ValueError, match="excluded region"):
        store.get_window(92, n)


def test_eligibility_never_crosses_centromere_or_arm():
    store, _ = _make_store()
    n = 16
    for a in store.eligible_starts(n, min_fraction=1.0):
        seg = store.arm_id[a : a + n]
        assert seg[0] != -1 and np.all(seg == seg[0])  # one arm, not excluded
    # a window straddling the centromere [90,100) must never be returned
    assert not any(a < 100 and a + n > 90 and a + n <= 100 for a in store.eligible_starts(n, min_fraction=1.0))


# --------------------------------------------------------------------------- #
# 3. Target parity: banded reconstruction -> create_expected_matrix == dense   #
# --------------------------------------------------------------------------- #
def test_create_expected_matrix_parity():
    store, M = _make_store()
    n = 16
    a = int(store.eligible_starts(n, min_fraction=1.0)[3])  # some valid window
    arm = int(store.arm_id[a])

    hic_b, weight_b, exp_b = store.get_window(a, n)  # from the band
    hic_d = M[:, a : a + n, a : a + n]  # from the dense matrix
    weight_d = store.weights[:, a : a + n]
    exp_d = store.exp[:, arm]

    # the reconstructions themselves must be identical
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
