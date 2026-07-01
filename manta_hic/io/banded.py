"""
Banded ("turned") Hi-C target storage -- prototype of the read side.

Instead of overlapping ``actual_size x actual_size`` square tiles (``io/cool_io.py``), store the first
``n_diag`` diagonals of each chromosome's contact map densely and once:

    band[c, x, d] = M_c[x, x + d]      for d in 0..n_diag-1

This is the layout of ``OnDiagonalHicAggregator`` (save_mcools/save_to_coolers_prototype.ipynb), reused
here for the *training target* rather than for aggregating predictions. It is gap-free, ~6x smaller than
overlapping squares, and any window is reconstructed on the fly by symmetry, so display never fails and
tile accept/reject is a sampling-time policy (per-bin vectors + prefix sums) rather than a baked-in build
decision. See docs/HIC_STORAGE.md.

This module is NumPy for clarity and a correctness reference; the production store would back ``band`` with
chunked HDF5 and reconstruct on-device with torch (the gather is identical). It deliberately does not read
real coolers -- that (and ``cooltools`` expected) is the build side, to be done un-sandboxed.
"""

import numpy as np

# --------------------------------------------------------------------------- #
# Band <-> square conversions                                                  #
# --------------------------------------------------------------------------- #


def band_from_dense(matrix: np.ndarray, n_diag: int) -> np.ndarray:
    """
    Encode a dense symmetric contact matrix into the banded layout, keeping the first ``n_diag`` diagonals.

    Parameters
    ----------
    matrix : np.ndarray
        ``[..., L, L]`` symmetric contact matrix (leading dims, e.g. channels, are kept).
    n_diag : int
        Number of diagonals to retain (``d = 0..n_diag-1``).

    Returns
    -------
    np.ndarray
        ``[..., L, n_diag]`` where ``band[..., x, d] = matrix[..., x, x + d]`` (0 where ``x + d >= L``).
    """
    L = matrix.shape[-1]
    band = np.zeros(matrix.shape[:-2] + (L, n_diag), dtype=matrix.dtype)
    for d in range(min(n_diag, L)):
        rows = np.arange(L - d)
        band[..., : L - d, d] = matrix[..., rows, rows + d]
    return band


def square_from_band(band: np.ndarray, start_bin: int, n: int) -> np.ndarray:
    """
    Reconstruct the ``n x n`` symmetric submatrix ``M[a:a+n, a:a+n]`` from a band ``[..., n_bins, n_diag]``.

    Uses ``M[i, j] = band[min(i, j), |i - j|]`` (a single gather). Requires ``n_diag >= n`` and the window
    to lie within the stored bins.

    Parameters
    ----------
    band : np.ndarray
        ``[..., n_bins, n_diag]`` banded data (leading dims kept).
    start_bin : int
        First bin of the window (``a``).
    n : int
        Window side length in bins.
    """
    a = start_bin
    n_bins, n_diag = band.shape[-2], band.shape[-1]
    if n > n_diag:
        raise ValueError(f"window {n} exceeds stored diagonals {n_diag}")
    if a < 0 or a + n > n_bins:
        raise ValueError(f"window [{a}, {a + n}) outside stored bins [0, {n_bins})")
    ii, jj = np.indices((n, n))
    x = a + np.minimum(ii, jj)
    d = np.abs(ii - jj)
    return band[..., x, d]  # [..., n, n]


# --------------------------------------------------------------------------- #
# Read-side store + sampling-time tile selection                              #
# --------------------------------------------------------------------------- #


class BandedHicStore:
    """
    Read-side banded store for one chromosome (the production version is one HDF5 group per chromosome).

    Parameters
    ----------
    band : np.ndarray
        ``[C, n_bins, n_diag]`` contact counts (band layout).
    weights : np.ndarray
        ``[C, n_bins]`` per-bin balancing weights (bad bins already zeroed).
    bad : np.ndarray
        ``[C, n_bins]`` boolean per-channel bad-bin mask.
    arm_id : np.ndarray
        ``[n_bins]`` int chromosomal-arm id; ``-1`` marks excluded bins (chrM/chrY/gap).
    fold_id : np.ndarray
        ``[n_bins]`` int Borzoi-fold id per bin.
    exp : np.ndarray
        ``[C, n_arms, n_diag]`` per-arm per-distance expected (first 2 diagonals zeroed), or
        ``[C, n_diag]`` if a single arm.
    """

    def __init__(self, band, weights, bad, arm_id, fold_id, exp):
        self.band = band
        self.weights = weights
        self.bad = bad
        self.arm_id = np.asarray(arm_id)
        self.fold_id = np.asarray(fold_id)
        self.exp = exp
        self.C, self.n_bins, self.n_diag = band.shape
        # prefix-sum scaffolding for O(1) window queries
        self._arm_changes = np.concatenate([[0], np.cumsum(self.arm_id[1:] != self.arm_id[:-1])])
        self._fold_changes = np.concatenate([[0], np.cumsum(self.fold_id[1:] != self.fold_id[:-1])])
        self._bad_cum = np.concatenate([np.zeros((self.C, 1)), np.cumsum(bad.astype(np.float64), axis=1)], axis=1)

    # -- window reconstruction (matches HiCDataset's get_slice output) ------- #
    def get_window(self, start_bin: int, n: int):
        """
        Reconstruct one training window. Returns ``(hic, weight, exp)`` ready for
        :func:`manta_hic.ops.hic_ops.create_expected_matrix` (add a leading batch dim per the caller):

        - ``hic``    ``[C, n, n]`` raw counts,
        - ``weight`` ``[C, n]`` per-bin weights,
        - ``exp``    ``[C, n_diag]`` per-distance expected for this window's arm.
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        hic = square_from_band(self.band, start_bin, n)  # also bounds-checks the window
        weight = self.weights[:, start_bin : start_bin + n]
        arm = int(self.arm_id[start_bin])
        if self.exp.ndim == 2:
            exp = self.exp  # single global arm
        elif arm < 0:
            raise ValueError(f"start_bin {start_bin} is in an excluded region (arm_id=-1); no per-arm expected")
        elif not np.all(self.arm_id[start_bin : start_bin + n] == arm):
            # eligible_starts guarantees single-arm windows; a hand-picked window crossing an arm boundary
            # would give a per-arm exp that is wrong for the far side, so reject it rather than mislead.
            raise ValueError(f"window [{start_bin}, {start_bin + n}) crosses an arm boundary; exp is ambiguous")
        else:
            exp = self.exp[:, arm]
        return hic, weight, exp

    # -- sampling-time tile selection (the inclusion criteria, as prefix sums) #
    def eligible_starts(self, n: int, *, min_fraction: float = 0.1, fold: int | None = None) -> np.ndarray:
        """
        Start bins whose window ``[a, a+n)`` passes every inclusion criterion (see docs/HIC_STORAGE.md):

        - lies within one arm (no centromere/arm/chrom-end crossing) and the arm is not excluded;
        - (if ``fold`` given) lies within that single Borzoi fold;
        - RMS over channels of the per-channel windowed mean ``bad`` fraction is ``< min_fraction``.
        """
        if n > self.n_bins:
            return np.empty(0, dtype=np.int64)
        a = np.arange(self.n_bins - n + 1)
        # arm: constant over window (zero change-points strictly inside) and not excluded
        arm_ok = (self._arm_changes[a + n - 1] - self._arm_changes[a] == 0) & (self.arm_id[a] != -1)
        # fold: constant over window and equal to the requested fold
        if fold is None:
            fold_ok = np.ones_like(a, dtype=bool)
        else:
            fold_ok = (self._fold_changes[a + n - 1] - self._fold_changes[a] == 0) & (self.fold_id[a] == fold)
        # bad: RMS over channels of windowed-mean bad fraction < min_fraction
        win_mean = (self._bad_cum[:, a + n] - self._bad_cum[:, a]) / n  # [C, S]
        bad_ok = np.sqrt((win_mean**2).mean(axis=0)) < min_fraction
        return a[arm_ok & fold_ok & bad_ok]

    @classmethod
    def from_dense(cls, matrix, weights, bad, arm_id, fold_id, exp_per_arm, n_diag):
        """Build a store from a dense ``[C, L, L]`` matrix (for tests / small data)."""
        return cls(band_from_dense(np.asarray(matrix), n_diag), weights, bad, arm_id, fold_id, exp_per_arm)

    @classmethod
    def from_hdf5(cls, path, chrom):
        """Load one chromosome from a banded HDF5 written by ``io.banded_write.coolers_to_banded``."""
        import h5py

        with h5py.File(path, "r") as f:
            g = f[chrom]
            store = cls(g["band"][:], g["weights"][:], g["bad"][:], g["arm_id"][:], g["fold_id"][:], f["exp"][:])
            store.shortnames = [s.decode() if isinstance(s, bytes) else s for s in f["provenance/shortnames"][:]]
            store.genome = f.attrs["genome"]
            store.resolution = int(f.attrs["resolution"])
        return store
