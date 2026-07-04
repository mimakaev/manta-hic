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

Two entry points:

- :class:`BandedHicStore` -- one chromosome. Reconstruct/eligibility primitive. ``band`` may be an in-RAM
  ndarray (tests / small data) **or** a lazy ``h5py`` dataset; windows read only the ``[C, n, :n]`` block, so
  a multi-GB 256 bp chromosome band is never materialized.
- :class:`BandedHicFile` -- one whole ``.bhic.h5`` (many chromosomes + genome metadata), the production read
  side. It serves the observed target and everything needed to plot it autonomously (no model, no fetcher);
  training samples over it via :class:`manta_hic.nn.dataset.HiCDataset`, prediction via
  :class:`manta_hic.nn.inference.MantaInference`.

The build side (reading real coolers + ``cooltools`` expected) lives in ``io/banded_write.py``.
"""

import h5py
import hdf5plugin  # noqa: F401  -- registers the Blosc filter so bands written with it are readable
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


_GATHER_INDEX: dict[tuple[int, int], np.ndarray] = {}


def _gather_index(n: int, width: int) -> np.ndarray:
    """Cached flat gather index ``min(i,j)*width + |i-j|`` for reconstructing an ``n x n`` square from a block
    of last-dim ``width``. Cached because it is constant for a given ``(n, width)`` and rebuilding the two
    ``n x n`` index grids every window read is a large fraction of the reconstruction cost."""
    key = (n, width)
    idx = _GATHER_INDEX.get(key)
    if idx is None:
        ii, jj = np.indices((n, n))
        idx = (np.minimum(ii, jj) * width + np.abs(ii - jj)).ravel()
        _GATHER_INDEX[key] = idx
    return idx


def square_from_block(block: np.ndarray, n: int) -> np.ndarray:
    """
    Reconstruct an ``n x n`` symmetric submatrix from a band *block* already offset to the window start.

    ``block`` is ``[..., n, n_diag_read]`` (rows are bins ``a..a+n-1``, columns are diagonals ``0..``), i.e.
    the slice ``band[..., a:a+n, :]``. The reconstruction is ``M[i, j] = block[min(i, j), |i - j|]``, done as a
    single cached ``take`` over the flattened last two dims (fast; no per-call index-grid construction).
    """
    block = np.asarray(block)  # materialize if an h5py dataset slice slipped through
    width = block.shape[-1]
    lead = block.shape[:-2]
    flat = block.reshape(lead + (block.shape[-2] * width,))
    return flat[..., _gather_index(n, width)].reshape(lead + (n, n))  # [..., n, n]


def square_from_band(band, start_bin: int, n: int) -> np.ndarray:
    """
    Reconstruct the ``n x n`` symmetric submatrix ``M[a:a+n, a:a+n]`` from a band ``[..., n_bins, n_diag]``.

    ``band`` may be an in-RAM ndarray or a lazy ``h5py`` dataset; only the block ``[..., a:a+n, :n]`` is read
    (so a giant on-disk chromosome band is never materialized). Requires ``n_diag >= n`` and the window to
    lie within the stored bins.

    Parameters
    ----------
    band : np.ndarray | h5py.Dataset
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
    return square_from_block(band[..., a : a + n, :n], n)


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
        # prefix-sum scaffolding for the vectorized eligibility computation
        self._arm_changes = np.concatenate([[0], np.cumsum(self.arm_id[1:] != self.arm_id[:-1])])
        self._fold_changes = np.concatenate([[0], np.cumsum(self.fold_id[1:] != self.fold_id[:-1])])
        self._bad_cum = np.concatenate([np.zeros((self.C, 1)), np.cumsum(bad.astype(np.float64), axis=1)], axis=1)
        self._mask_cache: dict = {}  # (n, min_fraction, fold_key) -> bool eligibility mask over start bins

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

    # -- sampling-time tile selection (the inclusion criteria) --------------- #
    @staticmethod
    def _fold_key(fold):
        """Hashable cache key for the ``fold`` argument (None / int / iterable of ints)."""
        if fold is None:
            return None
        return int(fold) if np.isscalar(fold) else frozenset(int(x) for x in fold)

    def eligible_mask(self, n: int, *, min_fraction: float = 0.1, fold=None) -> np.ndarray:
        """
        Boolean mask over start bins (length ``n_bins - n + 1``): ``mask[a]`` is True iff the window
        ``[a, a+n)`` passes every inclusion criterion (see docs/HIC_STORAGE.md):

        - lies within one arm (no centromere/arm/chrom-end crossing) and the arm is not excluded;
        - (if ``fold`` given) lies within a single Borzoi fold that is in ``fold``;
        - RMS over channels of the per-channel windowed mean ``bad`` fraction is ``< min_fraction``.

        ``fold`` may be ``None`` (any fold), a single int, or an iterable of ints (single-fold **and** in the
        set -- the set form lets a training dataset collect all train folds in one pass). The mask is computed
        once and **memoized** per ``(n, min_fraction, fold)``, so :meth:`eligible_starts` and
        :meth:`is_eligible` are then just a select / index over the precomputed array -- no recomputation and
        a single source of truth for the criteria.
        """
        key = (int(n), float(min_fraction), self._fold_key(fold))
        mask = self._mask_cache.get(key)
        if mask is None:
            mask = self._compute_mask(n, min_fraction, fold)
            self._mask_cache[key] = mask
        return mask

    def _compute_mask(self, n, min_fraction, fold):
        if n <= 0 or n > self.n_bins:
            return np.zeros(0, dtype=bool)
        a = np.arange(self.n_bins - n + 1)
        # arm: constant over window (zero change-points strictly inside) and not excluded
        arm_ok = (self._arm_changes[a + n - 1] - self._arm_changes[a] == 0) & (self.arm_id[a] != -1)
        # fold: single-fold window whose fold is the requested one / in the requested set
        if fold is None:
            fold_ok = np.ones_like(a, dtype=bool)
        else:
            single_fold = self._fold_changes[a + n - 1] - self._fold_changes[a] == 0
            if np.isscalar(fold):
                in_set = self.fold_id[a] == fold
            else:
                in_set = np.isin(self.fold_id[a], np.asarray(list(fold), dtype=self.fold_id.dtype))
            fold_ok = single_fold & in_set
        # bad: RMS over channels of windowed-mean bad fraction < min_fraction
        win_mean = (self._bad_cum[:, a + n] - self._bad_cum[:, a]) / n  # [C, S]
        bad_ok = np.sqrt((win_mean**2).mean(axis=0)) < min_fraction
        return arm_ok & fold_ok & bad_ok

    def eligible_starts(self, n: int, *, min_fraction: float = 0.1, fold=None) -> np.ndarray:
        """Start bins passing every inclusion criterion -- ``nonzero`` of :meth:`eligible_mask`."""
        return np.nonzero(self.eligible_mask(n, min_fraction=min_fraction, fold=fold))[0].astype(np.int64)

    def is_eligible(self, start_bin: int, n: int, *, min_fraction: float = 0.1, fold=None) -> bool:
        """
        Whether the exact window ``[start_bin, start_bin+n)`` is eligible -- a plain lookup into the
        (memoized) :meth:`eligible_mask`. Because a candidate window starts at every bin, inference/mutation
        code can ask this directly, with no search over a tile list.
        """
        mask = self.eligible_mask(n, min_fraction=min_fraction, fold=fold)
        return bool(0 <= start_bin < len(mask) and mask[start_bin])

    @classmethod
    def from_dense(cls, matrix, weights, bad, arm_id, fold_id, exp_per_arm, n_diag):
        """Build a store from a dense ``[C, L, L]`` matrix (for tests / small data)."""
        return cls(band_from_dense(np.asarray(matrix), n_diag), weights, bad, arm_id, fold_id, exp_per_arm)

    @classmethod
    def from_hdf5(cls, path, chrom):
        """Load one chromosome (band fully into RAM) from a banded HDF5. Prefer :class:`BandedHicFile` for
        production reads -- it keeps the (huge) band on disk and slices windows lazily."""
        with h5py.File(path, "r") as f:
            g = f[chrom]
            store = cls(g["band"][:], g["weights"][:], g["bad"][:], g["arm_id"][:], g["fold_id"][:], f["exp"][:])
            store.shortnames = [s.decode() if isinstance(s, bytes) else s for s in f["provenance/shortnames"][:]]
            store.genome = f.attrs["genome"]
            store.resolution = int(f.attrs["resolution"])
        return store


def _decode(x):
    """h5py string datasets come back as bytes under some settings; normalize to str."""
    return x.decode() if isinstance(x, bytes) else str(x)


# --------------------------------------------------------------------------- #
# Whole-file wrapper: many chromosomes + genome metadata, one .bhic.h5         #
# --------------------------------------------------------------------------- #


class BandedHicFile:
    """
    Read side of one ``.bhic.h5`` banded file (:func:`io.banded_write.coolers_to_banded`).

    Keeps the file handle open and builds a :class:`BandedHicStore` per chromosome **lazily**, with the
    per-bin metadata (weights/bad/arm_id/fold_id, a few MB) in RAM but the ``band`` itself left on disk and
    sliced one window at a time -- a single chromosome band is gigabytes at 256 bp and must never be
    materialized whole.

    It is deliberately model-free and fetcher-free: it serves the Hi-C *target* (and all the metadata needed
    to plot it autonomously). Training samples through :class:`manta_hic.nn.dataset.HiCDataset`; prediction
    goes through :class:`manta_hic.nn.inference.MantaInference`; both are thin layers over this file.

    Parameters
    ----------
    path : str | os.PathLike
        Path to the banded HDF5.

    Attributes
    ----------
    genome, resolution, n_diag, n_channels, group_name : file-level metadata.
    shortnames, uris : per-channel provenance (channel ``c`` is ``shortnames[c]``).
    chroms : list[str]
        Chromosomes that actually have a band group (the eligible autosomes/chrX; e.g. chrM/chrY are absent).
    chrom_lengths : dict[str, int], arms : dict of parallel arrays (name/chrom/start/end), exp : ndarray
        ``[C, n_arms, n_diag]`` per-arm expected (``arm_id`` bins index its second axis).
    """

    def __init__(self, path):
        self.path = str(path)
        f = self._f = h5py.File(self.path, "r")
        try:  # anything below can raise on a partial/malformed file; don't leak the open handle
            if not f.attrs.get("complete", False):
                raise ValueError(f"{self.path} is not marked complete (partial/in-progress write); refusing to open")
            self.genome = _decode(f.attrs["genome"])
            self.resolution = int(f.attrs["resolution"])
            self.n_diag = int(f.attrs["n_diag"])
            self.n_channels = int(f.attrs["n_channels"])
            self.group_name = _decode(f.attrs["group_name"]) if "group_name" in f.attrs else None
            self.shortnames = [_decode(s) for s in f["provenance/shortnames"][:]]
            self.uris = [_decode(s) for s in f["provenance/uris"][:]]
            self.chrom_lengths = {_decode(n): int(ln) for n, ln in zip(f["chroms/name"][:], f["chroms/length"][:])}
            self.arms = {
                "name": [_decode(s) for s in f["arms/name"][:]],
                "chrom": [_decode(s) for s in f["arms/chrom"][:]],
                "start": f["arms/start"][:],
                "end": f["arms/end"][:],
            }
            self.exp = f["exp"][:]  # [C, n_arms, n_diag] -- small, held in RAM
        except BaseException:
            f.close()
            raise
        # Real chromosomes are those carrying a band (chroms/name lists *all* cooler chromnames). Iterate in
        # chroms/name order (canonical genome order) rather than f.keys() (lexicographic: chr1, chr10, chr2, ...).
        self.chroms = [c for c in self.chrom_lengths if c in f and isinstance(f[c], h5py.Group) and "band" in f[c]]
        self._stores: dict[str, BandedHicStore] = {}

    # -- store access -------------------------------------------------------- #
    def store(self, chrom: str) -> BandedHicStore:
        """The (cached) :class:`BandedHicStore` for ``chrom``; its ``band`` is the on-disk h5py dataset."""
        if chrom not in self._stores:
            if chrom not in self.chroms:
                raise KeyError(f"{chrom!r} has no band group in {self.path} (chroms: {self.chroms})")
            g = self._f[chrom]
            self._stores[chrom] = BandedHicStore(
                g["band"], g["weights"][:], g["bad"][:], g["arm_id"][:], g["fold_id"][:], self.exp
            )
        return self._stores[chrom]

    # -- coordinate helpers -------------------------------------------------- #
    def bp_to_bin(self, start_bp: int) -> int:
        if start_bp % self.resolution:
            raise ValueError(f"start_bp {start_bp} is not a multiple of resolution {self.resolution}")
        return start_bp // self.resolution

    # -- autonomous target read (plotting / eval) ---------------------------- #
    def get_window(self, chrom: str, start_bp: int, n_bins: int):
        """Reconstruct one target window as ``(hic[C,n,n], weight[C,n], exp[C,n_diag])`` -- no model needed."""
        return self.store(chrom).get_window(self.bp_to_bin(start_bp), n_bins)

    def is_eligible(self, chrom: str, start_bp: int, n_bins: int, *, min_fraction: float = 0.1, fold=None) -> bool:
        """O(1) eligibility of the exact window at ``(chrom, start_bp)`` -- the inference fast path."""
        if chrom not in self.chroms:
            return False
        return self.store(chrom).is_eligible(self.bp_to_bin(start_bp), n_bins, min_fraction=min_fraction, fold=fold)

    def eligible_starts(self, chrom: str, n_bins: int, *, min_fraction: float = 0.1, fold=None) -> np.ndarray:
        """Eligible start *bins* for ``chrom`` (see :meth:`BandedHicStore.eligible_starts`)."""
        return self.store(chrom).eligible_starts(n_bins, min_fraction=min_fraction, fold=fold)

    def present_folds(self) -> list[int]:
        """Sorted distinct Borzoi fold ids present across all chromosomes (excludes the ``-1`` sentinel)."""
        seen: set[int] = set()
        for chrom in self.chroms:
            seen.update(int(x) for x in np.unique(self.store(chrom).fold_id))
        return sorted(seen - {-1})

    def close(self):
        self._f.close()
        self._stores.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
