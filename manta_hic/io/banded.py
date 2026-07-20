"""
Banded ("turned") Hi-C target storage.

Store the first ``n_diag`` diagonals of each chromosome's contact map densely and once:

    band[c, x, d] = M_c[x, x + d]      for d in 0..n_diag-1

Any ``n x n`` window is reconstructed on the fly by symmetry, so display never fails and tile accept/reject is a
sampling-time policy (per-bin vectors + prefix sums) rather than a baked-in build decision.

**Ownership.** :class:`BandedHicFile` is the single owner and primary source of everything read out of one
``.bhic.h5``: the genome-wide per-bin metadata (``weights`` / ``bad`` / ``arm_id`` / ``fold_id``, concatenated
across chromosomes into flat length-``N`` arrays), the per-chromosome ``band`` (kept lazily on disk -- a 256 bp
chromosome band is gigabytes and must never be materialized), the per-arm expected, and the eligibility logic.
Two coordinate systems sit on top of it:

- **humans / inference / plotting** use ``(chrom, start_bp)`` -- :meth:`get_window`, :meth:`is_eligible`;
- **sampling / training** use a single global bin index ``pos`` in ``[0, total_bins)`` -- :meth:`eligible_mask`
  (``np.nonzero`` it for the pool), :meth:`window_at`, with :meth:`pos_to_coord` / :meth:`coord_to_pos` bridging.

There is no per-chromosome store object: the file *is* the store. Because ``arm_id`` is globally unique per arm,
a window that would straddle a chromosome (or arm) boundary spans two arm ids and is rejected automatically, so
the flat genome-wide axis needs no special-casing at the seams.

The build side (reading real coolers + ``cooltools`` expected) lives in ``io/banded_write.py``.
"""

from functools import lru_cache

import h5py
import hdf5plugin  # noqa: F401  -- registers the Blosc filter so bands written with it are readable
import numpy as np

# --------------------------------------------------------------------------- #
# Band <-> square conversions                                                  #
# --------------------------------------------------------------------------- #


def band_from_dense(matrix: np.ndarray, n_diag: int) -> np.ndarray:
    """
    Encode a dense symmetric contact matrix into the banded layout, keeping the first ``n_diag`` diagonals.

    Returns ``[..., L, n_diag]`` where ``band[..., x, d] = matrix[..., x, x + d]`` (0 where ``x + d >= L``).
    """
    L = matrix.shape[-1]
    band = np.zeros(matrix.shape[:-2] + (L, n_diag), dtype=matrix.dtype)
    for d in range(min(n_diag, L)):
        rows = np.arange(L - d)
        band[..., : L - d, d] = matrix[..., rows, rows + d]
    return band


_GATHER_INDEX: dict[tuple[int, int], np.ndarray] = {}


def _gather_index(n: int, width: int) -> np.ndarray:
    """Cached flat gather index ``min(i,j)*width + |i-j|`` for reconstructing an ``n x n`` square from a block of
    last-dim ``width`` (cached because rebuilding the two ``n x n`` index grids per window read is costly)."""
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

    ``block`` is ``[..., n, n_diag_read]`` (rows are bins ``a..a+n-1``, columns are diagonals ``0..``), i.e. the
    slice ``band[..., a:a+n, :]``; ``M[i, j] = block[min(i, j), |i - j|]`` as a single cached ``take``.
    """
    block = np.asarray(block)  # materialize if an h5py dataset slice slipped through
    width = block.shape[-1]
    lead = block.shape[:-2]
    flat = block.reshape(lead + (block.shape[-2] * width,))
    return flat[..., _gather_index(n, width)].reshape(lead + (n, n))  # [..., n, n]


def square_from_band(band, start_bin: int, n: int) -> np.ndarray:
    """
    Reconstruct the ``n x n`` symmetric submatrix ``M[a:a+n, a:a+n]`` from a band ``[..., n_bins, n_diag]``.

    ``band`` may be an in-RAM ndarray or a lazy ``h5py`` dataset; only the block ``[..., a:a+n, :n]`` is read.
    Requires ``n_diag >= n`` and the window to lie within the stored bins.
    """
    a = start_bin
    n_bins, n_diag = band.shape[-2], band.shape[-1]
    if n > n_diag:
        raise ValueError(f"window {n} exceeds stored diagonals {n_diag}")
    if a < 0 or a + n > n_bins:
        raise ValueError(f"window [{a}, {a + n}) outside stored bins [0, {n_bins})")
    return square_from_block(band[..., a : a + n, :n], n)


def _decode(x):
    """h5py string datasets come back as bytes under some settings; normalize to str."""
    return x.decode() if isinstance(x, bytes) else str(x)


# --------------------------------------------------------------------------- #
# BandedHicFile: the single owner (genome-wide flat arrays + global-pos API)   #
# --------------------------------------------------------------------------- #


class BandedHicFile:
    """
    Read side of one ``.bhic.h5`` banded file -- the single owner of its data (see the module docstring).

    Build from a path (production) or, for tests, from in-RAM arrays via :meth:`from_arrays`.

    Attributes
    ----------
    chroms : list[str]
        Chromosomes carrying a band, in canonical genome order.
    total_bins : int
        ``N`` -- number of bins genome-wide (the length of the flat metadata arrays / the global ``pos`` axis).
    weights, bad : np.ndarray
        ``[C, N]`` per-bin balancing weights (bad bins zeroed) and boolean bad mask, concatenated across chroms.
    arm_id, fold_id : np.ndarray
        ``[N]`` global arm id (``-1`` excluded; unique per arm so it also encodes chrom boundaries) and Borzoi
        fold id (``-1`` none).
    exp : np.ndarray
        ``[C, n_arms, n_diag]`` per-arm per-distance expected (or ``[C, n_diag]`` for a single global arm).
    genome, resolution, n_diag, n_channels, shortnames, chrom_lengths, arms : file-level metadata.
    """

    # -- construction -------------------------------------------------------- #
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
            exp = f["exp"][:]  # [C, n_arms, n_diag] -- small, held in RAM
            # Real chromosomes carry a band group; iterate in chroms/name (canonical) order, not lexicographic.
            chroms = [c for c in self.chrom_lengths if c in f and isinstance(f[c], h5py.Group) and "band" in f[c]]
            bands, weights, bad, arm_id, fold_id = [], [], [], [], []
            for c in chroms:
                g = f[c]
                bands.append(g["band"])  # lazy on-disk h5py dataset (kept; self._f stays open)
                weights.append(g["weights"][:])
                bad.append(g["bad"][:])
                arm_id.append(g["arm_id"][:])
                fold_id.append(g["fold_id"][:])
            self._assemble(chroms, bands, weights, bad, arm_id, fold_id, exp)
        except BaseException:
            f.close()
            raise

    @classmethod
    def from_arrays(
        cls,
        chroms,
        bands,
        weights,
        bad,
        arm_id,
        fold_id,
        exp,
        *,
        resolution=1,
        genome="test",
        n_diag=None,
        shortnames=None,
    ):
        """Build an in-memory file from per-chromosome arrays (tests / small data). ``bands[i]`` is
        ``[C, n_bins, n_diag]``; ``weights[i]``/``bad[i]`` ``[C, n_bins]``; ``arm_id[i]``/``fold_id[i]`` ``[n_bins]``
        (arm ids must be globally unique across chroms). ``exp`` is ``[C, n_arms, n_diag]`` or ``[C, n_diag]``."""
        self = cls.__new__(cls)
        self._f = None
        self.path = "<in-memory>"
        self.genome, self.resolution = genome, int(resolution)
        self.n_diag = int(n_diag if n_diag is not None else bands[0].shape[-1])
        self.n_channels = bands[0].shape[0]
        self.shortnames = list(shortnames) if shortnames is not None else [f"ch{i}" for i in range(self.n_channels)]
        self.uris = list(self.shortnames)
        self.group_name = None
        self.chrom_lengths = {c: int(np.asarray(w).shape[1]) * self.resolution for c, w in zip(chroms, weights)}
        self.arms = None
        self._assemble(list(chroms), list(bands), weights, bad, arm_id, fold_id, exp)
        return self

    def _assemble(self, chroms, bands, weights, bad, arm_id, fold_id, exp):
        """Concatenate the per-chrom metadata into the genome-wide flat arrays this file owns + the offset table."""
        self.chroms = list(chroms)
        self._band = {c: b for c, b in zip(chroms, bands)}  # h5py dataset (path) or ndarray (in-memory)
        self.weights = np.concatenate([np.asarray(w) for w in weights], axis=1)  # [C, N]
        self.bad = np.concatenate([np.asarray(b) for b in bad], axis=1)  # [C, N]
        self.arm_id = np.concatenate([np.asarray(a) for a in arm_id])  # [N]
        self.fold_id = np.concatenate([np.asarray(fo) for fo in fold_id])  # [N]
        self.exp = exp
        self.C = self.weights.shape[0]
        nbins = [int(np.asarray(w).shape[1]) for w in weights]
        self._offsets = np.concatenate([[0], np.cumsum(nbins)]).astype(np.int64)  # len nchrom+1
        self.chrom_start = {c: int(self._offsets[i]) for i, c in enumerate(chroms)}
        self.chrom_nbins = {c: nbins[i] for i, c in enumerate(chroms)}
        self.total_bins = int(self._offsets[-1])
        # prefix-sum scaffolding for the vectorized (genome-wide, single-pass) eligibility computation
        self._arm_changes = np.concatenate([[0], np.cumsum(self.arm_id[1:] != self.arm_id[:-1])])
        self._bad_cum = np.concatenate([np.zeros((self.C, 1)), np.cumsum(self.bad.astype(np.float64), axis=1)], axis=1)
        # per-instance memo of the eligibility masks (a handful per file, all actively reused; freed with the file)
        self._eligible_mask = lru_cache(maxsize=None)(self._compute_eligible_mask)

    # -- coordinate mapping: humans use (chrom, start_bp); sampling uses global pos --------------------------- #
    def bp_to_bin(self, start_bp: int) -> int:
        if start_bp % self.resolution:
            raise ValueError(f"start_bp {start_bp} is not a multiple of resolution {self.resolution}")
        return start_bp // self.resolution

    def coord_to_pos(self, chrom: str, start_bp: int) -> int:
        """``(chrom, start_bp)`` -> global bin index ``pos``."""
        if chrom not in self.chrom_start:
            raise KeyError(f"{chrom!r} has no band in {self.path} (chroms: {self.chroms})")
        return self.chrom_start[chrom] + self.bp_to_bin(start_bp)

    def pos_to_coord(self, pos: int) -> tuple[str, int]:
        """Global bin index ``pos`` -> ``(chrom, start_bp)``."""
        ci = int(np.searchsorted(self._offsets, pos, side="right") - 1)
        if ci < 0 or ci >= len(self.chroms):
            raise IndexError(f"pos {pos} outside [0, {self.total_bins})")
        return self.chroms[ci], int(pos - self._offsets[ci]) * self.resolution

    # -- eligibility: the single source of truth (genome-wide, one vectorized pass, lru-cached) -------------- #
    def eligible_mask(
        self, n: int, *, max_bad_fraction: float = 0.1, fold=None, overlap_threshold: float = 0.9
    ) -> np.ndarray:
        """
        Boolean mask over global start positions (length ``total_bins - n + 1``): ``mask[pos]`` is True iff the
        window ``[pos, pos+n)`` passes every inclusion criterion:

        - lies within one arm (constant ``arm_id``, not excluded) -- which, since arm ids are globally unique,
          also forbids crossing a chromosome boundary;
        - RMS over channels of the windowed mean ``bad`` fraction is ``< max_bad_fraction``;
        - (if ``fold`` given) at least ``overlap_threshold`` of its bins are in the fold set ``fold``.

        The fold rule tolerates fold boundaries (Borzoi's short randomly-offset snippets): a window straddling
        several *train* folds is all-train and kept; train/val boundaries are dropped unless one side clears the
        threshold. Pass ``fold=train_folds`` for the train pool, ``fold={val}`` for validation. For a flat int
        array of the eligible positions themselves, ``np.nonzero(eligible_mask(...))[0]``. lru-cached per file.
        """
        if fold is not None:  # make it hashable for the cache (a frozenset of ints)
            fold = frozenset({int(fold)} if np.isscalar(fold) else (int(x) for x in fold))
        return self._eligible_mask(int(n), float(max_bad_fraction), fold, float(overlap_threshold))

    def eligible_positions(
        self, n: int, *, max_bad_fraction: float = 0.1, fold=None, overlap_threshold: float = 0.9
    ) -> np.ndarray:
        """Flat global start positions of every eligible window -- ``np.nonzero(eligible_mask(...))[0]``. This is
        the training-side pool (one global bin index per window); see :meth:`eligible_mask` for the criteria."""
        return np.nonzero(
            self.eligible_mask(n, max_bad_fraction=max_bad_fraction, fold=fold, overlap_threshold=overlap_threshold)
        )[0]

    def _compute_eligible_mask(self, n, max_bad_fraction, fold, overlap_threshold):
        N = self.total_bins
        if n <= 0 or n > N:
            return np.zeros(0, dtype=bool)
        a = np.arange(N - n + 1)
        arm_ok = (self._arm_changes[a + n - 1] - self._arm_changes[a] == 0) & (self.arm_id[a] != -1)
        win_mean = (self._bad_cum[:, a + n] - self._bad_cum[:, a]) / n  # [C, S] windowed mean bad fraction
        bad_ok = np.sqrt((win_mean**2).mean(axis=0)) < max_bad_fraction
        if fold is None:
            fold_ok = np.ones_like(a, dtype=bool)
        else:
            in_set = np.isin(self.fold_id, np.fromiter(fold, dtype=self.fold_id.dtype, count=len(fold)))
            cum = np.concatenate([[0], np.cumsum(in_set.astype(np.int64))])
            fold_ok = (cum[a + n] - cum[a]) / n >= overlap_threshold
        return arm_ok & bad_ok & fold_ok

    # -- window reconstruction ----------------------------------------------- #
    def window_at(self, pos: int, n: int):
        """Reconstruct the window at global ``pos`` -> ``(hic[C,n,n], weight[C,n], exp[C,n_diag])`` (add a batch
        dim per the caller, for :func:`manta_hic.ops.hic_ops.create_expected_matrix`)."""
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        ci = int(np.searchsorted(self._offsets, pos, side="right") - 1)
        if ci < 0 or ci >= len(self.chroms):
            raise IndexError(f"pos {pos} outside [0, {self.total_bins})")
        chrom = self.chroms[ci]
        local = int(pos - self._offsets[ci])
        if local + n > self.chrom_nbins[chrom]:
            raise ValueError(f"window at pos {pos} (chrom {chrom} local {local}, n={n}) runs off the chromosome")
        hic = square_from_band(self._band[chrom], local, n)
        weight = self.weights[:, pos : pos + n]
        arm = int(self.arm_id[pos])
        if self.exp.ndim == 2:
            exp = self.exp  # single global arm
        elif arm < 0:
            raise ValueError(f"pos {pos} is in an excluded region (arm_id=-1); no per-arm expected")
        elif not np.all(self.arm_id[pos : pos + n] == arm):
            raise ValueError(f"window at pos {pos} crosses an arm boundary; exp is ambiguous")
        else:
            exp = self.exp[:, arm]
        return hic, weight, exp

    # -- human/coordinate wrappers (inference, plotting) --------------------- #
    def get_window(self, chrom: str, start_bp: int, n_bins: int):
        """Observed target window at ``(chrom, start_bp)`` as ``(hic, weight, exp)`` -- no model needed."""
        return self.window_at(self.coord_to_pos(chrom, start_bp), n_bins)

    def is_eligible(
        self,
        chrom: str,
        start_bp: int,
        n_bins: int,
        *,
        max_bad_fraction: float = 0.1,
        fold=None,
        overlap_threshold: float = 0.9,
    ) -> bool:
        """O(1) eligibility of the exact window at ``(chrom, start_bp)`` -- the inference fast path."""
        if chrom not in self.chrom_start:
            return False
        pos = self.coord_to_pos(chrom, start_bp)
        mask = self.eligible_mask(
            n_bins, max_bad_fraction=max_bad_fraction, fold=fold, overlap_threshold=overlap_threshold
        )
        return bool(0 <= pos < len(mask) and mask[pos])

    def eligible_starts(
        self, chrom: str, n_bins: int, *, max_bad_fraction: float = 0.1, fold=None, overlap_threshold: float = 0.9
    ) -> np.ndarray:
        """Eligible *local* start bins within ``chrom`` (human convenience; training samples over global
        positions from :meth:`eligible_positions` instead)."""
        pos = self.eligible_positions(
            n_bins, max_bad_fraction=max_bad_fraction, fold=fold, overlap_threshold=overlap_threshold
        )
        lo = self.chrom_start[chrom]
        hi = lo + self.chrom_nbins[chrom]
        return (pos[(pos >= lo) & (pos < hi)] - lo).astype(np.int64)

    def present_folds(self) -> list[int]:
        """Sorted distinct Borzoi fold ids present genome-wide (excludes the ``-1`` sentinel)."""
        return sorted(set(int(x) for x in np.unique(self.fold_id)) - {-1})

    # -- lifecycle ----------------------------------------------------------- #
    def close(self):
        if self._f is not None:
            self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
