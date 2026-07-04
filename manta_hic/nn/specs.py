"""
L0 of the spec-based inference stack (see docs/INFERENCE_SPEC_PLAN.md). **Pure data + geometry** -- no
torch, no h5py, no GPU, so it is fully unit-testable on CPU.

A *spec* concretely and completely describes ONE Manta inference (one output map): which cached MicroZoi
run, orientation, tile shift, and -- if mutated -- exactly which MicroZoi tiles ("bricks") to recompute.
Every knob is *resolved* (no ``"auto"``), so the whole downstream pipeline is a deterministic function of
the spec. The payoff: a wild-type spec and a mutant spec that share a :class:`Background` produce a
byte-identical WT background and recompute, so ``mutant - wildtype`` is clean **without the engine knowing
they are a pair**.

The recompute geometry is fully worked out **here**, not on the GPU: from the mutation superset we derive
the coverage each mutation needs (``auto_patch_windows``) and lay it out as concrete MicroZoi tiles
(``tile_pattern``, with the greedy "keep tiling if the current tile runs into the next window" merge). L1
then just **rigidly shifts** that fixed brick pattern by ``tile_offset_bins`` and runs it -- no tiling or
merging logic downstream. The shift is a pure translation yet still augments: each mutation lands at a
different phase within its tile (a different receptive field), which is the point of run-averaging.

Two objects:

- :class:`Background` -- everything that fixes the cache read + recompute geometry, incl. the brick
  ``tiles``. Hashable; the cache-dedup / "compatibility" key.
- :class:`Spec` -- a Background plus this sample's ``mutations`` (possibly empty), output ``channels``
  subset, and free-form ``tags`` (labels for ``group_by``; excluded from the compute identity).

Mutations are independent of specs: build them (``replace``/``invert``/``shuffle[k]``/``inactivate``), size
the pattern from their spans, and -- for the random ops -- :func:`manta_hic.ops.seq_ops.freeze_mutations`
resolves them to concrete ``replace`` tuples so a spec stays deterministic.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from manta_hic.ops.seq_ops import freeze_mutations

BIN_BP = 256  # MicroZoi bin size in bp (mirrors manta.BIN_BP); the grid the pattern snaps to.
MICROZOI_RF_BP = 2**19 + 2**18  # MicroZoi receptive field (mirrors manta.MICROZOI_RECEPTIVE_FIELD).
RF_BINS = MICROZOI_RF_BP // BIN_BP  # 3072 output bins fed per tile.
DEFAULT_CROP_MHA_BINS = 768  # bins cropped from each tile edge; one tile then emits RF_BINS - 2*crop bins.

# Two independent knobs, both in bp (defaults expressed as a bin count for readability):
#   soft_causality -- the minimum recompute margin a mutation needs. A bin this close to a mutation must be
#                     recomputed from the mutated sequence, else it keeps stale wild-type activations. How
#                     far a local sequence change propagates through MicroZoi -- "not much" in practice, and
#                     less than the ~640-1024-bin crop_mha context. PLACEHOLDER, calibrate empirically.
#   max_shift      -- the range of the per-background rigid tile shift (augmentation / a clean randomness
#                     source for run-averaging, unlike a sub-bp shift that could split a motif across bins).
# The pattern is built with tolerance = soft_causality + max_shift, so after any shift up to max_shift a
# mutation still sits >= soft_causality from the recomputed/cached seam. (Max's worked example: 256 + 256 bins.)
DEFAULT_SOFT_CAUSALITY_BP = 256 * BIN_BP
DEFAULT_MAX_SHIFT_BP = 256 * BIN_BP


# --------------------------------------------------------------------------- #
# Mutation geometry, coverage, and the concrete tile pattern                   #
# --------------------------------------------------------------------------- #


def mutation_span(m) -> tuple[int, int]:
    """``(start_bp, end_bp)`` a single length-preserving mutation tuple touches.

    ``("replace", pos, seq)`` -> ``(pos, pos+len(seq))``; ``("invert"/"shuffle[k]"/"inactivate", p1, p2)`` ->
    ``(p1, p2)``. Length-changing ops (``insert``/deletions) are rejected -- they break the cache bin grid.
    """
    op = m[0]
    if op == "replace":
        return m[1], m[1] + len(m[2])
    if op == "invert" or op == "inactivate" or op.startswith("shuffle"):
        return m[1], m[2]
    raise ValueError(f"unknown/unsupported mutation op {op!r} (length-changing ops are not allowed)")


def _snap(lo_bp, hi_bp) -> tuple[int, int]:
    """Snap a bp interval outward to the MicroZoi bin grid (floor lo, ceil hi)."""
    return int(np.floor(lo_bp / BIN_BP) * BIN_BP), int(np.ceil(hi_bp / BIN_BP) * BIN_BP)


def _union(windows) -> tuple[tuple[int, int], ...]:
    """Merge overlapping/touching ``[lo, hi)`` windows into sorted disjoint intervals."""
    ws = sorted(tuple(w) for w in windows)
    if not ws:
        return ()
    out = [list(ws[0])]
    for lo, hi in ws[1:]:
        if lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return tuple((lo, hi) for lo, hi in out)


def tile_size_bp(crop_mha_bins: int = DEFAULT_CROP_MHA_BINS) -> int:
    """Output bp one MicroZoi tile emits: ``(RF_BINS - 2*crop) * BIN_BP`` (crop bins are dropped each edge)."""
    n = RF_BINS - 2 * int(crop_mha_bins)
    if n <= 0:
        raise ValueError(f"crop_mha_bins {crop_mha_bins} leaves no output bins (RF is {RF_BINS} bins)")
    return n * BIN_BP


def auto_patch_windows(
    mutations,
    *,
    soft_causality_bp: int = DEFAULT_SOFT_CAUSALITY_BP,
    max_shift_bp: int = DEFAULT_MAX_SHIFT_BP,
) -> tuple[tuple[int, int], ...]:
    """The *coverage* each mutation needs: its span expanded by ``soft_causality_bp + max_shift_bp`` (so any
    shift up to ``max_shift`` leaves >= ``soft_causality`` margin), bin-snapped and unioned. This is the
    minimal region that must be recomputed; :func:`tile_pattern` lays actual MicroZoi tiles over it."""
    tol = soft_causality_bp + max_shift_bp
    return _union(_snap(lo - tol, hi + tol) for lo, hi in (mutation_span(m) for m in mutations))


def tile_pattern(coverage_windows, *, crop_mha_bins: int = DEFAULT_CROP_MHA_BINS) -> tuple[tuple[int, int], ...]:
    """Lay concrete MicroZoi tiles (bricks) over ``coverage_windows``. Each brick ``(lo, hi)`` emits one
    tile's worth of output (``tile_size_bp`` wide) and is recomputed from an RF-wide input around it.

    Greedy tiling: step tiles across a window; when the next window starts before the current tile run ends,
    keep tiling straight into it (absorbing the small gap for free) instead of starting a fresh, possibly
    redundant tile. So two ~1.1-tile windows a fraction of a tile apart cost 3 tiles, not 4. The bricks are
    the *offset-0* pattern; L1 rigidly shifts the whole set by ``tile_offset_bins``."""
    step = tile_size_bp(crop_mha_bins)
    bricks, pos = [], None
    for lo, hi in _union(coverage_windows):
        if pos is None or lo > pos:  # a real gap -> start a fresh tile run at the window; else keep tiling
            pos = lo
        while pos < hi:
            bricks.append((pos, pos + step))
            pos += step
    return tuple(bricks)


def validate_patches(
    mutations,
    tiles,
    *,
    soft_causality_bp: int = DEFAULT_SOFT_CAUSALITY_BP,
    max_shift_bp: int = DEFAULT_MAX_SHIFT_BP,
) -> None:
    """Enforce soft causality against the *tile* coverage: every mutation's span expanded by
    ``soft_causality_bp + max_shift_bp`` must lie inside the union of ``tiles``. The ``max_shift`` headroom
    keeps the guarantee after the rigid shift; ``soft_causality`` is the margin left at the worst shift. A
    violation means a recomputed/cached seam would sit within the causality margin of a mutation, leaving
    stale wild-type activations there. (Seams *between* adjacent tiles are fine -- both recompute the
    mutation, and MicroZoi's large crop means each tile still sees it in context.)"""
    tol = soft_causality_bp + max_shift_bp
    merged = _union(tiles)
    for m in mutations:
        lo, hi = mutation_span(m)
        need_lo, need_hi = lo - tol, hi + tol
        if not any(t_lo <= need_lo and need_hi <= t_hi for t_lo, t_hi in merged):
            raise ValueError(
                f"soft-causality violation: mutation {m[0]} at [{lo}, {hi}) needs bins [{need_lo}, {need_hi}) "
                f"recomputed (soft_causality {soft_causality_bp} + max_shift {max_shift_bp} bp), but that is "
                f"not covered by a single contiguous run of tiles {tuple(merged)}. Widen the pattern, or lower "
                f"soft_causality/max_shift."
            )


# --------------------------------------------------------------------------- #
# The two objects                                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Background:
    """Everything that fixes the cache read + MicroZoi recompute geometry for one inference. Hashable, so it
    doubles as the cache-dedup / compatibility key: two specs with an equal ``Background`` share a background
    read and (given equal geometry) a clean difference.

    Parameters
    ----------
    chrom, map_start_bp : str, int
        The Hi-C map window is ``[map_start_bp, map_start_bp + n_bins*resolution)`` (n_bins/resolution come
        from the Manta object at run time; the spec is bp-based and model-agnostic).
    run_idx : int
        Concrete cached MicroZoi run (0..N_runs-1). Iterating ``0,1,2,...`` deterministically squeezes the
        cache; each run carries its own sub-bp shift, so this is also a randomness source.
    reverse : bool
        Reverse-complement orientation (a *different* cached read). L3 ``rc="average"`` pairs a forward and a
        reverse background and averages their (flipped-to-forward) maps.
    tile_offset_bins : int
        The resolved rigid shift of the whole ``tiles`` pattern, in ``-max_shift_bp/BIN_BP ..
        +max_shift_bp/BIN_BP`` (symmetric). L1 translates every brick by this; it changes each mutation's
        phase within its tile (the augmentation) without moving the cache read.
    crop_mha_bins : int
        MicroZoi crop; sets the tile size (``tile_size_bp``) the pattern was laid out with.
    tiles : tuple[(lo_bp, hi_bp), ...] | None
        The concrete offset-0 MicroZoi brick pattern to recompute and splice, from the mutation **superset**
        (so WT and every variant share it) or overridden for a fixed pattern (UC2). ``None`` = no recompute
        (pure cache read); a spec with such a background must have no mutations.
    soft_causality_bp, max_shift_bp : int
        The two knobs the pattern was sized with -- carried so validating a later mutation uses the same
        tolerance.
    """

    chrom: str
    map_start_bp: int
    run_idx: int
    reverse: bool = False
    tile_offset_bins: int = 0
    crop_mha_bins: int = DEFAULT_CROP_MHA_BINS
    tiles: tuple[tuple[int, int], ...] | None = None
    soft_causality_bp: int = DEFAULT_SOFT_CAUSALITY_BP
    max_shift_bp: int = DEFAULT_MAX_SHIFT_BP


@dataclass
class Spec:
    """One Manta inference = one output map. Immutable by convention; use :meth:`clone` to derive variants.

    ``mutations`` is a tuple of length-preserving edit tuples (empty = wild type); freeze random ops to
    ``replace`` first (see module docstring). ``channels`` is an optional output-channel subset (names or
    indices; ``None`` = all). ``tags`` are free-form labels used only by ``infer_grouped``'s ``group_by`` --
    they are **not** part of the compute identity.
    """

    bg: Background
    mutations: tuple = ()
    channels: tuple | None = None
    tags: dict = field(default_factory=dict)

    def clone(self, **overrides) -> "Spec":
        """A copy with fields overridden. ``mutations=...`` swaps the edits while keeping the same background
        (the basis for matched pairs / sweeps); ``tags={...}`` is merged into (not replacing) the tags."""
        tags = {**self.tags, **overrides.pop("tags", {})}
        bg = overrides.pop("bg", self.bg)
        mutations = tuple(overrides.pop("mutations", self.mutations))
        channels = overrides.pop("channels", self.channels)
        if overrides:
            raise TypeError(f"unexpected clone fields: {sorted(overrides)}")
        return Spec(bg=bg, mutations=mutations, channels=channels, tags=copy.deepcopy(tags))


# --------------------------------------------------------------------------- #
# Generators -- all randomness lives here, resolved into concrete specs         #
# --------------------------------------------------------------------------- #


def background_for(
    chrom,
    map_start_bp,
    *,
    run_idx,
    reverse=False,
    tile_offset_bins=0,
    crop_mha_bins=DEFAULT_CROP_MHA_BINS,
    mutations_superset=(),
    tiles=None,
    soft_causality_bp=DEFAULT_SOFT_CAUSALITY_BP,
    max_shift_bp=DEFAULT_MAX_SHIFT_BP,
) -> Background:
    """Build one Background, laying out the ``tiles`` pattern from the mutation *superset* unless given
    explicitly. Passing the superset (all mutations any sibling spec will carry) guarantees WT and every
    variant share one pattern. Validates soft causality and that the shift fits ``max_shift``."""
    if abs(tile_offset_bins) * BIN_BP > max_shift_bp:
        raise ValueError(
            f"tile_offset_bins {tile_offset_bins} (|{tile_offset_bins * BIN_BP}| bp) exceeds max_shift "
            f"{max_shift_bp} bp; the shift would spend more than the pattern's headroom."
        )
    if tiles is None and mutations_superset:
        coverage = auto_patch_windows(
            mutations_superset, soft_causality_bp=soft_causality_bp, max_shift_bp=max_shift_bp
        )
        tiles = tile_pattern(coverage, crop_mha_bins=crop_mha_bins)
    if tiles is not None:
        tiles = tuple((int(lo), int(hi)) for lo, hi in tiles)
        if mutations_superset:
            validate_patches(mutations_superset, tiles, soft_causality_bp=soft_causality_bp, max_shift_bp=max_shift_bp)
    return Background(
        chrom=chrom,
        map_start_bp=int(map_start_bp),
        run_idx=int(run_idx),
        reverse=bool(reverse),
        tile_offset_bins=int(tile_offset_bins),
        crop_mha_bins=int(crop_mha_bins),
        tiles=tiles,
        soft_causality_bp=int(soft_causality_bp),
        max_shift_bp=int(max_shift_bp),
    )


def variant_specs(bg, variants, *, tag_key="variant", channels=None) -> list[Spec]:
    """Fan one Background into N specs sharing it -- ``variants`` maps a tag value to a mutation list (or
    ``None``/`()` for wild type). E.g. ``{"wt": None, "mut": [("replace", p, s)]}`` -> a matched pair;
    ``{i: muts_i for i}`` -> an N-condition sweep. Each spec's mutations must lie within ``bg.tiles`` (build
    ``bg`` from the union of all variants' mutations)."""
    out = []
    for tag_val, muts in variants.items():
        muts = tuple(muts or ())
        if muts and bg.tiles is not None:
            validate_patches(muts, bg.tiles, soft_causality_bp=bg.soft_causality_bp, max_shift_bp=bg.max_shift_bp)
        out.append(Spec(bg=bg, mutations=muts, channels=channels, tags={tag_key: tag_val}))
    return out


def random_backgrounds(
    chrom: str,
    map_start_bp: int,
    *,
    runs: int | Iterable[int],
    rng: np.random.Generator | None = None,
    reverse: bool | str = False,
    crop_mha_bins: int = DEFAULT_CROP_MHA_BINS,
    mutations_superset=(),
    tiles=None,
    soft_causality_bp: int = DEFAULT_SOFT_CAUSALITY_BP,
    max_shift_bp: int = DEFAULT_MAX_SHIFT_BP,
) -> list[Background]:
    """Stamp out backgrounds to average over. ``runs`` is an int (use runs ``0..runs-1`` -- deterministic,
    squeezes the cache) or an explicit list of run indices. Each background draws an independent rigid shift
    in ``[-max_shift_bp/BIN_BP, +max_shift_bp/BIN_BP]`` bins. ``reverse`` may be ``False``, ``True``, or
    ``"both"`` (a forward and a reverse copy per run, for RC averaging). ``rng`` defaults to fresh entropy --
    don't hand a fixed seed to something you meant to average over. The tile pattern (shared by all of them)
    is laid out once from the mutation superset unless given explicitly."""
    rng = np.random.default_rng() if rng is None else rng
    run_list = list(range(runs)) if np.isscalar(runs) else list(runs)
    orients = [False, True] if reverse == "both" else [bool(reverse)]
    max_offset_bins = max_shift_bp // BIN_BP
    shared_tiles = tiles
    if shared_tiles is None and mutations_superset:
        coverage = auto_patch_windows(
            mutations_superset, soft_causality_bp=soft_causality_bp, max_shift_bp=max_shift_bp
        )
        shared_tiles = tile_pattern(coverage, crop_mha_bins=crop_mha_bins)
    out = []
    for r in run_list:
        for rev in orients:
            out.append(
                background_for(
                    chrom,
                    map_start_bp,
                    run_idx=r,
                    reverse=rev,
                    tile_offset_bins=int(rng.integers(-max_offset_bins, max_offset_bins + 1)),
                    crop_mha_bins=crop_mha_bins,
                    mutations_superset=mutations_superset,
                    tiles=shared_tiles,
                    soft_causality_bp=soft_causality_bp,
                    max_shift_bp=max_shift_bp,
                )
            )
    return out


def build_specs(
    backgrounds: list[Background],
    mutation_sets: dict[str, list],
    *,
    rng: np.random.Generator | None = None,
    fasta=None,
    background_meta: list[dict] | None = None,
    set_meta: dict[str, dict] | None = None,
    channels: Iterable[int | str] | None = None,
) -> list[Spec]:
    """
    Cross ``backgrounds`` x ``mutation_sets`` into the flat list of specs you actually run.

    ``mutation_sets`` maps a name to a mutation list (``[]`` = wild type). Each set is *frozen* per
    (background, set) cell (see :func:`manta_hic.ops.seq_ops.freeze_mutations`), which gives two behaviours
    for free:

    - a plain ``replace``/``invert`` set is identical across backgrounds (freezing is a no-op), so a known
      edit is fixed everywhere;
    - a random ``shuffle``/``inactivate`` set gets a **fresh realization per background**, so averaging over
      the background axis averages over runs, tile shifts *and* sequence realizations at once. This is why we
      steer users to backgrounds instead of hand-managed replicates: the replicate axis *is* ``backgrounds``.

    Each spec's ``tags`` carry ``{"background": i, "mutation_set": name}`` plus any user metadata:
    ``background_meta[i]`` (e.g. a dict describing run ``i``) and ``set_meta[name]``. Combine several matrices
    (different loci / patterns) by concatenating the returned lists. ``fasta`` (+ each background's
    ``chrom``) is required only if a set contains ``shuffle`` ops.
    """
    rng = np.random.default_rng() if rng is None else rng
    out = []
    for bi, bg in enumerate(backgrounds):
        bmeta = dict(background_meta[bi]) if background_meta is not None else {}
        for name, muts in mutation_sets.items():
            frozen = freeze_mutations(tuple(muts or ()), fasta=fasta, chrom=bg.chrom, rng=rng)
            if frozen:
                if bg.tiles is None:
                    raise ValueError(
                        f"mutation set {name!r} is non-empty but background {bi} has no tiles; build the "
                        f"backgrounds with mutations_superset covering every set."
                    )
                validate_patches(frozen, bg.tiles, soft_causality_bp=bg.soft_causality_bp, max_shift_bp=bg.max_shift_bp)
            tags = {"background": bi, "mutation_set": name, **bmeta, **((set_meta or {}).get(name, {}))}
            out.append(Spec(bg=bg, mutations=frozen, channels=channels, tags=tags))
    return out
