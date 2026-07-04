# Manta inference rewrite — spec-based plan

Status: **L0 implemented; L1/L2/L3 pending** (2026-07). Owner: Max. This is the undiluted record of the
design conversation so we can build against it. See also `INFERENCE.md` (usage) and `HIC_STORAGE.md`.

Progress:
- **L0 done** — `manta_hic/nn/specs.py` (`Background`, `Spec`, `clone`, mutation geometry, `auto_patch_windows`
  = coverage, `tile_pattern` = concrete greedy MicroZoi bricks, `validate_patches`, generators `background_for`
  / `variant_specs` / `random_backgrounds`, and `build_specs` = the background × mutation-set matrix). Tests:
  `tests/test_specs.py` (16). `build_specs` freezes each set per (background, set) cell, so random ops
  (`inactivate`/`shuffle`) get a fresh realization per background — the **background axis is the replicate
  axis** (average over it to average runs × shifts × realizations). Tags carry `{"background": i,
  "mutation_set": name, **user_meta}`.
- **The tile pattern is established in L0, not L1.** `Background.tiles` holds the concrete offset-0 MicroZoi
  brick pattern (from the mutation superset, or overridden). The greedy tiler ("keep tiling if the current
  run reaches the next window") does the tile-count optimization *here*, where tile size is known and it's
  deterministic — e.g. two 1.1-tile windows 0.2 tile apart → 3 tiles, not 4. `tile_offset_bins` is a
  **symmetric rigid shift** (`±max_shift/BIN_BP`) of the whole pattern; it's a pure translation yet still
  augments (each mutation lands at a different phase within its tile). **L1 just shifts the stored `tiles` and
  runs them — no tiling or merging logic downstream.** Coverage tolerance = `soft_causality + max_shift`. Mutations are **independent
  of specs**: `inactivate` is a real op, and `ops/seq_ops.freeze_mutations` resolves the random ops
  (`shuffle[k]`, `inactivate`) to concrete `replace` (so no `inactivate_specs` special case). `seqq` +
  `make_quiescent_seq` moved to `ops/seq_ops.py` (rng-aware; `mutate_manta` imports them).
- **Two-knob patch model** (replaces the single `radius`): the caller sets `soft_causality_bp` (min recompute
  margin a mutation needs) and `max_shift_bp` (per-background random tile-shift range); patch tolerance =
  their **sum**, so after any shift ≤ `max_shift` a mutation stays ≥ `soft_causality` from the seam.
  `cluster_mutations` merges mutations closer than ~2/3 of a MicroZoi tile into one patch.
- **L1 done** — `CachedStochasticActivationFetcher.fetch_activations_batch(specs, *, resolution, n_bins,
  bins_pad, device)` in `nn/manta.py`. Per spec: pure cache read if `bg.tiles is None`; else read the cached
  background and, for each contiguous run of `bg.tiles` **rigidly shifted by `tile_offset_bins`**, recompute
  from the (frozen) mutations via the existing `_recompute_patches` (soff=0; a WT spec recomputes wild-type so
  a pair cancels) and `_splice` it in. **Dedup:** specs sharing a Background's cache window + run + orientation
  read the cache once, then each patches a private copy. Returns one `[n_channels, window]` tensor per spec on
  device, in order. Validated on real data (demo cache + hg38): dedup (4 specs → 1 read), and mut−wt nonzero
  *exactly* inside the tiles, zero elsewhere. Tests: `tests/test_activations_batch.py` (CPU orchestration;
  GPU recompute validated by script). Note: `_recompute_patches` hardcodes `genome="hg38"` (mm10 TODO).
- **L1 note:** each spec's mutations are filtered **per tile-run** before recompute — a mutation only applies
  to the run whose window contains it; distant runs recompute wild-type (so a far cluster's tile still cancels
  in a matched pair). Needed once patterns have >1 run (e.g. an enhancer + a promoter 600 kb apart).
- **L2 done** — `MantaInference.infer` / `infer_grouped` in `nn/inference.py` (on top of
  `fetch_activations_batch`). `infer(specs)` -> per-spec forward-oriented, channel-sliced maps (Manta forward,
  **RC-flip to canonical forward on-GPU**, channel slice); returns `(maps, names)` for small/explicit sweeps.
  `infer_grouped(specs, group_by)` streams one Background at a time (cache read once), folds each
  forward-oriented map into an accumulator keyed by the `group_by` tags (everything else averaged;
  `group_by=[]` = average all), memory O(#groups). Map-space mean, RC-aware; per-pair-logratio reducer is a
  later opt-in. `predict_pair` was reimplemented on L1 earlier and stands; `predict`/`predict_pair` could be
  re-expressed on `infer`. Tests: `tests/test_inference_l2.py` (CPU: RC-flip, channel slice, group-average).
  Validated on GPU: E/P matrix (16 specs, 2 runs × both orientations) -> 4 averaged condition maps, all finite.
- **L3 done** — `MantaInference.predict_region(chrom, start_bp, conditions, *, runs=8,
  patch_max_random_offset_bins=100, rc="average", backgrounds="matched", channels=None, rng_seed=None)`.
  Generates specs and calls `infer_grouped(["mutation_set"])`; returns `{name: forward_map}`. `conditions` is
  a dict `{name: muts}` or a list (auto-named). `rc` is `False | True | "average"`. **The `backgrounds`
  switch needed no L0 change** — `"matched"` shares one background set (from the superset) across conditions
  (`build_specs(bgs, conditions)`); `"random"` calls `random_backgrounds` per condition (independent shifts,
  tiling from each condition's own mutations, so WT stays pure cache), then concatenates. Validated on GPU:
  random mode gives larger condition differences (dEP-WT 0.010 vs 0.0049 matched) — the natural-variability
  view. Tests: `tests/test_inference_l2.py` (matched shares 4 backgrounds, random makes 8; list/dict
  conditions).
- **User doc:** `manta-hic-dev/docs/INFERENCE.md` (usage + the conventions: RC is a map-level flip on-GPU at
  the L2 boundary / activations stay native; patch-not-fetch coordinates; matched vs random; freezing; etc.).
- **The full L0→L3 stack is built and validated.** Remaining polish: derive `resolution` from the checkpoint
  (drop the `resolution=` arg); optionally re-express `predict`/`predict_pair` on `infer`; pool recompute
  tiles across specs into shared MicroZoi batches (v1 recomputes per spec; dedup is on the cache read only);
  calibrate `soft_causality`/`max_shift`.

## Goal

Replace the ad-hoc inference/mutation code (`nn/mutate_manta.py` from-scratch recompute; the thin
`MantaInference` wrapper; the per-notebook hand-rolled sweeps) with a small **spec-based** stack. All
complexity moves to two ends — **generating specs** (upstream) and **collapsing outputs** (downstream) —
and the middle is a dumb, deterministic, batched engine.

## Core realization

If a spec carries **fully-resolved** randomness (an actual `run_idx`, an actual `shift_bp`, an actual tile
offset — nothing left as `'auto'`), then the whole pipeline is a **pure deterministic function of the spec**:
the cache read is deterministic given `run_idx`; the recompute is deterministic given
`(shift_bp, offset, crop, window, mutations)`.

Consequences:
- **Matched-pair correctness is free.** A WT spec and a mutant spec that share a `Background` produce
  byte-identical backgrounds and byte-identical WT recompute, so `mut − wt` is clean *whether or not the
  engine co-processes them*. The engine needs **no** "pair" concept.
- **Batching is pure performance**, never affects results. Cache-read dedup + MicroZoi tile pooling are
  optimizations that can be added/removed freely.
- **Reproducibility = the spec list.** Save the specs (they're plain data) and you've saved the experiment.

## Locked decisions

1. **No `resolution` argument.** A Manta checkpoint self-describes:
   - `output_channels` ← `final_conv.weight.shape[0]`
   - `tower_height` ← `len(conv_blocks_1d)`; `resolution = 2**(tower_height + 9)` (verified: res 1024→1
     block, 4096→3, 16384→5).
   - `n_bins` (1024) and `bins_pad` (128) become **module-level constants**; never trained otherwise, stop
     threading them as args.
   - Add a helper to interrogate a **live Manta object** →
     `{tower_height, resolution, n_bins, bins_pad, output_channels}`.
   - **Verify at construction**: `manta.output_channels == banded.n_channels` **and**
     `manta_resolution == banded.resolution`. Fail loud on mismatch.

2. **Two inference entry points**, same core, different memory contract:
   - `infer(specs)` → returns the full **stack** of maps (fine for small/explicit sweeps where you want
     every output).
   - `infer_grouped(specs, group_by, …)` → **streams** spec-batches and folds each output into a group
     accumulator; only ever holds the group means. Memory O(n_groups), not O(n_specs). Required because a
     100-pos × 16-variant × 2-RC sweep is thousands of specs / multi-GB of activations.

3. **Collapse = map-space mean, RC-aware; comparisons happen post-collapse.**
   - Averaging is linear, so `mean(mut) − mean(wt) == mean(mut − wt)` and the per-run seam cancels
     regardless of order. So average maps within a group; do wt-vs-mut ratio/logratio *after* collapse on
     the ~2 resulting maps. **"Average first, then divide"** is the rule of thumb.
   - A per-pair `log(mut/wt)`-then-average reducer is a **later opt-in**, not v1. (Rationale: `mutate_manta`
     used a per-rep median because a mutation can *create* a random TF site, so replicate spread mattered.
     For a **known** mutation you're just squeezing the existing model to be sure it didn't miss something on
     the first pass — map-space mean is the right default.)

4. **Explicit `rng` in every generator, default `None` → fresh entropy.** Reproducible when you pass one,
   but never footgun a scientist trying to average 10 random realizations with a fixed seed.

## Conventions

- **RC / flipping — the isolation boundary.** Hi-C tensors live **unflipped (native model orientation) on
  the GPU**; flipping to canonical forward happens as part of **leaving the GPU**. Each L2 tool flips RC maps
  (`torch.flip(dims=(-2,-1))`) **on-GPU, then pulls**. So every L2 output is already forward-oriented and the
  collapse layer is orientation-agnostic. L1 (activations) returns **spec-oriented** activations (reversed if
  `spec.bg.reverse`); a caller who bypasses L2 to run their own Manta owns the flip.
- **Sample meta / tags live *inside* the spec**, in a dedicated `tags: dict` field that is **excluded from
  `Background`** (the compute identity). Cache-dedup / "compatibility" keys on `Background` only; `tags` is
  pure label used solely by `infer_grouped`'s `group_by`. `clone` carries/overrides tags.
- **Soft-causality validation (L0 `validate_patches`, re-checked in L1).** Every mutation's span expanded by
  `soft_causality_bp + max_shift_bp` must lie inside the **union of `Background.tiles`** — else a
  recomputed/cached seam falls within the causality margin and a near-mutation bin keeps stale wild-type
  activations. Internal seams *between* adjacent tiles are fine (both recompute the mutation, and MicroZoi's
  large crop means each tile still sees it). Both knobs travel on the `Background` so validation uses the same
  tolerance the pattern was sized with. **Defaults are PLACEHOLDERS (`256×BIN_BP` each → 512-bin tolerance,
  256-bin post-shift margin) pending calibration. UNIT CHECK NEEDED: Max wrote both "256 bins" and "256 bp" —
  current code uses 256 *bins*; confirm.**
- **Spec is bound to a live Manta object** for the run — that's what gives `n_bins`/`bins_pad`/alignment/
  resolution for free (hence the introspection helper).
- **bp-offset is NOT a randomness source for patches.** A sub-bin bp shift can split a CTCF site across two
  bins and Manta hates that. Randomness comes from (a) pulling different cache runs and (b) random **tile
  offset in bins**. (Different cache runs already carry different bp offsets internally — that's fine, it's
  the *patch* bp-offset we avoid.)

## Layer cake

```
L0  data (pure, CPU, unit-tested)
      Background(frozen): chrom, map_start_bp, reverse, run_idx, shift_bp,
                          tile_offset_bins, crop_mha_bins, patch_window=None   # None=auto, set=FIXED (UC2)
      Spec(frozen):       bg, mutations=(), channels=None, tags={}
      clone(spec, **overrides)
      generators: matched_pair(bg, mutations), random_backgrounds(chrom, start, n, rng, ...),
                  inactivate_specs(bg, span, n, rng)   # bakes n pre-shuffled seqq strings -> n replace specs

L1  fetcher (MicroZoi/cache only)  — the "MZ GPU tensor out" mode
      fetch_activations_batch(specs, batch_size) -> (act_stack_on_gpu, order)
      * mutation-free specs = pure cache read (fast path)
      * mutated specs pool recompute tiles into shared MicroZoi batches (reuse _recompute_patches/_splice)
      * validates patch coverage (>=0.15*patch_size flank)

L2  MantaInference (fetcher + manta)
      infer(specs, batch_size) -> (map_stack_forward_oriented, names)   # L1 + Manta + RC-flip + channel slice

L2' MantaInference — the workhorse
      infer_grouped(specs, group_by, rc_average=True) -> [(key_dict, mean_map, names)]
      * streams batches, RC handled at L2, scatter-adds into O(groups) accumulators, divides at the end,
        deterministic key order

L3  MantaInference — one friendly tool
      predict_region(...)   (see below)
```

`predict`, `predict_pair`, `predict_conditions` (if kept) are all thin spec-generator + collapse on top —
"production single map", "matched pair", "matched N-conditions", "big sweep" are the same two engine
functions with different generators and different `group_by`.

## L3 — `predict_region` (the only L3 tool we build now)

```
predict_region(chrom, start_bp, mutations=[None], *,
               runs=8,                     # iterate cache runs 0..runs-1 (deterministic; squeeze
                                                 #   all cache info), average the output MAPS
               patch_max_random_offset_bins=100, # per-sample random tile offset (a real randomness source)
               avg_rc=True,                      # also run reverse (run r AND r_reverse), average maps
               infer_rc=False,                   # [SEMANTICS TO CONFIRM] return a normal-oriented map even
                                                 #   when using RC; can be set True if avg_rc=True
               rng_seed=None,
               channels=None)
      -> one averaged, forward-oriented map per entry in `mutations` (+ channel names)
```

- `mutations` is a **list of variants**: `None` = WT, or a mutation list — one output map per entry (so a
  matched pair is `mutations=[None, [muts]]`; N conditions is a longer list).
- Averaging sources, all map-space: `runs` (deterministic 0..N-1), random tile offset, RC. That's the
  full set of variation we can actually exploit; there is no other clean source.
- **Deliberately drops MicroZoi activation-run-averaging** (averaging several cache runs' *activations* into
  one input, the old `n_runs>1` mean) — pending the exploration below. Map-space run averaging stays.

## Open questions / to verify

- **`infer_rc` exact semantics** (confirm with Max).
- **EXPLORATION — DONE (2026-07). Verdict: axe MicroZoi activation-run-averaging from inference.**
  16 eligible chr1 windows, 4dn-diff_4096_all model + matching hg38_all cache, metric = Spearman of
  pred-OOE vs adaptive-coarsegrained observed-OOE (`coarsegrained_hic_corrs`, the training metric):

  | scheme    | Spearman | Δ vs single |
  |-----------|----------|-------------|
  | single    | 0.8234   | +0.0000     |
  | mz_avg2   | 0.8243   | +0.0008     |
  | mz_avg4   | 0.8241   | +0.0007     |
  | mz_avg8   | 0.8245   | +0.0011     |
  | map_avg2  | 0.8244   | +0.0010     |
  | map_avg4  | 0.8245   | +0.0010     |
  | map_avg8  | 0.8249   | +0.0014     |

  Two conclusions: (1) *any* run-averaging barely moves correlation (~0.1% Spearman; per-window noise is
  0.62–0.91, ~100× larger). (2) **Activation-averaging is not better than map-space averaging** —
  `map_avg2` (two full runs, averaged after Manta) already matches `mz_avg8`, and `map_avg8` is best
  overall. So the MicroZoi-activation mean buys nothing the (free, more useful) map-space average doesn't.
  → **Inference offers no activation-averaging.** `predict_region` averages *maps* over `runs`. The
  fetcher's `n_runs>1` activation mean stays available as a low-level primitive but is not wired into the
  spec engine.
  → **Training:** `sample_n_runs` (activation-averaging augmentation in `train_manta`) is a *different*
  rationale (mild denoising regularizer, not correlation) — left as-is for now, flagged as a low-value
  candidate for removal; measure separately before touching trained-model reproducibility.
  → Caveat: absolute-correlation test only. Map-space run averaging still earns its keep for **matched
  pairs** (denoising the mut−wt difference) and RC/variance, which this test does not probe.

## Use cases this must serve (for reference)

- **UC1 — explicit mutation sweep:** N paired WT/mutant pairs, each from a distinct cache run, same
  mutation, batched, per-sample tile offset. (= `random_backgrounds` → `matched_pair` per bg → `infer`.)
- **UC2 — local sweep / pairwise combos (e.g. all pairs of 7 CTCF sites):** needs a **fixed** patch window
  shared across conditions so tiling never drifts. (= set `Background.patch_window` on the base, clone with
  different mutation sets.)
- **UC3 — simple matched pair / single map:** `predict_region(mutations=[None, muts])`.
- **Many-Manta-models sweep:** call L1 `fetch_activations_batch` once, run the user's own N Manta heads over
  the returned GPU stack (the old `mutate_manta` pattern, minus the from-scratch cost).
