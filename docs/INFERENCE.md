# Spec-based Manta inference

How to predict Hi-C maps (and mutation effects) from a trained Manta model. This is the *usage* guide and
the record of the **conventions** the stack relies on; the design rationale is in
`docs/INFERENCE_SPEC_PLAN.md` (the design log).

## Quick start

```python
import pysam
from manta_hic.nn.fetchers import CachedMicrozoiFetcher
from manta_hic.nn.inference import MantaInference

fetcher = CachedMicrozoiFetcher("cache.h5", fasta_open=pysam.FastaFile("hg38.fa"))
infer = MantaInference("saved_model.pth", fetcher, device="cuda:0")  # resolution read from the checkpoint

# One averaged map per condition, at a locus. Silence an enhancer E and a promoter P:
E = ("inactivate", 22_000_000, 22_005_000)
P = ("inactivate", 22_600_000, 22_602_000)
maps, names = infer.predict_region(
    "chr1", 20_000_000,
    {"WT": [], "dE": [E], "dP": [P], "dEP": [E, P]},
    runs=8, rc="average",
)
# maps["dE"] - maps["WT"]  ->  the enhancer's effect, [C, 1024, 1024], forward-oriented
```

## The layers

| layer | where | what |
|---|---|---|
| **L0** specs | `nn/specs.py` | pure data: `Background`, `Spec`, mutation geometry, the tile pattern, generators (`random_backgrounds`, `variant_specs`, `build_specs`). No torch. |
| **L1** activations | `nn/fetchers.py` `CachedMicrozoiFetcher.fetch_activations_batch` | specs → activation tensors on GPU (cache read + mutation recompute + splice), dedup'd. |
| **L2** maps | `nn/inference.py` `MantaInference.infer` / `infer_grouped` | activations → Manta → forward-oriented, channel-sliced maps; `infer_grouped` streams a sweep and averages by tag. |
| **L3** one call | `nn/inference.py` `MantaInference.predict_region` | generate specs + `infer_grouped` for a locus. The tool you reach for. |

A *spec* fully describes one inference: a `Background` (cache run, orientation, tile pattern, tile shift) plus
this sample's `mutations`, output `channels`, and free-form `tags`. Because every knob is resolved, the whole
pipeline is a deterministic function of the spec.

## Checkpoint format

Trained checkpoints are **self-describing**: `save_manta_checkpoint` (called by training) writes
`{"state_dict": ..., "config": {resolution, n_bins, bins_pad, tower_height, output_channels, channel_names?,
model_params?}}`. `MantaInference` reads `resolution` straight from `config`, so it never has to guess it from
tensor shapes — that guess is ambiguous at ≤1024 bp (256/512/1024 all yield one conv block). A bare
state-dict still loads (falls back to `target=`/`tower_height=`/a shape guess with a warning), but prefer the
rich format.

To upgrade old bare `saved_model.pth` files, run `scripts/repack_manta_checkpoints.py` (resolution from a
`<name>_<res>_<fold>/` dir name or `--resolution`, optional `--channel-names-from target.bhic.h5`):

```bash
python scripts/repack_manta_checkpoints.py /path/to/2024_manta_trained_models/   # whole tree, in place with --in-place
```

## Conventions

These are load-bearing. Break them and results go subtly wrong rather than crash.

### Orientation (reverse complement) is a **map**-level concern, not an activation-level one
A reverse spec carries **reverse-complement activations all the way through L1** — the cache read is the
reverse dataset, and the mutation recompute uses reverse sequence. Those activations are **never flipped to
"forward"**; Manta consumes them natively and produces a map in the reverse orientation. **L2 flips that map
back to canonical forward** with `torch.flip(dims=(-2,-1))` — a cheap **GPU** op applied at the L2 boundary,
right before the map is returned or averaged. So:

- **Activations stay native (per-spec) orientation.** There is no "forward-canonical activation"; don't try
  to make one. (This directly answers "does the RC convention hold across activations?" — no.)
- **Maps are canonicalized to forward** the moment they leave L2. Everything downstream (averaging, plotting,
  differencing) sees forward-oriented maps, so it can ignore orientation entirely.
- Orientation is **bookkeeping in the spec** (`bg.reverse`, plain CPU-side data); the model is
  orientation-agnostic. Averaging forward and reverse (`rc="average"`) is clean *because* both are flipped to
  forward first.

### Coordinates are **patch** (splice/output) coordinates, not MicroZoi fetch coordinates
Everything you specify or read — `map_start_bp`, mutation positions, `Background.tiles` — is in genomic bp and
refers to the **output** window: the bins that are recomputed and **spliced** into the activation array. The
MicroZoi **input** window (a patch expanded by `crop_mha` on each side, one receptive field ≈ 786 kb wide) is
an internal detail of the recompute you never specify. "Patch/tile start–end" always means the spliced region;
the wider sequence fetch is implicit.

### Specs are fully resolved → matched pairs are clean *for free*
Randomness (which run, which shift, which inert sequence) is baked into the spec **when it's generated**, so
two specs that share a `Background` produce a byte-identical wild-type background and recompute. A WT/mutant
pair on one background therefore differs **only** where the mutation acts — the engine needs no "pair" logic.
Corollary: to average over runs/shifts/orientation, generate more specs; to compare conditions, keep the
`Background` shared (`build_specs(bgs, conditions)`).

### The tile pattern is set in L0; L1 only shifts it rigidly
`Background.tiles` is the concrete MicroZoi brick pattern (greedy-merged, so two nearby mutations share one
run of tiles; distant ones split). A per-background `tile_offset_bins` **rigidly translates the whole
pattern** — a pure shift that still augments (each mutation lands at a different phase within its tile). L1
does no tiling or merging; it shifts and runs. Each tile recomputes only the mutations that fall inside it
(distant clusters recompute wild-type there, so they cancel in a matched pair).

### Patch tolerance = `soft_causality + max_shift` (in **bins**)
A patch extends `soft_causality_bp + max_shift_bp` beyond each mutation, so after any shift up to `max_shift`
the mutation stays ≥ `soft_causality` from the recomputed/cached seam. Defaults are **256 bins each** (a
512-bin tolerance, 256-bin post-shift margin) — placeholders pending calibration. `soft_causality` is the
assumed reach of a mutation's influence through MicroZoi; `max_shift` is the augmentation shift range.

### Mutations are independent of specs; random ops are *frozen* before use
Ops: `("replace", pos, seq)`, `("invert", p1, p2)`, `("shuffle[k]", p1, p2)`, `("inactivate", p1, p2)`.
`inactivate` and `shuffle` are **random**; `freeze_mutations` (and `build_specs`, which calls it per
(background, condition) cell) resolves them to concrete `replace` tuples. So a random mutation gets a **fresh
realization per background** — averaging over backgrounds also averages over inert-sequence realizations.
Length-changing ops (`insert`, deletions) are rejected: the cache is a fixed genomic bin grid.

### Channels are a post-Manta slice
`Spec.channels` (indices or names, or `None` for all) selects output channels **after** the Manta forward —
it shrinks the returned stack, not the compute. Names come from the target file's `shortnames`.

### `matched` vs `random` backgrounds (in `predict_region`)
- **matched** (default): all conditions share each background — synchronized run / tile shift / tiling. The
  per-replicate difference is the mutation alone; averaging gives a **polished** comparison.
- **random**: each condition draws its own independent backgrounds and tiles only what it must (wild type
  stays a pure cache read). Nothing is synchronized, so the spread across conditions reflects the model's
  **natural variability**, not an isolated effect. Use it to see the noise floor.

## Averaging (`infer_grouped`)

`infer_grouped(specs, group_by)` streams one `Background` at a time (so each cache read happens once and the
whole sweep is never held in memory), flips RC maps to forward, and folds each into an accumulator keyed by
the spec's `group_by` **tags**. Everything *not* in `group_by` (runs, shifts, orientation, realizations) is
averaged. Memory is O(#groups). Rule of thumb: **average maps first, compare after** — the workhorse only
averages within a group; wt-vs-mut ratios/log-ratios are a post-step on the resulting maps (a per-pair
log-ratio reducer is a possible future opt-in). `group_by=[]` averages everything into one map.

## API reference (essentials)

**L0 — build specs** (`manta_hic.nn.specs`)
- `random_backgrounds(chrom, start_bp, *, runs, reverse=False|"both", max_shift_bp=..., mutations_superset=..., rng=None)` → list of `Background` (runs `0..runs-1`, each a random shift; the pattern is laid out once from the superset).
- `build_specs(backgrounds, conditions, *, rng=None, fasta=None, channels=None, background_meta=None, set_meta=None)` → flat `[Spec]` crossing backgrounds × conditions, freezing each condition per cell; tags carry `{"background": i, "mutation_set": name, **meta}`.
- `variant_specs(bg, {name: muts})`, `background_for(...)`, `freeze_mutations(muts, *, fasta, chrom, rng)` for lower-level control.

**L1 — activations** (`CachedMicrozoiFetcher`)
- `fetch_activations_batch(specs, *, resolution, n_bins=1024, bins_pad=128, device) → [Tensor]` — one `[n_channels, window]` per spec; dedups identical cache reads within the batch.

**L2 — maps** (`MantaInference`)
- `infer(specs, *, batch_size=4) → (maps, names)` — per-spec forward-oriented, channel-sliced maps (small/explicit sweeps).
- `infer_grouped(specs, group_by, *, batch_size=4) → [(key, mean_map, names)]` — streaming group-average (big sweeps).

**L3 — one call** (`MantaInference`)
- `predict_region(chrom, start_bp, conditions, *, runs=8, max_shift_bins=100, rc="average", backgrounds="matched", channels=None, rng_seed=None) → (maps, names)` — `conditions` a dict `{name: muts}` (or list, auto-named); returns `{name: forward_map}`.

## Examples

Matched pair, one call:
```python
maps, _ = infer.predict_region("chr1", 20_000_000, {"WT": [], "dE": [E]}, runs=8)
effect = maps["dE"] - maps["WT"]
```

Explicit sweep you want every map from (no averaging):
```python
from manta_hic.nn.specs import random_backgrounds, build_specs
bgs = random_backgrounds("chr1", 20_000_000, runs=4, reverse="both", mutations_superset={E, P})
specs = build_specs(bgs, {"WT": [], "dE": [E], "dEP": [E, P]})
maps, names = infer.infer(specs)            # 12 maps, one per (background, condition)
```

Natural-variability view (unsynchronized):
```python
maps, _ = infer.predict_region("chr1", 20_000_000, {"WT": [], "dE": [E]}, backgrounds="random")
```
