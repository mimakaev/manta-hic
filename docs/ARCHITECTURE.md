# Architecture

How the pieces fit together. For the inference API see [`INFERENCE.md`](INFERENCE.md); for the target storage
see [`HIC_STORAGE.md`](HIC_STORAGE.md).

## Two models, one cache

Predicting Hi-C from sequence is split into two models trained separately:

- **MicroZoi** (`manta_hic/nn/microzoi.py`, class `Microzoi`) — a Borzoi-like sequence model at a single 256 bp
  resolution: convolutions → a rotary-embedding transformer tower. Sequence → per-256bp-bin activations. The
  cached output is the transformer trunk ("mha", 1024 channels); it is **genome-agnostic** in that the trunk is
  shared, though of course the activations depend on the input sequence. Receptive field = 2^19 + 2^18 =
  786,432 bp.
- **Manta** (`manta_hic/nn/manta.py`, class `Manta`) — a 1D→2D convolutional model. It consumes MicroZoi
  activations (1024 trunk channels + 8 positional-encoding channels = 1032) over a wide window and produces a
  2D Hi-C map `[channels, n_bins, n_bins]`, with `n_bins = 1024` and `bins_pad = 128` fixed by the trained
  models. `channels` is the number of Hi-C tracks the model was trained on.

Because MicroZoi is expensive, its activations are **precomputed once per genome into a large HDF5 cache**, and
Manta trains and infers by slicing that cache. Running Manta itself is cheap.

## The activation cache

A cache (`microzoi_cache_<genome>_<fold>.h5`, built by `manta_hic io fill-cache`) stores, for one genome:

- root attrs: `genome`, `BIN_BP` (256), `CACHE_OVERHANG_BP`, `N_runs`, and `model_blob` — the exact MicroZoi
  weights the cache was built with (so a fetcher reconstructs the right model for mutation recompute).
- `N_runs` (=16) **stochastic runs** as groups `run0`…`run{N-1}`. Each run is a different augmentation
  (sub-bin shift, tile offset, crop) and holds both the **forward** and **reverse-complement** activations per
  chromosome. Averaging over runs smooths predictions; build a smaller, shareable cache by rerunning
  `manta_hic io fill-cache` with a lower `--n-runs` (e.g. `--n-runs 4`).

`CachedMicrozoiFetcher` (`manta_hic/nn/fetchers.py`) reads windows from the cache and, for a **mutation**,
recomputes only the tiles the edit touches (rebuilding MicroZoi from `model_blob`, using the FASTA) and splices
them in — so a wild-type/mutant pair differs only where the mutation acts.

### Building vs. downloading a cache

Building a genome-wide cache is the one genuinely GPU-heavy step. On an Apple-Silicon laptop (M4 Pro, `mps`)
`fill-cache` runs at roughly one MicroZoi tile-batch of 4 in ~2.2 s and stores ~6.7 MB per Mb per run
(compressed float16). For hg38 (chr1–22+X, ~3.0 Gb) that works out to about:

| runs | build time (M4 Pro) | size on disk |
|---|---|---|
| 2  | ~6.4 h  | ~41 GB  |
| 4  | ~12.5 h | ~81 GB  |
| 8  | ~24.7 h | ~162 GB |
| 16 | ~49 h   | ~325 GB |

(Higher runs are progressively slower per run: `crop_mha` grows across runs, so later runs lay more overlapping
tiles.) **Recommendation: download a prebuilt cache rather than building on a laptop** — the shared host carries
genome-wide caches (~90 GB / 4 runs each, one per held-out MicroZoi fold). Building locally only makes sense on
a CUDA machine, for a genome with no prebuilt cache, or for a single chromosome. Note MicroZoi batching does
**not** speed up `mps` (per-sample time is flat and memory scales linearly — ~8 GB per batch item), so on a Mac
use `--batch-size 1` or `2`; the CUDA default of 4 reserves ~26 GB and can push a laptop into swap.

## Resolutions

A Manta model works at one resolution, set by its tower height: `resolution = 2 ** (tower_height + 9)` — i.e.
1024, 2048, 4096, 8192, 16384 bp (and 256/512 at the floor). Trained resolutions are a property of the
checkpoint, not something you pass in.

## Self-describing checkpoints

`save_manta_checkpoint` writes `{"state_dict": ..., "config": {...}}`, where `config` carries `resolution`,
`n_bins`, `bins_pad`, `tower_height`, `output_channels`, `channel_names`, and `genome`. So loading is just
`MantaInference("saved_model.pth", fetcher)` — nothing about the model's shape, labels, or genome has to be
supplied or guessed, and a model whose genome disagrees with the target/data is rejected.

## Targets

Training targets (observed Hi-C) are stored in the **banded** `.bhic.h5` format (`manta_hic/io/banded.py`),
which is also what `MantaInference(target=…)` reads to plot predictions against truth. See
[`HIC_STORAGE.md`](HIC_STORAGE.md).

## Package layout

```
manta_hic/
  nn/     microzoi.py, manta.py, layers.py      models
          fetchers.py                           cache read + mutation recompute (CachedMicrozoiFetcher)
          inference.py                          MantaInference (turnkey prediction)
          specs.py                              pure-data spec layer for mutation/background sweeps
          dataset.py, train_manta.py, train_microzoi.py, fill_cache.py
  io/     banded.py, banded_write.py            banded Hi-C target storage
  ops/    seq_ops.py, kshuffle.py, hic_ops.py, tensor_ops.py
```
