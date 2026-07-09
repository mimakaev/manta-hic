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
  chromosome. Averaging over runs smooths predictions; `scripts/subsample_cache.py` copies a cache keeping only
  a few runs for a smaller, shareable file.

`CachedMicrozoiFetcher` (`manta_hic/nn/fetchers.py`) reads windows from the cache and, for a **mutation**,
recomputes only the tiles the edit touches (rebuilding MicroZoi from `model_blob`, using the FASTA) and splices
them in — so a wild-type/mutant pair differs only where the mutation acts.

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
