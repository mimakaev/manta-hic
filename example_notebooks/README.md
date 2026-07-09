# Example notebooks

Runnable examples of using a trained Manta model for inference. Read them in order — each builds on the last.

| notebook | what it shows |
|---|---|
| [`01_getting_started`](01_getting_started.ipynb) | The core objects: `BandedHicFile` (observed Hi-C), `MantaInference` (predict a map), and plotting prediction vs. truth for a window. Start here. |
| [`02_loop_anchor_silencing`](02_loop_anchor_silencing.ipynb) | A first mutation experiment: silence each anchor of a predicted loop and see whether the model says the loop depends on it. |
| [`03_enhancer_promoter`](03_enhancer_promoter.ipynb) | Comparing several conditions at once (WT / dE / dP / dEP), averaged over cache "backgrounds" for a polished readout. |
| [`advanced_spec_geometry`](advanced_spec_geometry.ipynb) | Internals, not everyday use: how a mutation is turned into the MicroZoi tiles that get recomputed. Read `docs/INFERENCE.md` alongside it. |

## Prerequisites

- `manta-hic` installed (see the top-level [README](../README.md)) and a **CUDA GPU** for the inference notebooks.
- The data files in [`data/`](data/) — they are git-ignored; download them once (see [`data/README.md`](data/README.md)).

The notebooks resolve their inputs from `data/` relative to this folder, so launch Jupyter from here (or with
this folder as the working directory) and they just run.

## Where to go next

- [`docs/INFERENCE.md`](../docs/INFERENCE.md) — the full inference API and conventions.
- [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) — how the two models and the activation cache fit together.
