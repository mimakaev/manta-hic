# manta-hic

**Predict Hi-C contact maps from DNA sequence** — a neural network that turns a stretch of genome (with any
edits you like) into a predicted Hi-C map, so you can ask *in silico* how sequence changes reshape 3D genome
organization.

```python
maps, names = infer.predict_region("chr1", 20_000_000, {"WT": [], "dE": [enhancer]}, runs=8)
effect = maps["dE"] - maps["WT"]        # the enhancer's predicted effect on the contact map
```

## What it is

Two models trained separately and joined by a **cache of activations**:

1. **MicroZoi** — a Borzoi-like sequence model (single 256 bp resolution, transformer with rotary
   embeddings). Sequence → per-bin activations. Its receptive field is ~786 kb.
2. **Manta** — takes MicroZoi activations over a wide window → a 2D Hi-C map
   `[channels, n_bins, n_bins]` (1024×1024 bins) at the resolution the model was trained for
   (1024 / 2048 / 4096 / … bp).

MicroZoi is expensive, so its activations are **precomputed once per genome into large HDF5 caches**, and
Manta predicts by slicing those caches. For a mutation, only the handful of cache tiles the edit touches are
recomputed on the fly — so a wild-type-vs-mutant comparison is cheap and exact. Trained Manta checkpoints are
**self-describing**: resolution, channels, channel names, and genome all travel inside the file.

## Install

Requires **Python ≥ 3.12**, PyTorch ≥ 2.6, and a CUDA GPU for real inference.

```bash
pip install git+https://github.com/mimakaev/manta-hic
# or, from a clone:
pip install -e .
```

Everything is plain `pip`-installable (no build hacks). See `requirements.txt` for the dependency floors.

## Data

Inference needs two files plus a genome:

| file | what | needed for |
|---|---|---|
| a **Manta checkpoint** (`*.pth`) | the trained head (self-describing) | always |
| a **MicroZoi cache** (`*.h5`) | precomputed activations for the genome (MicroZoi weights embedded) | always |
| the **reference FASTA** (`hg38.fa` / `mm10.fa`) | genome sequence | only for **mutations** |

Trained models and MicroZoi caches are available from a shared HTTP host — **contact the authors for access**
(some of it is still being finalized). The reference genomes are public (e.g. UCSC `hg38.fa`). Caches are
large (genome-wide, 16 stochastic runs); a 4-run subset (~4× smaller) is enough for most inference and can be
made with `scripts/subsample_cache.py`.

## Quick start

```python
import pysam
from manta_hic.nn.fetchers import CachedMicrozoiFetcher
from manta_hic.nn.inference import MantaInference

fetcher = CachedMicrozoiFetcher("microzoi_cache_hg38.h5", fasta_open=pysam.FastaFile("hg38.fa"))
infer = MantaInference("saved_model.pth", fetcher, device="cuda:0")   # resolution/channels/genome from the file

# one averaged map for a locus
maps, names = infer.predict_region("chr1", 20_000_000, {"WT": []}, runs=8)
wt = maps["WT"]                              # [channels, 1024, 1024] torch tensor

# mutation effect: silence an enhancer and difference against WT
E = ("inactivate", 20_400_000, 20_402_000)
maps, _ = infer.predict_region("chr1", 20_000_000, {"WT": [], "dE": [E]}, runs=8)
effect = maps["dE"] - maps["WT"]
```

Point `MantaInference(..., target="….bhic.h5")` at the observed Hi-C to plot prediction vs. truth for the
same window. Full walkthroughs are in the notebooks below.

## Example notebooks

Runnable, in reading order — see [`example_notebooks/`](example_notebooks/) (each resolves its data from
`example_notebooks/data/`; download instructions are in that folder's README).

| notebook | what it shows |
|---|---|
| [`01_getting_started`](example_notebooks/01_getting_started.ipynb) | load a model + cache, predict a map, plot it against the observed Hi-C |
| [`02_loop_anchor_silencing`](example_notebooks/02_loop_anchor_silencing.ipynb) | does silencing a loop anchor break the loop? a first mutation experiment |
| [`03_enhancer_promoter`](example_notebooks/03_enhancer_promoter.ipynb) | compare several conditions (WT / dE / dP / dEP) averaged over backgrounds |
| [`advanced_spec_geometry`](example_notebooks/advanced_spec_geometry.ipynb) | internals: how a mutation becomes the tiles that get recomputed |

## Documentation (deep dives)

Reference material for when you need the details:

- [`docs/INFERENCE.md`](docs/INFERENCE.md) — the inference API, the load-bearing conventions (orientation,
  coordinates, matched pairs), and how to build spec sweeps.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — the two-model design, the activation cache, and resolutions.
- [`docs/HIC_STORAGE.md`](docs/HIC_STORAGE.md) — the banded (`.bhic.h5`) Hi-C storage used for targets.

## Command line

Beyond the Python API, `manta_hic` exposes a small CLI (mostly for producing data, not inference):

- `manta_hic train manta` / `manta_hic train microzoi` — train the models.
- `manta_hic io fill-cache` — build a MicroZoi activation cache for a genome.

## Attribution & license

Includes code derived from Meta's Llama 2 (see `NOTICE` / `LICENSE`). Tensor-shape convention used throughout:
`B` batch, `C` channels, `N` bins, `H` heads.
