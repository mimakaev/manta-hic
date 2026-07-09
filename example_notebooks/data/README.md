# example_notebooks/data/ — data for the example notebooks

This folder is **git-ignored**. The notebooks download what they need into here automatically (the *Setup*
cell at the top of each), so normally you don't touch this folder. The manual links are below for reference.

| file | what | ~size |
|---|---|---|
| `cache_hg38_chr14_2runs.h5` | small MicroZoi activation cache — **hg38, chr14 only, 2 stochastic runs** (fwd+rev), with the MicroZoi weights embedded. A slice of the full genome-wide cache, enough to run inference + mutations on chr14. | ~2 GB |
| `krietenstein_2048.pth` | trained **Manta** model — Krietenstein Micro-C, 2048 bp bins, 2 channels (`krietenstein-esc`, `krietenstein-hff1`). | ~120 MB |
| `krietenstein_2048.bhic.h5` | the **observed** Hi-C target (banded), for plotting prediction vs. truth. | ~460 MB |
| `hg38.fa` (+ `.fai`) | reference genome (needed for mutations); public, from UCSC. | ~3.2 GB |

## Download (manual)

The Setup cell in each notebook does this for you. To fetch by hand instead:

```bash
cd example_notebooks/data
curl -L -o cache_hg38_chr14_2runs.h5 "https://www.dropbox.com/scl/fi/e16epekvs0vjz2kcwk4za/cache_hg38_chr14_2runs.h5?rlkey=hm4ql129sg9ljxqdx2ik3woxo&dl=1"
curl -L -o krietenstein_2048.pth     "https://www.dropbox.com/scl/fi/m8gsdk6rb8ax5xsnofkj8/krietenstein_2048.pth?rlkey=7i3iokalnv2lkig51hamc0gff&dl=1"
curl -L -o krietenstein_2048.bhic.h5 "https://www.dropbox.com/scl/fi/rjvykayzga1luo5dd7azu/krietenstein_2048.bhic.h5?rlkey=el7o90wesehyrsbiif0b8z0f4&dl=1"
# hg38 reference (public):
curl -L https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz | gunzip > hg38.fa
samtools faidx hg38.fa   # or: python -c "import pysam; pysam.faidx('hg38.fa')"
```

## Notes

- The demo cache is single-chromosome (**chr14**) and 2-run, so use `runs`/`run_idx` ≤ 2 and query chr14. Both
  example notebooks live on chr14. The full genome-wide, 16-run caches are downloaded separately; this is the
  portable subset.
- The MicroZoi cache is dataset-agnostic (the same hg38 activations feed any hg38-trained Manta model), so you
  can swap in another `*_2048.pth` Manta head and reuse this cache.
