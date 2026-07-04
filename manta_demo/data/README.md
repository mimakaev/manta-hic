# manta_demo/data/ — downloaded demo data

This folder is **git-ignored**. It holds the (largish) inputs the demo notebooks need; download them once and
drop them here. Nothing to build.

| file | what | ~size |
|---|---|---|
| `cache_hg38_chr22_2runs.h5` | small MicroZoi activation cache — **hg38, chr22 only, 2 stochastic runs** (fwd+rev), with the MicroZoi weights embedded. A slice of the full genome-wide cache, enough to run inference + mutations on chr22. | ~2 GB |
| `krietenstein_2048.pth` | trained **Manta** model — Krietenstein Micro-C, 2048 bp bins, 2 channels (`krietenstein-esc`, `krietenstein-hff1`). | ~120 MB |
| `krietenstein_2048.bhic.h5` | the **observed** Hi-C target (banded), for plotting prediction vs. truth. | ~1.3 GB |
| `hg38.fa` (+ `.fai`) | reference genome (needed for mutations). | ~3.2 GB |

## Download

Placeholder — fetch from the shared HTTP host into this folder, e.g.:

```bash
cd manta_demo/data
for f in cache_hg38_chr22_2runs.h5 krietenstein_2048.pth krietenstein_2048.bhic.h5 hg38.fa hg38.fa.fai; do
    curl -O "https://<host>/manta-demo/$f"
done
```

The notebooks resolve paths relative to this folder, so once the files are here they just run.

## Notes

- The cache is chr22-only and 2-run, so use `run_idx`/`manta_runs` ≤ 2 and query chr22. The full genome-wide,
  16-run caches live on the compute host; this is the portable subset.
- The MicroZoi cache is dataset-agnostic (same hg38 activations feed any hg38-trained Manta model), so you can
  swap in another `*_2048.pth` Manta head and reuse this cache.
