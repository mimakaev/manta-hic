# Hi-C target storage: the banded format

How Manta's Hi-C *training targets* are stored and fetched. This is the data the model is trained against
(loss) and compared to for display — it does **not** touch the model weights, so the storage format can be
changed freely as long as it reconstructs the same `(hic, weight, exp)` triple the loss consumes.

Read side + sampling: `manta_hic/io/banded.py` (`BandedHicStore`/`BandedHicFile`), tested in
`tests/test_banded.py` and `tests/test_banded_file.py`. Write side: `manta_hic/io/banded_write.py`
(`coolers_to_banded`), tested in `tests/test_banded_write.py`.

## Band layout

Store the first `n_diag` diagonals of each chromosome's contact map densely, **once**, gap-free:

```
band[c, x, d] = M_c[x, x + d]      for d in 0..n_diag-1   (n_diag = n_bins = 1024)
```

A window `[a, a+n)` is reconstructed by symmetry: `M_win[i,j] = band[a + min(i,j), |i−j|]` (a gather;
`n_diag ≥ n` required). Compared to storing overlapping dense tiles this is ~6× smaller (each contact is
stored once, only within the band) and has no gaps, so display / `get_slice_by_coords` for an arbitrary
locus always succeeds.

Stored per chromosome (HDF5 group), gap-free over the whole chromosome:
- `band` `[C, n_bins, n_diag]` int16 — raw counts (clip 32000);
- `weights` `[C, n_bins]` float32 — per-bin balancing weight (bad bins → 0);
- `bad` `[C, n_bins]` bool — per-channel bad mask;
- `arm_id` `[n_bins]` int32 — chromosomal-arm id (`-1` for chrM/excluded);
- `fold_id` `[n_bins]` int32 — Borzoi fold per bin (`-1` none);
- `exp` `[C, n_arms, n_diag]` float32 — per-arm per-distance expected (first 2 diagonals zeroed).

## Tile selection is a sampling-time policy

There is no baked-in tiling: a start `a` is eligible iff over `[a, a+n)` the `arm_id` is constant (no
centromere/arm/chrom-end crossing), the `fold_id` is constant (and the desired fold), and the RMS-over-
channels windowed mean of `bad` is `< max_bad_fraction`. All three are **1-D prefix-sum window tests**
(`arm`/`fold`: zero change-points inside the window; `bad`: cumsum windowed mean), so a threshold change
(`BandedHicStore.eligible_starts(max_bad_fraction=…)`) is free — no re-tiling, no re-writing.

**Target pipeline**: reconstruct the `n×n` count square from `band`, slice `weights[:, a:a+n]`, take
`exp[:, arm, :n]` → feed the existing `create_expected_matrix(hic, weight, exp)` → `(observed,
expected_matrix)`, OOE = obs/exp, then `adaptive_coarsegrain_torch` for display/correlation. Reconstructing
the square is a gather (~ms, GPU-able since the fetcher is torch/on-device); at 256 bp the band is ~24
GB/channel so it is kept HDF5-chunked (`[a:a+n, :]` reads), not all-in-RAM.

## Write side — `coolers_to_banded`

Source coolers (one per channel, sharing a bin grid) → one self-contained banded HDF5. The band reconstructs
`cooler.matrix(...).fetch` **exactly** (`max|diff|=0`).

### Decisions
- **Pixels, not `cooler.fetch`.** Cooler pixels are sparse and upper-triangular; read one chromosome's
  pixels, keep `d = bin2−bin1 < n_diag`, and scatter into `band[bin1_local, d]`. This reads only what the
  band needs (a `fetch` would materialise dense squares) and is provably exact. (`chrom_band_from_pixels`)
- **Expected per arm**, not per window: `exp[C, n_arms, n_diag]` — the same vector for every window in an arm.
- **Raw counts in the band** (not zeroed at bad bins): bad bins are masked in the loss via their zeroed
  *weights* → zeroed expected, so keeping raw counts costs nothing and makes display show real data.

### The file is autonomous (no external manifest)
Provenance is stored *in* the file, so nothing external is needed to train or plot:
`provenance/{uris, shortnames, sizes, nnz, sum}` (cooler `sum`+`nnz` identify a cooler — no hashing),
`chroms/{name, length}`, `arms/{name, chrom, start, end}`, and attrs `{genome, resolution, n_diag,
group_name, n_channels}`. A reader gets channel shortnames, coordinates, and source identity from the file
alone.

### Bad bins (`compute_bad_bins`)
A histogram-peak coverage filter. Cooler weight ≈ 1/√coverage, so well-covered bins form a sharp peak in the
log-binned weight histogram and low-coverage bins are a high-weight tail. It keeps the contiguous band around
the peak (while per-bin count ≥ an adaptive `n_bins/30000` cutoff), then additionally flags `weight>0.1`
below an exponentially-rising count requirement (~50× stricter by `weight=0.3`) and `weight>0.3` outright;
chrY and non-finite weights are bad. Deliberately a bit stringent.

The eligibility gate measures the bad fraction over the exact `n_bins` tile (a sampling-time knob), while the
per-*bin* bad definition is fixed. Validated on 4dn-diff (5 channels) @ 1024 bp across all 23 chromosomes
(11,458 step-256 candidates): the tile gate accepts 86.1% at `frac<0.1`, agreeing with a 1.25×-window gate to
98.6% (99.3% of the wider gate's accepts are also accepted here).

## Output HDF5 layout
```
/                         attrs: format, genome, resolution, n_diag, group_name, n_channels
/provenance/{uris, shortnames, sizes, nnz, sum}
/chroms/{name, length}
/arms/{name, chrom, start, end}
/exp                      [C, n_arms, n_diag]  per-arm per-distance expected (first 2 diagonals zeroed)
/<chrom>/band             [C, n_bins, n_diag]  int16 raw counts, band[c,x,d] = M_c[x, x+d]
/<chrom>/{weights,bad}    [C, n_bins]          weight (bad→0), bool bad mask
/<chrom>/{arm_id,fold_id} [n_bins]             per-bin arm id (-1 excluded) / Borzoi fold id (-1 none)
```
