# Hi-C target storage: current approach and the banded redesign

How Manta's Hi-C *training targets* are stored and fetched. This is the data the model is trained against
(loss) and compared to for display — it does **not** touch the model weights, so the storage format can be
changed freely as long as it reconstructs the same `(hic, weight, exp)` triple the loss consumes.

## Current approach (overlapping square tiles) — `io/cool_io.py` + `nn/manta.py:HiCDataset`

For publication, the pipeline as of 2026-06-30:

### Build (`save_coolers_for_manta` / `process_mcools`)
Per dataset (group of coolers sharing a `group_name`) and per resolution:

1. **Arms.** `bioframe.make_chromarms(chromsizes, centromeres)` (or whole-chrom for mm10); drop `chrM`.
2. **Bad bins** (`get_bad_bin_masks`, per cooler): a bin is *bad* if its balancing weight is non-finite,
   falls outside the log-binned main peak (±an adaptive count cutoff that scales with resolution via
   `len(weights)/30000`), or `weight>0.1` below an adaptive count curve, or `weight>0.3` (≲10 reads).
   `chrY` is skipped → its bins are all marked bad.
3. **Candidate windows.** Tile each chromosome from 0 in steps of `step_bins·res`, window size
   `actual_size = target_size + step_bins` bins (= 1.25·n_bins; the extra `step_bins` is overhang for the
   stochastic crop). Only windows with `start ≤ chrom_len − actual_size·res` (must fully fit).
4. **Window inclusion filters:**
   - window **fully inside one arm** (else skipped — this is what creates centromere/arm-boundary gaps);
   - **RMS over coolers** of the per-cooler mean bad-bin fraction in the window `< min_fraction` (0.1).
5. **Expected.** `cooltools.expected_cis(view_df=arms, smooth=True)` per arm; keep `dist < actual_size`.
6. **Per-window storage** (HDF5, one file `…/{group}_{res}.h5`), datasets indexed by window:
   - `hic`  `[n_win, C, actual_size, actual_size]` int16 — raw counts, clipped at 32000, bad rows/cols zeroed, lzf-compressed;
   - `weights` `[n_win, C, actual_size]` float32 — per-bin balancing weight, bad bins zeroed;
   - `exp` `[n_win, C, actual_size]` float32 — per-distance expected (`balanced.avg.smoothed`), first 2 diagonals zeroed;
   - `chrom`, `start`, `end` — window coordinates.

### Fetch (`HiCDataset`)
Holds 1.25·n_bins squares saved every 0.25·n_bins, so any `n_bins×n_bins` map is a crop of a stored
square (stochastic offset). `assign_fold_type` (criterion 6) labels each window train/val/test/**discard**
by overlap (≥0.8) with the Borzoi fold intervals — windows straddling folds are discarded.
`get_slice_by_index`/`get_slice_by_coords` return `{hic_slice, weight_slice, exp, …}`; `run_epoch` builds
the target via `create_expected_matrix(hic, weight, exp)` → `(observed, expected_matrix)`, OOE = obs/exp,
and `adaptive_coarsegrain_torch` for display/correlation.

### Inclusion criteria (complete)
1. `chrM` excluded. 2. `chrY` excluded (all-bad). 3. Window fits in chromosome. 4. Window fully within one
arm (centromere gap). 5. RMS-over-coolers bad-bin fraction `< 0.1`. 6. Single Borzoi fold (≥0.8 overlap).

### Pain points
- **Giant files**: overlapping squares store each contact ~5× (overlap) and the full square, not just the
  band ⇒ ≈ `L × 6400` values/channel vs the `L × n_diag` actually needed (~6× bloat).
- **Gaps**: criteria 3/4 (and discards) mean large genomic stretches have **no** stored tile, so
  `get_slice_by_coords` raises "Region not found" — display/inference for arbitrary loci fails.
- Inclusion is baked in at build time; changing a threshold means re-tiling and re-writing everything.

## Banded ("turned") redesign

Store the first `n_diag` diagonals of each chromosome densely, **once**, gap-free:

```
band[c, x, d] = M_c[x, x + d]      for d in 0..n_diag-1   (n_diag = n_bins = 1024)
```

This is exactly Max's `OnDiagonalHicAggregator` layout (`save_mcools/save_to_coolers_prototype.ipynb`),
used there to aggregate *predictions* into coolers; here it is the *target* store. A window `[a, a+n)` is
reconstructed by symmetry: `M_win[i,j] = band[a + min(i,j), |i−j|]` (a gather; `n_diag ≥ n` required).

Stored per chromosome (HDF5 group), all gap-free over the whole chromosome:
- `band` `[C, n_bins, n_diag]` int16 — the counts (clip 32000 as now);
- `weights` `[C, n_bins]` float32 — per-bin balancing weight (bad bins → 0);
- `bad` `[C, n_bins]` bool — per-channel bad mask (criterion 5 ingredient);
- `arm_id` `[n_bins]` int32 — chromosomal-arm id (criteria 1,3,4); `-1` for chrM/excluded;
- `fold_id` `[n_bins]` int32 — Borzoi fold per bin (criterion 6);
- `exp` `[C, n_arms, n_diag]` float32 — per-arm per-distance expected (first 2 diagonals zeroed).

**Tile selection becomes a sampling-time policy** (no re-tiling): a start `a` is eligible iff over `[a,a+n)`
the `arm_id` is constant (no centromere/arm/chrom-end crossing), the `fold_id` is constant (and the desired
fold), and the RMS-over-channels windowed mean of `bad` is `< min_fraction`. All three are **1-D prefix-sum
window tests** (`arm`/`fold`: zero change-points inside the window; `bad`: cumsum windowed mean). At 256 bp
that's ~12M bins × a few int32/bool vectors — cheap to hold and recompute.

**Target pipeline unchanged**: reconstruct the `n×n` count square from `band`, slice `weights[:, a:a+n]`,
take `exp[:, arm, :n]` → feed the existing `create_expected_matrix` verbatim. (`exp` being per-arm and
per-distance is now stored once per arm instead of once per window.)

### Wins
~6× smaller, no gaps, display/`get_slice_by_coords` always succeeds, arbitrary (sub-`step_bins`) offsets,
and inclusion thresholds become cheap sampling-time knobs rather than baked-in build decisions.

### Costs / notes
- Per fetch: reconstruct the square (a gather) instead of a whole-matrix read — ~ms, and GPU-able now that
  the fetcher is torch/on-device.
- At 256 bp the band is ~24 GB/channel ⇒ keep it HDF5-chunked (`[a:a+n, :]` reads), not all-in-RAM.
- `HiCDataset` keeps its interface; only its backend (and `save_coolers_for_manta`) change.

Prototype of the read side + selection: `manta_hic/io/banded.py`, tested in `tests/test_banded.py`
(round-trip square↔band, eligibility vs brute force, and `create_expected_matrix` parity).

## Write side — `io/banded_write.py` (`coolers_to_banded`)

Source coolers (one per channel, sharing a bin grid) → one self-contained banded HDF5. Prototyped on
**4dn-diff @ 16384** (5 channels): the band reconstructs `cooler.matrix(...).fetch` **exactly**
(`max|diff|=0`); conversion is ~70 s for 4 chroms (dominated by the whole-genome `cooltools.expected_cis`;
the pixel→band step is sub-second per chrom). Self-contained write-side test on a synthetic cooler:
`tests/test_banded_write.py`.

### Decisions
- **Pixels, not `cooler.fetch`.** Cooler pixels are sparse and upper-triangular; we read one chromosome's
  pixels, keep `d = bin2−bin1 < n_diag`, and scatter into `band[bin1_local, d]`. This reads only what the
  band needs (a `fetch` would materialise dense squares) and is provably exact. (`chrom_band_from_pixels`)
- **Expected per arm**, not per window: `exp[C, n_arms, n_diag]` (the same vector for every window in an
  arm — the old per-window storage duplicated it thousands of times).
- **Raw counts in the band** (not zeroed at bad bins): bad bins are masked in the loss via their zeroed
  *weights* → zeroed expected, so keeping raw counts costs nothing and makes display show real data.

### Manifest elimination — the file is autonomous
Provenance is stored *in* the file, so nothing external is needed to train or plot:
`provenance/{uris, shortnames, sizes, nnz, sum}` (cooler `sum`+`nnz` identify a cooler -- no hashing),
`chroms/{name, length}`, `arms/{name, chrom, start, end}`,
and attrs `{genome, resolution, n_diag, group_name, n_channels}`. A reader gets channel shortnames,
coordinates, and source identity from the file alone — the CSV manifest is no longer required at
train/plot time (it remains only an *input* list for the conversion driver).

### Bad bins — "what did I invent" (`compute_bad_bins`)
A histogram-peak coverage filter (cleaned re-implementation of `cool_io.get_bad_bin_masks`, same
behaviour). Cooler weight ≈ 1/√coverage, so well-covered bins form a sharp peak in the log-binned weight
histogram and low-coverage bins are a high-weight tail. It keeps the contiguous band around the peak (while
per-bin count ≥ an adaptive `n_bins/30000` cutoff), then additionally flags `weight>0.1` below an
exponentially-rising count requirement (~50× stricter by `weight=0.3`) and `weight>0.3` outright; chrY and
non-finite weights are bad. It is deliberately a bit stringent — that's the "creative" part.

### Threshold relaxation — and validation against the old gate
The old per-window gate measured the bad fraction over the **1.25× n_bins** window; the new gate is exact
per *n_bins* tile and is a sampling-time knob (`BandedHicStore.eligible_starts(min_fraction=…)`), so it can
be changed without re-tiling. The per-*bin* bad definition is unchanged.

**Validated** on 4dn-diff (5 channels) **@ 1024 bp** across all 23 chromosomes (11,458 step-256
candidates): OLD (1280-win, frac<0.1) accepted 9,837 (85.9%); NEW (1024-win, frac<0.1) accepted 9,870
(86.1%) — **ratio 1.003, 98.6% agreement, 99.3% of old-accepted also new-accepted**. So the new gate
rejects essentially the same windows (not ~5× more, not everything through). The "relax slightly" intuition
nets to a wash: the 1024 window drops the old extra 256 edge bins, which are often *more* bad, so 0.1 is
already a match (0.125 → +2%).

### Output HDF5 layout
```
/                         attrs: format, genome, resolution, n_diag, group_name, n_channels
/provenance/{uris, shortnames, sizes, nnz, sum}
/chroms/{name, length}
/arms/{name, chrom, start, end}
/exp                      [C, n_arms, n_diag]  per-arm per-distance expected (first 2 diagonals zeroed)
/<chrom>/band             [C, n_bins, n_diag]  int16 raw counts (lzf), band[c,x,d] = M_c[x, x+d]
/<chrom>/{weights,bad}    [C, n_bins]          weight (bad→0), bool bad mask
/<chrom>/{arm_id,fold_id} [n_bins]             per-bin arm id (-1 excluded) / Borzoi fold id (-1 none)
```

### Left for the un-sandboxed en-masse rebuild
Run `coolers_to_banded` over every dataset/resolution from the manifest (it's a thin driver loop); wire
`HiCDataset` onto `BandedHicStore` (sample a chrom → torch
window reconstruction on-device). At fine resolutions `expected_cis` and pixel reads dominate — embarrassingly
parallel across datasets/chroms.
