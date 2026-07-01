"""
Write side of the banded ("turned") Hi-C store: source coolers -> a self-contained banded HDF5.

One output file per dataset (a group of coolers sharing a bin grid) and resolution. It stores, gap-free
over the whole genome:

- per chromosome: ``band`` ``[C, n_bins, n_diag]`` raw counts (band[c,x,d] = M_c[x, x+d]), ``weights``
  ``[C, n_bins]`` (balancing weight, bad bins -> 0), ``bad`` ``[C, n_bins]`` bool, ``arm_id`` / ``fold_id``;
- ``exp`` ``[C, n_arms, n_diag]`` per-arm per-distance expected (first 2 diagonals zeroed);
- provenance so the file is autonomous (no manifest needed): source URIs, channel shortnames, file sizes,
  cooler ``sum`` + ``nnz`` (which together identify a cooler -- no hashing needed), genome / resolution /
  arms / chromsizes.

The band stores RAW counts (not zeroed at bad bins): the loss masks bad bins via the zeroed weights ->
expected, and keeping raw counts means display shows real data everywhere. Tile eligibility is computed at
sampling time from ``bad``/``arm_id``/``fold_id`` (see ``banded.BandedHicStore.eligible_starts``), so the
window bad-fraction threshold is a knob, not baked in.

See docs/HIC_STORAGE.md. The build side reads real coolers + cooltools, so it is meant to run un-sandboxed
for the full en-masse conversion; this module is the (tested) reference, prototyped on 4dn-diff @ 16384.
"""

import os

import bioframe
import cooler
import cooltools
import h5py
import numpy as np
import pandas as pd

from manta_hic.training_meta import fold_df

COUNT_CLIP = 32000  # clip raw counts to fit int16 (matches cool_io)


# --------------------------------------------------------------------------- #
# Bad-bin masking -- "what did I invent": a histogram-peak coverage filter      #
# --------------------------------------------------------------------------- #
def compute_bad_bins(clr, *, cov_cutoff_div=30000.0):
    """
    Per-bin "bad" mask for one cooler (low coverage / weight outliers). Cleaned re-implementation of the
    original ``cool_io.get_bad_bin_masks`` with the same behaviour.

    The idea (the "creative" bit): cooler balancing weight ~ 1/sqrt(coverage), so well-covered bins form a
    sharp peak in the (log-binned) weight histogram and poorly-covered bins are a high-weight tail. We:
      1. find the histogram peak and keep the contiguous band around it whose per-bin count stays above an
         adaptive ``cutoff = n_bins / cov_cutoff_div`` (so the cutoff scales with resolution);
      2. additionally flag ``weight > 0.1`` bins below an exponentially-rising count requirement
         (``cutoff * exp((w-0.1)*5*ln(50))`` -- ~50x stricter by w=0.3), and ``weight > 0.3`` always
         (those are bins with ~<10 reads);
      3. non-finite weights and all of chrY are bad.

    Returns a boolean mask in full cooler bin order.
    """
    bins_all = cooler.Cooler(clr).bins()[:] if isinstance(clr, str) else clr.bins()[:]
    test = bins_all[(bins_all["chrom"] != "chrY") & np.isfinite(bins_all["weight"])].copy()
    weights = test["weight"].values

    hbins = np.logspace(-4, 0, 40)
    cutoff = len(weights) / cov_cutoff_div
    test["wcat"] = pd.cut(test["weight"], hbins)
    counts = test.groupby("wcat", observed=True)["chrom"].count().rename("count")

    # contiguous band around the histogram peak, expanding while count >= cutoff
    peak = counts.idxmax()
    left = right = counts.index.get_loc(peak)
    while left > 0 and counts.iloc[left - 1] >= cutoff:
        left -= 1
    while right < len(counts) - 1 and counts.iloc[right + 1] >= cutoff:
        right += 1
    kept_cats = set(counts.iloc[left : right + 1].index.tolist())

    test = test.join(counts, on="wcat")
    bad_test = (
        (~test["wcat"].isin(kept_cats))
        | ((test["weight"] > 0.1) & (test["count"] < cutoff * np.exp((test["weight"] - 0.1) * 5 * np.log(50))))
        | (test["weight"] > 0.3)
    )

    bad = np.ones(len(bins_all), dtype=bool)  # default bad (covers chrY, non-finite, missing)
    bad[test.index.values] = bad_test.values
    return bad


# --------------------------------------------------------------------------- #
# Per-chromosome band from cooler pixels                                       #
# --------------------------------------------------------------------------- #
def chrom_band_from_pixels(clr, chrom, n_diag):
    """
    Build the raw-count band ``[n_bins, n_diag]`` for one chromosome straight from cooler pixels.

    Pixels are sparse and upper-triangular (bin1 <= bin2), so we read only this chromosome's pixels, keep
    ``d = bin2 - bin1 < n_diag``, and scatter into ``band[bin1_local, d]``. Far cheaper than fetching dense
    squares. Counts are clipped to int16 range.
    """
    lo, hi = clr.extent(chrom)
    n_bins = hi - lo
    px = clr.matrix(balance=False, as_pixels=True, join=False).fetch(chrom)
    b1 = px["bin1_id"].values - lo
    d = px["bin2_id"].values - px["bin1_id"].values
    keep = d < n_diag
    band = np.zeros((n_bins, n_diag), dtype=np.int32)
    band[b1[keep], d[keep]] = px["count"].values[keep]
    np.clip(band, 0, COUNT_CLIP, out=band)
    return band.astype(np.int16)


# --------------------------------------------------------------------------- #
# Per-arm expected, and per-bin arm / fold ids                                 #
# --------------------------------------------------------------------------- #
def chromarms(genome):
    """Chromosomal arms as a cooltools view frame (whole-chrom for mm10, which lacks centromeres here)."""
    chromsizes = bioframe.fetch_chromsizes(genome)
    if genome in ("mm10",):
        arms = pd.DataFrame(
            {"chrom": chromsizes.index, "start": 0, "end": chromsizes.values, "name": chromsizes.index + "_full"}
        )
    else:
        arms = bioframe.make_chromarms(chromsizes, bioframe.fetch_centromeres(genome))
    arms = arms[arms["chrom"] != "chrM"].reset_index(drop=True)
    arms.columns = ["chrom", "start", "end", "name"][: arms.shape[1]]
    return arms


def expected_per_arm(coolers, arms, n_diag, nproc=8):
    """``[C, n_arms, n_diag]`` per-arm per-distance expected (balanced.avg.smoothed), first 2 diags zeroed."""
    name_to_idx = {n: i for i, n in enumerate(arms["name"])}
    exp = np.zeros((len(coolers), len(arms), n_diag), dtype=np.float32)
    for ci, clr in enumerate(coolers):
        cvd = cooltools.expected_cis(clr=clr, view_df=arms, smooth=True, aggregate_smoothed=False, nproc=nproc)
        cvd = cvd[cvd["dist"] < n_diag]
        for name, g in cvd.groupby("region1"):
            if name in name_to_idx:
                exp[ci, name_to_idx[name], g["dist"].values] = np.nan_to_num(g["balanced.avg.smoothed"].values)
        exp[ci, :, :2] = 0.0
    return exp


def assign_arm_fold_ids(bins, arms, genome):
    """Per-bin ``arm_id`` (-1 = chrM/excluded) and ``fold_id`` (-1 = none) from arms + Borzoi folds."""
    chroms = bins["chrom"].astype(str).values
    mid = (bins["start"].values + bins["end"].values) // 2
    arm_id = np.full(len(bins), -1, dtype=np.int32)
    for ai, a in arms.iterrows():
        arm_id[(chroms == a["chrom"]) & (mid >= a["start"]) & (mid < a["end"])] = ai

    fold_id = np.full(len(bins), -1, dtype=np.int32)
    for f in fold_df.filter(fold_df["genome"] == genome).iter_rows(named=True):  # polars, no pyarrow
        m = (chroms == f["chrom"]) & (mid >= f["start"]) & (mid < f["end"])
        fold_id[m] = int(str(f["fold"]).replace("fold", ""))
    return arm_id, fold_id


def zero_bad_in_weights(weights, bad):
    """Zero the balancing weight at bad bins (this is what masks them in the loss/expected)."""
    w = weights.copy()
    w[bad] = 0.0
    return w


def eligible_start_fraction(bad, arm_id, n, *, min_fraction=0.1):
    """
    Fraction of length-``n`` window starts that pass the arm-containment + RMS bad-fraction gate -- the
    fold-agnostic case of :meth:`banded.BandedHicStore.eligible_starts`, and *the* number to watch across a
    conversion: the share of candidate training windows that survive coverage filtering. A sudden drop vs
    other resolutions/datasets means a coverage/balancing problem.

    Operates on genome-wide concatenated per-bin arrays (``bad`` ``[C, total]``, ``arm_id`` ``[total]``);
    ``arm_id`` changes at every arm/chromosome boundary, so no accepted window spans a boundary.

    Returns ``(fraction_of_in_arm_windows, n_eligible, n_candidate_in_arm)``.
    """
    C, total = bad.shape
    if n > total:
        return 0.0, 0, 0
    a = np.arange(total - n + 1)
    arm_changes = np.concatenate([[0], np.cumsum(arm_id[1:] != arm_id[:-1])])
    arm_ok = (arm_changes[a + n - 1] - arm_changes[a] == 0) & (arm_id[a] != -1)
    bad_cum = np.concatenate([np.zeros((C, 1)), np.cumsum(bad.astype(np.float64), axis=1)], axis=1)
    win_mean = (bad_cum[:, a + n] - bad_cum[:, a]) / n
    bad_ok = np.sqrt((win_mean**2).mean(axis=0)) < min_fraction
    n_cand = int(arm_ok.sum())
    n_elig = int((arm_ok & bad_ok).sum())
    return (n_elig / n_cand if n_cand else 0.0), n_elig, n_cand


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def _file_provenance(uris, coolers):
    """Source identity: file size, plus the cooler's ``sum`` and ``nnz`` -- together these pin a cooler
    (its total contacts and number of non-zero pixels), and both are in ``cooler.info`` (no hashing needed)."""
    sizes = [os.path.getsize(u.split("::")[0]) if os.path.exists(u.split("::")[0]) else -1 for u in uris]
    nnz = [int(c.info["nnz"]) for c in coolers]
    sums = [int(c.info["sum"]) for c in coolers]
    return sizes, nnz, sums


def coolers_to_banded(
    cooler_uris,
    shortnames,
    output_path,
    *,
    genome,
    resolution,
    group_name,
    n_diag=1024,
    chroms=None,
    nproc=8,
    dry_run=False,
):
    """
    Convert a group of cooler URIs (same bin grid) into one self-contained banded HDF5 file.

    Parameters
    ----------
    cooler_uris : list[str]
        Cooler URIs, e.g. ``["x.mcool::resolutions/16384", ...]`` -- one per channel.
    shortnames : list[str]
        Channel shortnames (e.g. from the manifest ``name`` column), stored for autonomous plotting.
    output_path : str
        Output ``.bhic.h5`` path.
    genome, resolution, group_name : str, int, str
        Provenance / config recorded in the file.
    n_diag : int
        Diagonals to store (= model ``n_bins``; default 1024).
    chroms : list[str] | None
        Restrict to these chromosomes (default: all in the cooler that have an arm).
    dry_run : bool
        If True, do only the cheap parts -- open every cooler (which validates the resolution exists),
        check the bin grids match, resolve arms / chromosome selection, and gather source sizes -- then
        return a summary dict **without** computing expected, bands, or bad-bin masks and without writing
        any file. Use it to validate a whole en-masse plan in seconds before the long run.

    Returns
    -------
    str | dict
        The ``output_path`` on a real run; a summary dict on a ``dry_run`` (channels, chroms kept/dropped,
        total bins, arm count, estimated band size, source sizes).
    """
    coolers = [cooler.Cooler(u) for u in cooler_uris]  # opening validates the resolution exists
    C = len(coolers)
    assert all(c.binsize == resolution for c in coolers), "resolution mismatch"
    if len(shortnames) != C:
        raise ValueError(f"{len(shortnames)} shortnames for {C} coolers")
    # All channels must share one bin grid: weights/bad are sliced with coolers[0]'s offsets, so a
    # differing grid would silently misalign per-bin data across channels. Identical chromsizes +
    # identical binsize => identical cooler bin tables, so this guards the whole grid.
    ref_sizes = coolers[0].chromsizes
    for u, c in zip(cooler_uris, coolers):
        if not c.chromsizes.equals(ref_sizes):
            raise ValueError(
                f"bin grid mismatch: {u} has different chromsizes than {cooler_uris[0]}; all coolers in a "
                "group must share one bin grid (per-bin weights/bad are indexed with the first cooler's offsets)."
            )
    arms = chromarms(genome)
    arm_chroms = set(arms["chrom"])
    use_chroms = chroms if chroms is not None else [c for c in coolers[0].chromnames if c in arm_chroms]
    dropped = [c for c in coolers[0].chromnames if c not in set(use_chroms)]
    sizes, nnz, sums = _file_provenance(cooler_uris, coolers)

    if dry_run:
        chrom_bins = {c: int(coolers[0].extent(c)[1] - coolers[0].extent(c)[0]) for c in use_chroms}
        total_bins = sum(chrom_bins.values())
        return dict(
            output_path=output_path,
            genome=genome,
            resolution=resolution,
            group_name=group_name,
            n_channels=C,
            shortnames=list(shortnames),
            uris=list(cooler_uris),
            n_arms=len(arms),
            chroms_used=list(use_chroms),
            chroms_dropped=dropped,  # no arm (e.g. chrM, scaffolds)
            total_bins=total_bins,
            est_band_bytes=total_bins * C * n_diag * 2,  # int16 band, gap-free
            source_sizes=sizes,
            source_nnz=nnz,
            source_sum=sums,
        )

    bins = coolers[0].bins()[:]
    arm_id_all, fold_id_all = assign_arm_fold_ids(bins, arms, genome)

    bad_all = np.stack([compute_bad_bins(c) for c in coolers])  # [C, total_bins]
    weight_all = np.stack([np.nan_to_num(c.bins()["weight"][:]).astype(np.float32) for c in coolers])
    weight_all = np.stack([zero_bad_in_weights(weight_all[i], bad_all[i]) for i in range(C)])
    exp = expected_per_arm(coolers, arms, n_diag, nproc=nproc)

    # The watch metric: fraction of candidate n_diag-windows that survive coverage filtering. Persisted in
    # the file (autonomous) so a conversion can be audited afterwards without recomputing.
    accept_frac, n_elig, n_cand = eligible_start_fraction(bad_all, arm_id_all, n_diag)

    str_dt = h5py.string_dtype()

    with h5py.File(output_path, "w") as f:
        f.attrs.update(
            dict(
                format="manta-banded-v1",
                genome=genome,
                resolution=resolution,
                n_diag=n_diag,
                group_name=group_name,
                n_channels=C,
                accepted_fraction=accept_frac,
                n_eligible=n_elig,
                n_candidate=n_cand,
                accept_min_fraction=0.1,
            )
        )
        prov = f.create_group("provenance")
        prov.create_dataset("uris", data=np.array(cooler_uris, dtype=object), dtype=str_dt)
        prov.create_dataset("shortnames", data=np.array(list(shortnames), dtype=object), dtype=str_dt)
        prov.create_dataset("sizes", data=np.array(sizes, dtype=np.int64))
        prov.create_dataset("nnz", data=np.array(nnz, dtype=np.int64))
        prov.create_dataset("sum", data=np.array(sums, dtype=np.int64))

        ch = f.create_group("chroms")  # for autonomous plotting: name -> length (bp)
        ch.create_dataset("name", data=np.array(coolers[0].chromnames, dtype=object), dtype=str_dt)
        ch.create_dataset("length", data=coolers[0].chromsizes.values.astype(np.int64))

        av = f.create_group("arms")
        av.create_dataset("name", data=np.array(arms["name"].tolist(), dtype=object), dtype=str_dt)
        av.create_dataset("chrom", data=np.array(arms["chrom"].tolist(), dtype=object), dtype=str_dt)
        av.create_dataset("start", data=arms["start"].values.astype(np.int64))
        av.create_dataset("end", data=arms["end"].values.astype(np.int64))
        f.create_dataset("exp", data=exp)

        for chrom in use_chroms:
            lo, hi = coolers[0].extent(chrom)
            g = f.create_group(chrom)
            band = np.stack([chrom_band_from_pixels(c, chrom, n_diag) for c in coolers])  # [C, nb, n_diag]
            g.create_dataset("band", data=band, compression="lzf", chunks=(1, min(2048, band.shape[1]), n_diag))
            g.create_dataset("weights", data=weight_all[:, lo:hi])
            g.create_dataset("bad", data=bad_all[:, lo:hi])
            g.create_dataset("arm_id", data=arm_id_all[lo:hi])
            g.create_dataset("fold_id", data=fold_id_all[lo:hi])

    return output_path
