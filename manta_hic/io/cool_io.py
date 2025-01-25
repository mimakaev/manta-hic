# ------------------------------------
# Part 2: The function doing the cooler -> HDF5 conversion
# ------------------------------------
import os

import numpy as np
import pandas as pd
import cooler
import h5py
import bioframe
import cooltools


def get_bad_bin_masks(uri):
    """
    Get a boolean mask of "bad" bins in a cooler file.
    We define "bad" bins as those with low coverage, or those that are outliers in the distribution of weights.

    Parameters
    ----------
    uri : str
        URI of the cooler file, e.g. "x.mcool::resolutions/1024".

    Returns
    -------
    np.ndarray
        Boolean mask of "bad" bins. Same length as the number of bins in the cooler.

    Notes
    -----
    * Chromosome Y is skipped.
    * Weights are log-binned and the main peak is kept, with surrounding bins up to a cutoff.
    * At low weights, we use an adaptive cutoff between 0.1 and 0.3.
    * Weights above 0.3 are considered "bad" no matter what (it is really bins with <10 reads, etc).
    """
    clr = cooler.Cooler(uri)
    bins_all = clr.bins()[:]
    # skip NaNs, chrY, etc.
    bins_test = bins_all[bins_all["chrom"] != "chrY"]
    bins_test = bins_test[np.isfinite(bins_test["weight"])]
    weights = bins_test["weight"].values

    # Weighted histogram approach
    hbins = np.logspace(-4, 0, 40)
    cutoff = len(weights) / 30000.0  # empirically found cutoff that adapts to cooler resolution
    bins_test["weight_cat"] = pd.cut(bins_test["weight"], hbins)
    counts = bins_test.groupby("weight_cat", observed=True)["chrom"].count().rename("count")

    # Finding the main peak that we want to keep, and the surrounding bins up to cutoff
    peak = counts.idxmax()
    left = right = counts.index.get_loc(peak)
    while left > 0 and counts.iloc[left - 1] >= cutoff:
        left -= 1
    while right < len(counts) - 1 and counts.iloc[right + 1] >= cutoff:
        right += 1
    selected_weight_cat = counts.iloc[left : right + 1].index.tolist()

    # At low high weights (low coverage) we use adaptive cutoff between weights of 0.1 and 0.3
    bins_test = bins_test.join(counts, on="weight_cat")
    bad = (
        (~bins_test["weight_cat"].isin(selected_weight_cat))
        | (
            (bins_test["weight"] > 0.1)
            & (bins_test["count"] < cutoff * np.exp((bins_test["weight"] - 0.1) * 5 * np.log(50)))
        )
        | (bins_test["weight"] > 0.3)
    )
    # put it back in full cooler order
    bins_all["bad"] = True
    idx_common = bins_all.index.intersection(bins_test.index)
    bins_all.loc[idx_common, "bad"] = bad.loc[idx_common]
    return bins_all["bad"].values


def save_coolers_for_manta(
    cooler_uris,
    output_filename,
    genome,
    resolution,
    target_size,
    step_bins,
):
    """
    Convert one or more .cool / .mcool URIs at a given resolution into an HDF5 file
    with datasets: 'hic', 'weights', 'exp', plus region coordinates.

    Parameters
    ----------
    cooler_uris : list of str
        List of cooler URIs, e.g. ["x.mcool::resolutions/1024", "y.mcool::resolutions/1024"].
    output_filename : str
        The .h5 file to save outputs into.
    genome : str
        E.g. 'hg38', 'mm10', etc. Will be used to fetch chromsizes and centromeres.
    resolution : int
        Resolution in bp, e.g. 1000, 1024, etc.
    target_size : int
        Size of the final window in bins. (We actually fetch bigger internally if needed.)
    step_bins : int
        Step size in bins, used for generating tiled windows across the genome.

    Returns
    -------
    None
        Saves the results into the specified output_filename.

    Notes
    -----
    * We use cooltools to compute expected values.
    * We use get_bad_bin_masks to identify "bad" bins in the coolers.
    * We skip chromosome Y.
    * We skip windows that have >10% "bad" bins on average, but using quadratic mean (closer to max).

    """

    # We fetch a bigger window to accommodate the model's step usage
    actual_size = target_size + step_bins

    # Read chromsizes; fetch arms if we have them, else full-chrom.
    chromsizes = bioframe.fetch_chromsizes(genome)
    if genome not in ["mm10"]:
        cens = bioframe.fetch_centromeres(genome)
        arms = bioframe.make_chromarms(chromsizes, cens)
    else:
        arms = pd.DataFrame(
            {
                "chrom": chromsizes.index,
                "start": [0] * len(chromsizes),
                "end": chromsizes.values,
                "name": [c + "_full" for c in chromsizes.index],
            }
        )
    arms = arms.reset_index(drop=True)

    # Build a dict of "bad bin masks" for each cooler
    bad_bin_masks = {}
    for uri in cooler_uris:
        bad_bin_masks[uri] = get_bad_bin_masks(uri)

    # Build windows that tile the genome in increments of step_bins
    windows = []
    for chrom in chromsizes.index:
        chrom_len = chromsizes[chrom]
        max_start = chrom_len - actual_size * resolution
        start_ = 0
        while start_ <= max_start:
            windows.append((chrom, start_, start_ + actual_size * resolution))
            start_ += step_bins * resolution
    window_df = pd.DataFrame(windows, columns=["chrom", "start", "end"])

    # Open all coolers
    all_coolers = [cooler.Cooler(uri) for uri in cooler_uris]

    # compute expected using cooltools
    cvd_list = [
        cooltools.expected_cis(clr=c, view_df=arms, smooth=True, aggregate_smoothed=False, nproc=20)
        for c in all_coolers
    ]
    # Keep only distances < actual_size (to match maximum snippet size)
    cvd_list = [i[i["dist"] < actual_size].copy() for i in cvd_list]
    # Merge arms info for quick grouping
    cvd_list = [i.join(arms.set_index("name"), on="region1") for i in cvd_list]
    # Build groupby objects on (chrom, start, end) -> region, for quick lookups
    expdfs = [exp.groupby(["chrom", "start", "end"]) for exp in cvd_list]
    expgroups = pd.DataFrame(expdfs[0].groups.keys(), columns=["chrom", "start", "end"])

    # Filter out windows that have > 10% "bad" bins on average.
    # (in reality we use a second power mean to avoid being affected too much by one "bad" cooler)
    # We'll store them into regs_final.
    regs_final = []
    for _, win in window_df.iterrows():
        chrom, start_, end_ = win
        # check if window is fully within an arm for which we computed expected
        group_cur = expgroups[
            (expgroups["chrom"] == chrom) & (expgroups["start"] <= start_) & (expgroups["end"] >= end_)
        ]
        if len(group_cur) == 0:
            continue  # not fully in a recognized region
        if len(group_cur) > 1:
            raise ValueError("Multiple arms match window: unexpected scenario")

        # measure fraction of bad bins for each cooler
        bad_fracs = []
        for clr_i, clr in enumerate(all_coolers):
            chrom_offset = clr.offset(chrom)
            start_bin = chrom_offset + (start_ // resolution)
            end_bin = chrom_offset + (end_ // resolution)
            bad_mask = bad_bin_masks[cooler_uris[clr_i]][start_bin:end_bin]
            bad_fracs.append(bad_mask.mean())

        if np.sqrt(np.mean(np.array(bad_fracs) ** 2)) < 0.1:
            regs_final.append((chrom, start_, end_))

    # Convert regs_final to a DataFrame
    regs_final = pd.DataFrame(regs_final, columns=["chrom", "start", "end"])
    n_snips_final = len(regs_final)
    out_shape = (n_snips_final, len(all_coolers), actual_size, actual_size)
    weight_shape = (n_snips_final, len(all_coolers), actual_size)

    # Create the output HDF5
    with h5py.File(output_filename, "w") as hf:
        hf.create_dataset(
            "hic", shape=out_shape, dtype=np.int16, compression="lzf", chunks=(1, 1, actual_size, actual_size)
        )
        hf.create_dataset("weights", shape=weight_shape, dtype=np.float32)
        hf.create_dataset("exp", shape=weight_shape, dtype=np.float32)

    # Fill the datasets (main loop - repeating the loop above)
    counter = 0
    for chrom, start_, end_ in regs_final.itertuples(index=False):
        group_cur = expgroups[
            (expgroups["chrom"] == chrom) & (expgroups["start"] <= start_) & (expgroups["end"] >= end_)
        ]
        key = tuple(group_cur.iloc[0].values)
        exps = [expdf.get_group(key) for expdf in expdfs]

        regstr = f"{chrom}:{start_}-{end_}"
        snippet_list = []
        weight_list = []
        exp_list = []

        # Load the snippets, zero out bad bins, clip to 32000, and store. Also store weights.
        for idx_c, clr in enumerate(all_coolers):
            mat = clr.matrix(balance=False).fetch(regstr).astype(np.int32)
            mat[mat > 32000] = 32000
            mat = mat.astype(np.int16)

            chrom_offset = clr.offset(chrom)
            start_bin = chrom_offset + (start_ // resolution)
            end_bin = chrom_offset + (end_ // resolution)
            bad_mask = bad_bin_masks[cooler_uris[idx_c]][start_bin:end_bin]
            w = clr.bins().fetch(regstr)["weight"].values.copy()

            # Zero out bins that are 'bad'
            mat[bad_mask, :] = 0
            mat[:, bad_mask] = 0
            w[bad_mask] = 0
            snippet_list.append(mat)
            weight_list.append(w.astype(np.float32))

        # Build the local expected track
        for exp_df in exps:
            arr = np.arange(actual_size, dtype=np.float32)
            arr[exp_df["dist"].values] = exp_df["balanced.avg.smoothed"]
            arr[0:2] = 0  # zero out the first two diagonals
            exp_list.append(arr)

        with h5py.File(output_filename, "a") as hf:
            hf["hic"][counter] = np.array(snippet_list, dtype=np.int16)
            hf["weights"][counter] = np.array(weight_list, dtype=np.float32)
            hf["exp"][counter] = np.array(exp_list, dtype=np.float32)

        counter += 1

    # Store region info
    with h5py.File(output_filename, "a") as hf:
        for col in ["chrom", "start", "end"]:
            hf.create_dataset(col, data=regs_final[col].values)

    print(f"Saved {n_snips_final} snippets to {output_filename}")


def process_mcools(manifest_path, output_folder, resolutions, target_size, step_bins):

    resolutions = [int(r) for r in resolutions.split(",")]
    df = pd.read_csv(manifest_path)

    # Group by 'group_name'. Each group_name -> one file per resolution in output_folder.
    for group_name, group_df in df.groupby("group_name", observed=True):
        genome_vals = group_df["genome"].unique()
        if len(genome_vals) != 1:
            raise ValueError(f"Multiple genomes for group {group_name} - not supported.")
        genome = genome_vals[0]

        # Build output filenames for each resolution
        for res in resolutions:
            out_filename = os.path.join(output_folder, f"{group_name}_{res}.h5")
            cooler_uris = [f"{row['filepath']}::resolutions/{res}" for _, row in group_df.iterrows()]

            # Call the function from Part 2
            save_coolers_for_manta(
                cooler_uris=cooler_uris,
                output_filename=out_filename,
                genome=genome,
                resolution=res,
                target_size=target_size,
                step_bins=step_bins,
            )
