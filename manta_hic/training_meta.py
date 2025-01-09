"""
This is a small module that provides access to the training data for the Borzoi model. The data is stored in parquet
files in the `manta_hic/data` directory, i.e. inside of the library.


Versioning is slightly unclear, and thus I'm using year to indicate the version of Borzoi data. Current files that I
downloaded are timestamped in Dec 2023, and were the only version at the time of. Now there is borzoi "v2" data, I
believe that it is what is referred to "latest" rather than "legacy" here
https://github.com/calico/borzoi/tree/main?tab=readme-ov-file but I haven't exploerd it in depth yet.

Code that was used to generate the data is below

import polars as pl

seqs = pl.read_csv(
    f"/ssdhome/magus/work/activities/deep_learning/borzoi_inference/shared/seqs_hg38.bed",
    has_header=False,
    separator="\t",
    new_columns=["chrom", "start", "end", "fold"],
).with_columns(pl.lit("hg38").alias("genome"))

seqs2 = pl.read_csv(
    f"/ssdhome/magus/work/activities/deep_learning/borzoi_inference/shared/seqs_mm10.bed",
    has_header=False,
    separator="\t",
    new_columns=["chrom", "start", "end", "fold"],
).with_columns(pl.lit("mm10").alias("genome"))

targs = pl.read_csv("/home/magus/nn/hg38/targets.txt", separator="\t", columns=(1,2,3,4,5,6,7,8))
targs2 = pl.read_csv("/home/magus/nn/mm10/targets.txt", separator="\t", columns=(1,2,3,4,5,6,7,8))

fold_df = (
    seqs.sort("genome", "chrom", "start", "end")
    .group_by(
        "genome",
        "chrom",
        (pl.col("fold") != pl.col("fold").shift(1).over("genome", "chrom").fill_null(pl.col("fold")))
        .cum_sum()
        .alias("index"),
    )
    .agg(pl.col("start").min(), pl.col("end").max(), pl.col("fold").first(), pl.col("fold").n_unique().alias("n_folds"))
    .select("genome", "chrom", "start", "end", "fold", "n_folds")
    .sort("genome", "chrom", "start")
)
assert (fold_df["n_folds"] == 1).all()
fold_df.write_parquet("saved_datafiles_for_library/borzoi_folds.pq")
seqs.write_parquet("saved_datafiles_for_library/borzoi_seqs_hg38.pq",compression_level=9)
seqs2.write_parquet("saved_datafiles_for_library/borzoi_seqs_mm10.pq", compression_level=9)
targs.write_parquet("saved_datafiles_for_library/borzoi_targs_hg38.pq", compression_level=9)
targs2.write_parquet("saved_datafiles_for_library/borzoi_targs_mm10.pq", compression_level=9)

"""

import os
from importlib.resources import files

import numpy as np
import polars as pl

seqs_hg38 = pl.read_parquet(os.path.join(files("manta_hic"), "data", "borzoi_seqs_hg38.pq"))
seqs_mm10 = pl.read_parquet(os.path.join(files("manta_hic"), "data", "borzoi_seqs_mm10.pq"))
targs_hg38 = pl.read_parquet(os.path.join(files("manta_hic"), "data", "borzoi_targs_hg38.pq"))
targs_mm10 = pl.read_parquet(os.path.join(files("manta_hic"), "data", "borzoi_targs_mm10.pq"))

strand_pair_hg38 = targs_hg38["strand_pair"].to_numpy()
strand_pair_mm10 = targs_mm10["strand_pair"].to_numpy()
strand_pair_hg38
strand_pair_mm10


def get_seqs_targs(genome, version="2023"):
    """
    Returns the sequences and targets for the specified genome and "Borzoi data version" (see top docstring).

    Data is cached in memory on import as it is small, but is rapidly needed for training etc.

    Parameters
    ----------
    genome : str
        The genome to get the sequences and targets for. Can be "hg38" or "mm10".
    version : str
        The version of the genome to get the sequences and targets for. Default is "2023".

    Returns
    -------
    polars.DataFrame, polars.DataFrame
        The sequences and targets for the specified genome.
    """
    if genome == "hg38":
        return seqs_hg38.clone(), targs_hg38.clone()
    elif genome == "mm10":
        return seqs_mm10.clone(), targs_mm10.clone()
    else:
        raise ValueError(f"Genome {genome} not supported. Must be 'hg38' or 'mm10'.")


def get_strand_pair(genome, version="2023"):
    """
    Returns the strand pair for the specified genome and "Borzoi data version".
    """
    if genome == "hg38":
        return np.array(strand_pair_hg38)
    elif genome == "mm10":
        return np.array(strand_pair_mm10)
    else:
        raise ValueError(f"Genome {genome} not supported. Must be 'hg38' or 'mm10'.")


def assign_fold_type(df, val_fold, test_fold, genome, overlap_threshold=0.8):
    """
    Assigns folds to the given dataframe based on the specified validation and test folds.

    Identifies the rows that do not have a corresponding fold in the fold dataframe
    or overlap with multiple fold types (unless the overlap with the fold is above the specified threshold).
    Those rows are labelled with a fold type "discard".

    Names of the assigned "fold_type" are "train", "val", "test", and "discard".

    Parameters
    ----------
    df : polars.DataFrame
        The dataframe containing the data to be assigned folds.
    val_fold : str
        The fold to be used for validation (e.g. "fold3")
    test_fold : str
        The fold to be used for testing (e.g. "fold4").
    genome : str
        The genome identifier to filter the folds.
    overlap_threshold : float
        The minimum fraction of overlap between the fold and the data for it to be assigned the fold type.
        Default is 0.8.

    Returns
    -------
    polars.DataFrame
        A dataframe with an additional column 'fold_type', the latter indicating whether each row is part
        of the training, validation, or test set.

    Raises
    ------
    ValueError
        If `val_fold` or `test_fold` are not present in the unique folds of the fold dataframe.
    """

    df = pl.DataFrame(df)
    fold_df = pl.read_parquet(os.path.join(files("manta_hic"), "data", "borzoi_folds.pq"))

    # check if val_fold and test_fold are present in the fold_df
    unique_folds = fold_df["fold"].unique().to_list()
    if val_fold not in unique_folds or test_fold not in unique_folds:
        raise ValueError(f"Validation fold {val_fold} or test fold {test_fold} not found in {unique_folds}")

    # assign fold_type to the fold_df dataframe
    fold_df = fold_df.filter(pl.col("genome") == genome).with_columns(
        pl.when(pl.col("fold") == val_fold)
        .then(pl.lit("val"))
        .otherwise(pl.when(pl.col("fold") == test_fold).then(pl.lit("test")).otherwise(pl.lit("train")))
        .alias("fold_type")
    )

    # overlap, add overlap length, overlap fraction, filter out rows with multiple fold types
    df_with_folds = (
        df.join_where(
            fold_df,
            pl.col("chrom") == pl.col("chrom_right"),
            pl.col("start") < pl.col("end_right"),
            pl.col("start_right") < pl.col("end"),
            suffix="_right",
        )
        .with_columns(overlap_length=pl.min_horizontal("end", "end_right") - pl.max_horizontal("start", "start_right"))
        .with_columns((pl.col("overlap_length") / (pl.col("end") - pl.col("start"))).alias("overlap_fraction"))
        .filter(
            (pl.col("fold_type").n_unique().over("chrom", "start", "end") == 1)
            | (pl.col("overlap_fraction") > overlap_threshold)
        )
        .sort("overlap_fraction", descending=True)
        .group_by("chrom", "start", "end")
        .agg(pl.col("fold_type").first())
    )

    # verify that all 3 fold types are present
    assert df_with_folds["fold_type"].n_unique() == 3

    # Join the assignments back into the original dataframe
    df2 = df.join(df_with_folds, on=["chrom", "start", "end"], how="left")
    df2 = df2.with_columns(pl.col("fold_type").fill_null("discard"))

    # verify that dataframes are the same length and starts are the same
    assert len(df) == len(df2)
    assert (df["chrom"].to_numpy() == df2["chrom"].to_numpy()).all()

    return df2
