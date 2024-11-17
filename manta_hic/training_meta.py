"""
This is a small module that provides access to the training data for the Borzoi model. The data is stored in parquet
files in the `manta_hic/data` directory, i.e. inside of the library.


Versioning is slightly unclear, and thus I'm using year to indicate the version of Borzoi data. Current files that I
downloaded are timestamped in Dec 2023, and were the only version at the time of. Now there is borzoi "v2" data, I
believe that it is what is referred to "latest" rather than "legacy" here
https://github.com/calico/borzoi/tree/main?tab=readme-ov-file but I haven't exploerd it in depth yet.

"""

import polars as pl
import os
from importlib.resources import files

seqs_hg38 = pl.read_parquet(os.path.join(files("manta_hic"), "data", "borzoi_seqs_hg38.pq"))
seqs_mm10 = pl.read_parquet(os.path.join(files("manta_hic"), "data", "borzoi_seqs_mm10.pq"))
targs_hg38 = pl.read_parquet(os.path.join(files("manta_hic"), "data", "borzoi_targs_hg38.pq"))
targs_mm10 = pl.read_parquet(os.path.join(files("manta_hic"), "data", "borzoi_targs_mm10.pq"))

strand_pair_hg38 = targs_hg38["strand_pair"].to_numpy()
strand_pair_mm10 = targs_mm10["strand_pair"].to_numpy()
strand_pair_hg38.setflags(write=False)  # make it read-only
strand_pair_mm10.setflags(write=False)


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
    Returns the strand pair for the specified genome and "Borzoi data version". Strand pairs are cached in memory as
    read-only numpy arrays as they are rapidly needed during training.
    """
    if genome == "hg38":
        return strand_pair_hg38
    elif genome == "mm10":
        return strand_pair_mm10
    else:
        raise ValueError(f"Genome {genome} not supported. Must be 'hg38' or 'mm10'.")


def assign_folds(df, val_fold, test_fold, genome):
    """
    Assigns folds to the given dataframe based on the specified validation and test folds.

    Discards rows that do not have a corresponding fold in the fold dataframe or overlap with multiple fold types.
    Names of the assigned "fold_type" are "train", "val", and "test".

    Parameters
    ----------
    df : polars.DataFrame
        The dataframe containing the data to be assigned folds.
    val_fold : int
        The fold to be used for validation (e.g. "fold3")
    test_fold : int
        The fold to be used for testing (e.g. "fold4").
    genome : str
        The genome identifier to filter the folds.

    Returns
    -------
    polars.DataFrame
        A dataframe with an additional column 'fold' and 'fold_type', the latter indicating whether each row is part
        of the training, validation, or test set.

    Raises
    ------
    ValueError
        If `val_fold` or `test_fold` are not present in the unique folds of the fold dataframe.
    """
    df = pl.DataFrame(df)
    fold_df = pl.read_parquet(os.path.join(files("manta_hic"), "data", "borzoi_folds.pq"))

    unique_folds = fold_df["fold"].unique().to_list()
    if val_fold not in unique_folds and test_fold not in unique_folds:
        raise ValueError(f"Validation fold {val_fold} or test fold {test_fold} not found in {unique_folds}")

    fold_df = fold_df.filter(pl.col("genome") == genome).with_columns(
        pl.when(pl.col("fold") == val_fold)
        .then(pl.lit("val"))
        .otherwise(pl.when(pl.col("fold") == test_fold).then(pl.lit("test")).otherwise(pl.lit("train")))
        .alias("fold_type")
    )

    overlap = (
        df.join_where(  # sorry for the beta functionality. Living on the edge, hoping it stays.
            fold_df,
            pl.col("chrom") == pl.col("chrom_right"),
            pl.col("start") < pl.col("end_right"),
            pl.col("start_right") < pl.col("end"),
            suffix="_right",
        )
        .filter(pl.col("fold_type").n_unique().over("chrom", "start", "end") == 1)
        .select(list(df.columns) + ["fols", "fold_type"])
        .unique()
    )

    return overlap
