"""
Utilities for working with DNA sequences, including fast onehot and reverse complement functions.
"""

from typing import Optional

import numpy as np
import pysam
import ushuffle


def open_fasta_chromsizes(
    in_fasta,
    chroms=("#", "chrX"),
):
    """
    Opens a genome fasta file, or takes a pysam.FastaFile like object, and returns open file and chromsizes.

    Parameters
    ----------
    in_fasta : str or pysam.FastaFile (or compatible)
        Path to the genome fasta file or an open pysam.FastaFile object.
    chroms : Iterable, optional
        Chromosome names. Default is ["#", "chrX"] where "#" is all the numeric chromosomes chr{number}.

    Returns
    -------
    pysam.FastaFile, dict
        An open pysam.FastaFile object and a dictionary of chromosome sizes.
    """
    # Open the genome FASTA file if a path is provided
    if isinstance(in_fasta, str):
        genome_open = pysam.FastaFile(in_fasta)
    else:
        genome_open = in_fasta

    # Fetch all chromosome names and sizes
    all_chroms = genome_open.references
    chromsizes = {chrom: genome_open.get_reference_length(chrom) for chrom in all_chroms}

    # Filter chromosomes based on `chroms`
    filtered_chromsizes = {}
    for chrom in chroms:
        if chrom == "#":
            # Handle numeric chromosomes like chr1, chr2, ..., chr22
            numeric_chroms = [f"chr{i}" for i in range(1, 23)]
            for c in numeric_chroms:
                if c in chromsizes:
                    filtered_chromsizes[c] = chromsizes[c]
        else:
            # Handle explicitly specified chromosomes
            if chrom in chromsizes:
                filtered_chromsizes[chrom] = chromsizes[chrom]
            else:
                raise ValueError(f"Chromosome {chrom} not found in the genome.")

    return genome_open, filtered_chromsizes


# precomputing the array for the onehot encoding
ar = np.zeros(256, dtype=np.int32) + 4  # 4 for unknown characters - last row is discarded
ar[np.array(["ACGTacgt"], dtype="S").view(np.int8)] = np.array([0, 1, 2, 3, 0, 1, 2, 3])


def onehot_turbo(
    seq_string: str,
    mutate: Optional[list[tuple[str, int, str] | tuple[str, int, int]]] = None,
) -> np.ndarray:
    """
    A turbocharged make one hot function. Possibly the fastest you can do without Cython
    (will buy you a drink if you beat it). Uses coding ACGT as rows 0123, so reverse complement is just flipping.

    Other characters are treated as N, with is (0,0,0,0) (not 0.25).

    Parameters
    ----------

    seq_string : str
        A string containing the DNA sequence.
    mutate : list of tuples, optional
        List of tuple ("replace", position, sequence) or ("invert", position, position2). Default is None.

    Returns
    -------
    np.array

    """
    if not seq_string:
        return np.empty((0, 4), dtype=bool)
    np_string = np.array([seq_string], dtype="S").view(dtype="S1")
    if mutate:
        for m in mutate:
            if m[0] == "replace":
                np_string[m[1] : m[1] + len(m[2])] = np.array([m[2]], dtype="S").view(dtype="S1")
            elif m[0] == "invert":
                seq_rc = reverse_complement(seq_string[m[1] : m[2]])
                np_string[m[1] : m[2]] = np.array([seq_rc], dtype="S").view(dtype="S1")
            elif "shuffle" in m[0]:
                if m[0] == "shuffle":
                    shuffle_by = 2
                else:  # xtract shuffle out of "shuffle2" etc.
                    shuffle_by = int(m[0].split("shuffle")[1])
                start, end = m[1], m[2]
                seq_to_shuffle = seq_string[start:end].encode()
                shuffled_seq = ushuffle.shuffle(seq_to_shuffle, shuffle_by).decode()
                assert len(shuffled_seq) == end - start
                np_string[start:end] = np.array([shuffled_seq], dtype="S").view(dtype="S1")
            else:
                raise ValueError(f"Unknown mutation type {m[0]}.")

    np_int = np_string.view(np.int8)
    N = len(np_int)
    new_onehot = np.zeros((N, 5), dtype=bool)  # last row is discarded
    # put_along_axis is faster than fancy indexing
    np.put_along_axis(new_onehot, ar[np_int][:, None], True, axis=1)
    return new_onehot[:, :4]


def make_seq_1hot(
    genome_open: pysam.FastaFile,
    chrm: str,
    start: int,
    end: int,
    reverse: bool = False,
    mutate: Optional[list[tuple[str, int, str] | tuple[str, int, int]]] = None,
) -> np.ndarray:
    """
    Fetches a sequence from a genome and converts it to one-hot encoding.

    Parameters
    ----------
    genome_open : pysam.FastaFile
        An open pysam.FastaFile object.
    chrm : str
        Chromosome name.
    start : int
        Start position of the sequence.
    end : int
        End position of the sequence.
    reverse : bool, optional
        If True, the sequence is reverse complemented. Default is False.
    mutate : list of tuples, optional
        List of tuple ("replace", pos, seq), ("invert", pos1, pos2), or ("shuffle[1..9]", pos, pos2) Default is None.

    Notes
    -----
    Shuffling is done with ushuffle. Shuffle2 is dinucleotide shuffling (most frequent)
    so "shuffle" is a shorthand for "shuffle2".

    """

    seq_len = end - start

    if end < 0:
        seq_dna = "N" * (end - start)
    else:
        seq_dna = genome_open.fetch(chrm, max(start, 0), end)
        if start < 0:
            seq_dna = "N" * (-start) + seq_dna
        if len(seq_dna) < seq_len:
            seq_dna += "N" * (seq_len - len(seq_dna))

    # making sure mutations are not beyond the sequence, and making positions relative

    mutate_new = []
    if mutate:
        for m in mutate:
            if m[0] == "replace":
                if m[1] < start:
                    raise ValueError(f"Mutation position {m[1]} is before the sequence start {start}.")
                if m[1] + len(m[2]) > end:
                    raise ValueError(f"Mutation position {m[1]} is beyond the sequence end {end}.")
                mutate_new.append(("replace", m[1] - start, m[2]))
            elif m[0] == "invert" or "shuffle" in m[0]:
                if m[1] < start or m[2] > end:
                    raise ValueError(f"Mutation positions {m[1]} and {m[2]} are beyond the sequence.")
                mutate_new.append((m[0], m[1] - start, m[2] - start))
            else:
                raise ValueError(f"Unknown mutation type {m[0]}.")

    seq_1hot = onehot_turbo(seq_dna, mutate=mutate_new)
    assert len(seq_1hot) == end - start

    if reverse:
        return seq_1hot[::-1, ::-1].astype(np.float32)
    return seq_1hot.astype(np.float32)


# Precompute the complement mapping using NumPy arrays
_complement_array = np.zeros(128, dtype=np.uint8) + ord("N")  # other letters -> N
for i, j in zip(b"ACGTacgtN", b"TGCAtgcaN"):
    _complement_array[i] = j


def reverse_complement(seq):
    """
    Compute the reverse complement of a DNA sequence.

    Parameters:
    seq (str): A string representing the DNA sequence (only characters A, T, G, C, N).

    Returns:
    str: The reverse complement of the input DNA sequence.
    """
    # Convert the sequence string to a NumPy array of ASCII values
    seq_array = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)

    # Map each nucleotide to its complement
    comp_seq_array = _complement_array[seq_array]

    # Reverse the complemented sequence
    rev_comp_seq_array = comp_seq_array[::-1]

    # Convert the NumPy array back to a string
    return rev_comp_seq_array.tobytes().decode("ascii")


class InMemoryFasta:
    """
    A class that preloads chromosome sequences from a FASTA file into memory for faster access.

    Parameters
    ----------
    fasta_open : pysam.FastaFile or similar
        An open FASTA file object that supports the 'fetch' method.
    chroms : list(str), optional
        Chromosome names to preload. Default is ["#", "X", "Y"] where "#" is all the numeric chromosomes chr{number}.

    Methods
    -------
    fetch(chrom, start, end)
        Retrieves a subsequence from the preloaded chromosome sequence.
    get_reference_length(chrom)
        Retrieves the length of the chromosome sequence (mimics the 'get_reference_length' method of pysam.FastaFile).
    """

    def __init__(self, fasta_open: pysam.FastaFile, chroms: tuple[str] = ("#", "chrX")):
        chromsizes = fasta_open.get_reference_length()
        if "#" in chroms:
            try_chroms = [f"chr{i}" for i in range(1, 23)]
            chrom_names = [c for c in try_chroms if c in chromsizes]
        for chrom in chroms:
            if chrom not in chromsizes:
                raise ValueError(f"Chromosome {chrom} not found in FASTA file.")
            chrom_names.append(chrom)

        self.seqs = {}
        for chrom in chrom_names:
            self.seqs[chrom] = fasta_open.fetch(chrom, 0, chromsizes[chrom])
        self.references = list(self.seqs.keys())  # for compatibility with pysam.FastaFile

    def fetch(self, chrom: str, start: str, end: str) -> str:
        seq = self.seqs[chrom]
        return seq[start:end]

    def get_reference_length(self, chrom: str) -> int:
        return len(self.seqs[chrom])
