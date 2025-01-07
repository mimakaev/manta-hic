"""
Utilities for working with DNA sequences, including fast onehot and reverse complement functions.
"""

import numpy as np
import pysam

# precomputing the array for the onehot encoding
ar = np.zeros(256, dtype=np.int32) + 4  # 4 for unknown characters - last row is discarded
ar[np.array(["ACGTacgt"], dtype="S").view(np.int8)] = np.array([0, 1, 2, 3, 0, 1, 2, 3])


def onehot_turbo(seq_string: str) -> np.ndarray:
    """
    A turbocharged make one hot function. Possibly the fastest you can do without Cython
    (will buy you a drink if you beat it). Uses coding ACGT as rows 0123, so reverse complement is just flipping.

    Other characters are treated as N, with is (0,0,0,0) (not 0.25).

    Parameters
    ----------

    seq_string : str
        A string containing the DNA sequence.

    Returns
    -------
    np.array

    """
    if not seq_string:
        return np.empty((0, 4), dtype=bool)
    np_string = np.array([seq_string], dtype="S")
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
) -> np.ndarray:

    seq_len = end - start

    if end < 0:
        seq_dna = "N" * (end - start)
    else:
        seq_dna = genome_open.fetch(chrm, max(start, 0), end)
        if start < 0:
            seq_dna = "N" * (-start) + seq_dna
        if len(seq_dna) < seq_len:
            seq_dna += "N" * (seq_len - len(seq_dna))

    seq_1hot = onehot_turbo(seq_dna)
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
    """

    def __init__(self, fasta_open: pysam.FastaFile, chroms: tuple[str] = ("#", "chrX", "chrY")):
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

    def fetch(self, chrom: str, start: str, end: str) -> str:
        seq = self.seqs[chrom]
        return seq[start:end]
