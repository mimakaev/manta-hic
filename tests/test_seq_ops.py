import numpy as np
import pytest

from manta_hic.ops.seq_ops import onehot_turbo, reverse_complement


def test_onehot_turbo():
    # Test case 1: Standard DNA sequence
    seq = "ACGT"
    expected_output = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=bool)
    np.testing.assert_array_equal(onehot_turbo(seq), expected_output)

    # Test case 2: Lowercase DNA sequence
    seq = "acgt"
    expected_output = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=bool)
    np.testing.assert_array_equal(onehot_turbo(seq), expected_output)

    # Test case 3: Mixed case DNA sequence
    seq = "AcGt"
    expected_output = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=bool)
    np.testing.assert_array_equal(onehot_turbo(seq), expected_output)

    # Test case 4: Sequence with unknown characters
    seq = "ACGTN"
    expected_output = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=bool)
    np.testing.assert_array_equal(onehot_turbo(seq), expected_output)

    # Test case 5: Empty sequence
    seq = ""
    expected_output = np.empty((0, 4), dtype=bool)
    np.testing.assert_array_equal(onehot_turbo(seq), expected_output)

    # Test case 6: unknown characters in sequence treated as N
    seq = "AZ"
    expected_output = np.array([[1, 0, 0, 0], [0, 0, 0, 0]], dtype=bool)
    np.testing.assert_array_equal(onehot_turbo(seq), expected_output)


def test_reverse_complement():
    # Test case 1: Standard DNA sequence
    seq = "ACGT"
    expected_output = "ACGT"
    assert reverse_complement(seq) == expected_output

    # Test case 2: Lowercase DNA sequence
    seq = "acct"
    expected_output = "aggt"
    assert reverse_complement(seq) == expected_output

    # Test case 3: Mixed case DNA sequence
    seq = "AcGt"
    expected_output = "aCgT"
    assert reverse_complement(seq) == expected_output

    # Test case 4: Sequence with unknown characters
    seq = "ACGTN"
    expected_output = "NACGT"
    assert reverse_complement(seq) == expected_output

    # Test case 5: Empty sequence
    seq = ""
    expected_output = ""
    assert reverse_complement(seq) == expected_output

    # Test case 6: Sequence with only unknown characters
    seq = "NNNN"
    expected_output = "NNNN"
    assert reverse_complement(seq) == expected_output

    # Test case 7: Sequence with unknown characters treated as N
    seq = "AZ"
    expected_output = "NT"
    assert reverse_complement(seq) == expected_output
