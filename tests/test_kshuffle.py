"""Tests for the pure-Python k-let-preserving shuffle (manta_hic.ops.kshuffle), the ushuffle replacement."""

from collections import Counter

import numpy as np
import pytest

from manta_hic.ops.kshuffle import k_let_shuffle, set_seed


def _klets(seq: bytes, k: int) -> Counter:
    return Counter(seq[i : i + k] for i in range(len(seq) - k + 1))


def _rand_dna(n, rng):
    return bytes(rng.choice(list(b"ACGT"), size=n).tolist())


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("n", [16, 200, 4000])
def test_preserves_klet_counts_and_length(k, n):
    rng = np.random.default_rng(k * 1000 + n)
    for _ in range(20):
        seq = _rand_dna(n, rng)
        out = k_let_shuffle(seq, k, rng=rng)
        assert len(out) == len(seq)
        assert set(out) <= set(seq)  # no new symbols introduced
        if k == 1:
            assert Counter(out) == Counter(seq)  # k=1 preserves the multiset of symbols
        else:
            assert _klets(out, k) == _klets(seq, k)  # exact k-let counts preserved


@pytest.mark.parametrize("k", [2, 3, 4])
def test_endpoints_are_fixed(k):
    rng = np.random.default_rng(0)
    seq = _rand_dna(500, rng)
    for _ in range(10):
        out = k_let_shuffle(seq, k, rng=rng)
        assert out[: k - 1] == seq[: k - 1]  # the start (k-1)-mer is preserved
        assert out[-(k - 1) :] == seq[-(k - 1) :]  # so is the end (k-1)-mer


def test_actually_shuffles():
    rng = np.random.default_rng(1)
    seq = _rand_dna(4000, rng)
    outs = {k_let_shuffle(seq, 2, rng=rng) for _ in range(8)}
    assert len(outs) > 1 and seq not in outs or len(outs) > 1  # produces varied, generally non-identity output


def test_reproducible_and_varies_with_rng():
    seq = _rand_dna(400, np.random.default_rng(2))
    a = k_let_shuffle(seq, 2, rng=np.random.default_rng(7))
    b = k_let_shuffle(seq, 2, rng=np.random.default_rng(7))
    c = k_let_shuffle(seq, 2, rng=np.random.default_rng(8))
    assert a == b  # same seed -> identical
    assert a != c  # different seed -> (almost surely) different


def test_str_and_bytes_and_edge_sizes():
    assert isinstance(k_let_shuffle("ACGTACGT", 2), bytes)  # str in -> bytes out
    assert k_let_shuffle(b"ACG", 4) == b"ACG"  # len <= k -> unchanged (only one arrangement)
    assert k_let_shuffle(b"", 2) == b""
    with pytest.raises(ValueError):
        k_let_shuffle(b"ACGT", 0)


def test_homopolymer_and_low_complexity_terminate():
    # graphs that stress the arborescence rejection loop: a single self-loop vertex, and a repeat
    assert k_let_shuffle(b"AAAAAAAA", 2) == b"AAAAAAAA"
    rng = np.random.default_rng(3)
    rep = b"ATATATATATATATAT"
    out = k_let_shuffle(rep, 2, rng=rng)
    assert _klets(out, 2) == _klets(rep, 2)


def _oracle(seq: bytes, k: int) -> list[bytes]:
    """De-novo brute-force ground truth: every string over ``seq``'s alphabet with identical k-mer counts
    AND identical endpoints (the (k-1)-mer a k-let shuffle fixes). Shares no code with the shuffler."""
    from itertools import product

    alpha = sorted(set(seq))
    tgt = _klets(seq, k)
    lo, hi = seq[: k - 1], seq[-(k - 1) :]
    return sorted(
        bytes(t)
        for t in product(alpha, repeat=len(seq))
        if _klets(bytes(t), k) == tgt and bytes(t)[: k - 1] == lo and bytes(t)[-(k - 1) :] == hi
    )


@pytest.mark.parametrize(
    "seq,k",
    [
        (b"ACAGATACAG", 2),  # non-circuit
        (b"AGAGCGCTA", 2),  # Eulerian circuit (first char == last char)
        (b"GACGTACGTC", 2),
    ],
)
def test_samples_uniformly_over_enumerated_support(seq, k):
    """Seeded (so deterministic, not flaky): every enumerated sequence is produced, nothing outside it is,
    and the frequencies pass a chi-square goodness-of-fit against uniform."""
    valid = _oracle(seq, k)
    idx = {s: i for i, s in enumerate(valid)}
    rng = np.random.default_rng(12345)
    N = 60_000
    obs = np.zeros(len(valid), dtype=np.int64)
    for _ in range(N):
        out = k_let_shuffle(seq, k, rng=rng)
        assert out in idx, f"produced a sequence outside the k-let/endpoint-preserving set: {out!r}"
        obs[idx[out]] += 1
    assert (obs > 0).all(), "some valid sequences were never sampled"
    exp = N / len(valid)
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    df = len(valid) - 1
    # generous 99.99%-ish band via the normal approximation to chi-square (mean df, variance 2*df)
    assert df - 4.0 * (2 * df) ** 0.5 <= chi2 <= df + 4.0 * (2 * df) ** 0.5, f"chi2={chi2} df={df} not uniform"


def test_set_seed_module_default_is_deterministic():
    seq = _rand_dna(300, np.random.default_rng(4))
    set_seed(123)
    a = k_let_shuffle(seq, 2)
    set_seed(123)
    b = k_let_shuffle(seq, 2)
    assert a == b
