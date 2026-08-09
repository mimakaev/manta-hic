"""Epoch normalization: total window coverage per epoch is ~2*n_eligible (one forward + one reverse genome
pass) no matter how many n_bins sizes are trained -- each size draws the same window count."""
import numpy as np
import pytest

from manta_hic.nn.train_manta import N_BINS_DEFAULT, make_batches


def _sub(n_pos, n_models=3):
    return {"pos": np.arange(n_pos, dtype=np.int64), "elig": np.ones((n_pos, n_models), dtype=bool)}


def _nwin(sub, nb, batch_size, sum_nb):
    batches = make_batches(sub, nb, batch_size, np.random.default_rng(0), sum_nb)
    assert all(b[0] == nb and len(b[1]) == batch_size for b in batches)
    return sum(len(b[1]) for b in batches)


def test_single_size_matches_classic():
    # with one size, sum_nb == nb -> the classic 2*n_eligible/n_bins window count
    sub, bs = _sub(2_000_000), 4
    nwin = _nwin(sub, 1024, bs, 1024)
    assert nwin == int(np.ceil(2 * 2_000_000 / 1024 / bs) * bs)


@pytest.mark.parametrize("sizes", [N_BINS_DEFAULT, (512, 1024), (256, 512, 768, 896, 960, 1024)])
def test_multi_size_is_one_genome_pass(sizes):
    sub, bs = _sub(2_000_000), 4
    counts = {nb: _nwin(sub, nb, bs, sum(sizes)) for nb in sizes}
    # every size draws the same window count (identical formula, batch-rounded)
    assert len(set(counts.values())) == 1
    # summed bin coverage = 2*n_eligible within batch-rounding slack
    coverage = sum(nb * c for nb, c in counts.items())
    target = 2 * len(sub["pos"])
    assert target <= coverage <= target + bs * sum(sizes)


def test_empty_pool_returns_no_batches():
    # the no-holdout (--val-fold -1) run has an empty val pool; must not raise on rng.integers(0, 0)
    sub = {"pos": np.zeros(0, np.int64), "elig": np.zeros((0, 3), bool)}
    assert make_batches(sub, 1024, 4, np.random.default_rng(0), sum(N_BINS_DEFAULT)) == []


def test_default_sizes_not_5x():
    # the regression: pre-fix each size drew 2*P/nb windows -> ~5x coverage with the 5-size default
    sub, bs = _sub(2_000_000), 4
    coverage = sum(nb * _nwin(sub, nb, bs, sum(N_BINS_DEFAULT)) for nb in N_BINS_DEFAULT)
    assert coverage < 1.1 * 2 * len(sub["pos"])
