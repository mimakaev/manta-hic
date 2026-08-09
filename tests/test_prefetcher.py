"""_Prefetcher failure semantics: a worker-thread error must surface in the main thread (not hang training on an
empty queue), and an early-exiting consumer must release the worker (not leave it blocked on a full queue)."""

import signal
import time
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from manta_hic.nn.train_manta import _Prefetcher


@contextmanager
def _deadline(seconds):
    """Fail the test instead of hanging it -- the whole point of the prefetcher contract."""

    def _boom(signum, frame):
        raise TimeoutError(f"prefetcher test exceeded {seconds}s deadline (hang regression)")

    old = signal.signal(signal.SIGALRM, _boom)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _models():
    banded = SimpleNamespace(
        pos_to_coord=lambda pos: ("chr1", int(pos) * 1024),
        window_at=lambda pos, nb: (np.ones((1, nb, nb), np.float32), np.ones((1, nb), np.float32),
                                   np.ones((1, nb * 5 // 4), np.float32)),
    )
    return [SimpleNamespace(banded=banded)]


def _batches(n, nb=16, batch=2):
    return [(nb, np.arange(batch), np.ones((batch, 1), bool), np.zeros(batch, bool)) for _ in range(n)]


def _pf(batches, fetch, queue_size=1):
    fetcher = SimpleNamespace(fetch=fetch)
    return _Prefetcher(batches, _models(), fetcher, bins_pad=2, res=1024, n_runs_fn=lambda: 1,
                       queue_size=queue_size)


def _ok_fetch(chrom, s, e, **kw):
    return torch.zeros(3, (e - s) // 1024)


def test_yields_all_batches_then_stops():
    with _deadline(30):
        items = list(_pf(_batches(3), _ok_fetch))
    assert len(items) == 3
    nb, acts, elig, targets = items[0]
    assert nb == 16 and acts.shape[0] == 2 and 0 in targets


def test_worker_exception_propagates_instead_of_hanging():
    def bad_fetch(chrom, s, e, **kw):
        raise KeyError(f"{chrom} not in cache")

    with _deadline(30):
        with pytest.raises(KeyError, match="not in cache"):
            list(_pf(_batches(3), bad_fetch))


def test_mid_stream_exception_after_good_batches():
    calls = {"n": 0}

    def flaky_fetch(chrom, s, e, **kw):
        calls["n"] += 1
        if calls["n"] > 3:
            raise ValueError("window crosses an arm boundary")
        return _ok_fetch(chrom, s, e)

    with _deadline(30):
        got = []
        with pytest.raises(ValueError, match="arm boundary"):
            for item in _pf(_batches(5), flaky_fetch):
                got.append(item)
    assert len(got) == 1  # the first (fully fetched) batch arrived before the failure surfaced


def test_early_consumer_exit_releases_worker():
    pf = _pf(_batches(50), _ok_fetch, queue_size=1)
    with _deadline(30):
        it = iter(pf)
        next(it)
        it.close()  # consumer walks away mid-epoch; generator finally must stop the worker
        assert pf._stop.is_set()
        deadline = time.monotonic() + 10  # worker notices the stop event within ~a put timeout
        while pf._thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not pf._thread.is_alive(), "worker thread still blocked after consumer exit"
