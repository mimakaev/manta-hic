"""L1 orchestration tests (CPU, no cache/model): cache-read dedup, the pure-cache fast path, and localized
splicing. The MicroZoi recompute + geometry is validated separately on GPU against real data."""

import torch

from manta_hic.nn.fetchers import CachedMicrozoiFetcher
from manta_hic.nn.specs import Spec, background_for, variant_specs

RES, PAD, NB = 4096, 128, 1024


def _stub_fetcher():
    """A fetcher with the cache/model I/O stubbed out, so we exercise only the batching logic."""
    f = object.__new__(CachedMicrozoiFetcher)
    f._reads = []

    def fetch(chrom, fs, fe, *, reverse=False, run_idx=None, device="cpu"):
        f._reads.append((chrom, fs, fe, reverse, run_idx))
        return torch.zeros(4, (fe - fs) // 256)

    def recompute_jobs(model, jobs, crop, device):  # mutant tiles -> 1.0, wild-type recompute -> 0.5
        return {job: torch.full((4, (job[2] - job[1]) // 256), 1.0 if job[3] else 0.5) for job in jobs}

    def splice(acts, win_lo, win_hi, patch, fs, fe, reverse):
        i0, i1 = (win_lo - fs) // 256, (win_hi - fs) // 256
        acts[:, i0:i1] = patch

    f.fetch = fetch
    f._fetch_microzoi_model = lambda device: "M"
    f._recompute_jobs = recompute_jobs
    f._splice = splice
    return f


def test_l1_dedup_pure_cache_and_localized_splice():
    f = _stub_fetcher()
    mut = (("replace", 22_000_000, "A" * 400),)
    bg = background_for("chr1", 20_000_000, run_idx=0, mutations_superset=mut)
    specs = variant_specs(bg, {"wt": None, "mut": mut, "mut2": mut})
    specs.append(Spec(bg=background_for("chr1", 20_000_000, run_idx=1)))  # tiles=None -> pure cache

    out = f.fetch_activations_batch(specs, resolution=RES, n_bins=NB, bins_pad=PAD, device="cpu")
    assert len(out) == 4

    # dedup: three specs share the run-0 background -> one read; the pure-cache spec (run 1) -> one more
    assert len(f._reads) == 2

    assert out[3].abs().sum().item() == 0  # pure-cache spec is the untouched (zero) background

    # mutant vs wild-type differ only inside the tile region
    fs = 20_000_000 - PAD * RES
    tlo = (bg.tiles[0][0] - fs) // 256
    thi = (bg.tiles[-1][1] - fs) // 256
    d = (out[1] - out[0]).abs().sum(0)
    assert d[:tlo].sum().item() == 0 and d[thi:].sum().item() == 0
    assert d[tlo:thi].sum().item() > 0


def test_l1_merge_runs():
    step = 393216
    # adjacent bricks merge into one run; a gap stays separate
    assert CachedMicrozoiFetcher._merge_runs([(0, step), (step, 2 * step)]) == [(0, 2 * step)]
    assert CachedMicrozoiFetcher._merge_runs([(0, step), (5 * step, 6 * step)]) == [
        (0, step),
        (5 * step, 6 * step),
    ]
