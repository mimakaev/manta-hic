"""L2 orchestration tests (CPU, no model/cache): RC-flip to canonical forward, channel slicing, and the
streaming group-average in ``infer`` / ``infer_grouped``. The Manta forward itself is validated on GPU."""

import torch

from manta_hic.nn.inference import MantaInference
from manta_hic.nn.specs import Spec, background_for

RES = 256
MAP = 1000 * RES  # bin-aligned map start


def _stub_inference(output_channels=3, names=("a", "b", "c"), n=4):
    """A MantaInference with __init__ skipped and the fetcher + model stubbed. The stub model writes an
    ASYMMETRIC map (so a flip is detectable) whose value is the mean of the activations, and the stub fetcher
    returns one activation tensor per spec whose value is the spec's ``value`` tag."""
    m = object.__new__(MantaInference)
    m.resolution, m.n_bins, m.bins_pad, m.device = RES, n, 1, "cpu"
    m.output_channels, m.channel_names = output_channels, list(names)

    class _Fetcher:
        N_runs = 4
        fasta_open = None

        def __init__(self):
            self.seen = set()  # distinct Backgrounds fetched (to check matched vs random)

        def fetch_activations_batch(self, specs, **kw):
            self.seen.update(s.bg for s in specs)
            return [torch.full((2, 6), float(s.tags.get("value", 0))) for s in specs]

    def _model(batch):  # [b, Cin, L] -> [b, C, n, n]
        out = torch.zeros(batch.shape[0], output_channels, n, n)
        for i in range(batch.shape[0]):
            out[i, :, 0, 1] = batch[i].mean()  # a single off-diagonal entry -> asymmetric
        return out

    m.fetcher = _Fetcher()
    m.model = _model
    return m, n


def test_l2_rc_flip_to_forward():
    m, n = _stub_inference()
    fwd = Spec(bg=background_for("chr1", MAP, run_idx=0, reverse=False), tags={"value": 1})
    rev = Spec(bg=background_for("chr1", MAP, run_idx=0, reverse=True), tags={"value": 1})
    maps, names = m.infer([fwd, rev])
    assert names == ["a", "b", "c"]
    # both come from the same native map; the reverse spec's is flipped back to forward
    assert torch.equal(maps[1], torch.flip(maps[0], dims=(-2, -1)))
    assert not torch.equal(maps[1], maps[0])  # asymmetric -> the flip actually did something


def test_l2_channel_slice():
    m, n = _stub_inference()
    s = Spec(bg=background_for("chr1", MAP, run_idx=0), channels=("b",), tags={"value": 1})
    maps, names = m.infer([s])
    assert names == ["b"] and maps[0].shape == (1, n, n)


def test_l2_infer_grouped_averages_over_backgrounds():
    m, n = _stub_inference()
    specs = [
        Spec(bg=background_for("chr1", MAP, run_idx=run), tags={"mutation_set": cond, "value": val})
        for run in range(4)
        for cond, val in [("wt", 10.0), ("mut", 20.0)]
    ]
    groups = m.infer_grouped(specs, ["mutation_set"])
    assert [g[0]["mutation_set"] for g in groups] == ["mut", "wt"]  # sorted by key
    by = {g[0]["mutation_set"]: g[1] for g in groups}
    # each group is the mean over its 4 runs of the constant value it was tagged with
    assert torch.allclose(by["wt"][:, 0, 1], torch.full((3,), 10.0))
    assert torch.allclose(by["mut"][:, 0, 1], torch.full((3,), 20.0))


def test_l2_infer_grouped_applies_rc_flip():
    m, n = _stub_inference()
    # a single reverse spec: infer_grouped must flip its (native) map back to forward before accumulating,
    # so the stub's asymmetric entry moves from native [0,1] to the flipped position [n-1, n-2].
    s = Spec(bg=background_for("chr1", MAP, run_idx=0, reverse=True), tags={"cond": "x", "value": 5.0})
    ((key, mean_map, _),) = m.infer_grouped([s], ["cond"])
    assert key == {"cond": "x"}
    assert torch.allclose(mean_map[:, n - 1, n - 2], torch.full((3,), 5.0))
    assert torch.allclose(mean_map[:, 0, 1], torch.zeros(3))


def test_l3_matched_shares_backgrounds_random_does_not():
    m, _ = _stub_inference()
    E = ("inactivate", 1_000_000, 1_000_300)
    conds = {"wt": [], "dE": [E]}
    # matched: 2 runs x both orientations = 4 backgrounds, SHARED across the 2 conditions
    m.fetcher.seen = set()
    maps, _ = m.predict_region("chr1", MAP, conds, runs=2, rc="average", backgrounds="matched", rng_seed=0)
    assert set(maps) == {"wt", "dE"}
    assert len(m.fetcher.seen) == 4
    # random: each condition draws its own -> 8 distinct backgrounds
    m.fetcher.seen = set()
    m.predict_region("chr1", MAP, conds, runs=2, rc="average", backgrounds="random", rng_seed=0)
    assert len(m.fetcher.seen) == 8


def test_l3_conditions_as_list_autonames():
    m, _ = _stub_inference()
    maps, _ = m.predict_region("chr1", MAP, [None, [("inactivate", 1_000_000, 1_000_300)]], runs=1, rc=False)
    assert set(maps) == {"wt", "cond1"}
