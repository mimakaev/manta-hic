"""L0 spec-layer tests (pure, CPU): mutation geometry, the concrete tile pattern + greedy merge, the
soft-causality rule (soft_causality + max_shift tolerance), the generators, and mutation freezing."""

import numpy as np
import pytest

from manta_hic.nn.specs import (
    BIN_BP,
    Spec,
    auto_patch_windows,
    background_for,
    build_specs,
    mutation_span,
    random_backgrounds,
    tile_pattern,
    tile_size_bp,
    validate_patches,
    variant_specs,
)
from manta_hic.ops.seq_ops import freeze_mutations


def test_mutation_span():
    assert mutation_span(("replace", 1000, "ACGT")) == (1000, 1004)
    assert mutation_span(("invert", 1000, 1500)) == (1000, 1500)
    assert mutation_span(("shuffle2", 1000, 1500)) == (1000, 1500)
    assert mutation_span(("inactivate", 1000, 1500)) == (1000, 1500)
    with pytest.raises(ValueError):
        mutation_span(("insert", 1000, "ACGT"))  # length-changing -> rejected


def test_auto_patch_windows_tolerance_is_causality_plus_shift():
    sc, ms = 3 * BIN_BP, 2 * BIN_BP
    wins = auto_patch_windows([("replace", 100_000, "A" * BIN_BP)], soft_causality_bp=sc, max_shift_bp=ms)
    assert len(wins) == 1
    lo, hi = wins[0]
    assert lo % BIN_BP == 0 and hi % BIN_BP == 0
    assert lo <= 100_000 - (sc + ms) and 100_000 + BIN_BP + (sc + ms) <= hi


def test_tile_pattern_greedy_merge():
    step = tile_size_bp()  # one MicroZoi tile in bp
    # two ~1.1-tile windows, 0.2 tile apart -> greedy tiling keeps going -> 3 tiles, not 4
    A = (0, int(1.1 * step))
    B = (int(1.3 * step), int(2.4 * step))
    tiles = tile_pattern([A, B])
    assert len(tiles) == 3
    assert all(hi - lo == step for lo, hi in tiles)  # each brick is one tile wide
    assert tiles == tuple(sorted(tiles))  # contiguous, ordered
    # a real gap (window far away) does NOT merge
    C = (int(3.0 * step), int(4.1 * step))
    assert len(tile_pattern([A, C])) == 4
    # a sub-tile window is one brick
    assert len(tile_pattern([(0, step // 4)])) == 1


def test_soft_causality_rule_maxs_example():
    muts = [("replace", 90 * BIN_BP, "A" * BIN_BP), ("replace", 110 * BIN_BP, "A" * BIN_BP)]
    kw = dict(soft_causality_bp=10 * BIN_BP, max_shift_bp=0)
    validate_patches(muts, [(0, 200 * BIN_BP)], **kw)  # one covering tile: ok
    validate_patches(muts, [(0, 100 * BIN_BP), (100 * BIN_BP, 200 * BIN_BP)], **kw)  # adjacent seam: ok
    with pytest.raises(ValueError, match="soft-causality"):
        validate_patches(muts, [(102 * BIN_BP, 201 * BIN_BP)], **kw)  # starts inside the radius
    with pytest.raises(ValueError, match="soft-causality"):
        validate_patches(muts, [(0, 95 * BIN_BP), (115 * BIN_BP, 200 * BIN_BP)], **kw)  # gap in the radius


def test_shift_eats_headroom_not_causality():
    mut = [("replace", 100_000, "A" * BIN_BP)]
    cover = auto_patch_windows(mut, soft_causality_bp=2 * BIN_BP, max_shift_bp=0)
    validate_patches(mut, cover, soft_causality_bp=2 * BIN_BP, max_shift_bp=0)
    with pytest.raises(ValueError, match="soft-causality"):
        validate_patches(mut, cover, soft_causality_bp=2 * BIN_BP, max_shift_bp=5 * BIN_BP)


def test_background_hashable_and_is_dedup_key():
    bg1 = background_for("chr1", 1_000_000, run_idx=0)
    bg2 = background_for("chr1", 1_000_000, run_idx=0)
    bg3 = background_for("chr1", 1_000_000, run_idx=1)
    assert bg1 == bg2 and hash(bg1) == hash(bg2)
    assert bg1 != bg3
    assert len({bg1, bg2, bg3}) == 2


def test_background_rejects_shift_beyond_max():
    with pytest.raises(ValueError, match="max_shift"):
        background_for("chr1", 1_000_000, run_idx=0, tile_offset_bins=10, max_shift_bp=5 * BIN_BP)


def test_background_lays_tiles_and_validates():
    muts = [("replace", 1_000_000, "A" * 100)]
    bg = background_for("chr1", 900_000, run_idx=0, mutations_superset=muts, soft_causality_bp=BIN_BP, max_shift_bp=0)
    assert bg.tiles is not None and len(bg.tiles) == 1  # tiny mutation fits one tile
    with pytest.raises(ValueError, match="soft-causality"):
        background_for(
            "chr1",
            900_000,
            run_idx=0,
            tiles=[(1_000_000, 1_000_256)],
            mutations_superset=[("replace", 2_000_000, "A" * 100)],
            soft_causality_bp=BIN_BP,
            max_shift_bp=0,
        )


def test_variant_specs_share_background_clean_pair():
    muts = (("replace", 1_000_000, "ACGTACGT"),)
    bg = background_for("chr1", 900_000, run_idx=3, mutations_superset=muts, soft_causality_bp=BIN_BP, max_shift_bp=0)
    wt, mut = variant_specs(bg, {"wt": None, "mut": muts})
    assert wt.bg is mut.bg
    assert wt.mutations == () and mut.mutations == muts
    assert wt.tags == {"variant": "wt"} and mut.tags == {"variant": "mut"}


def test_clone_swaps_mutations_keeps_background_merges_tags():
    bg = background_for("chr1", 900_000, run_idx=0)
    s = Spec(bg=bg, tags={"pos": 5})
    s2 = s.clone(mutations=(("invert", 950_000, 950_500),), tags={"variant": "mut"})
    assert s2.bg is bg and s2.mutations == (("invert", 950_000, 950_500),)
    assert s2.tags == {"pos": 5, "variant": "mut"}
    assert s.mutations == () and s.tags == {"pos": 5}


def test_random_backgrounds_runs_shift_and_rc():
    rng = np.random.default_rng(0)
    bgs = random_backgrounds("chr1", 1_000_000, runs=8, rng=rng, reverse="both", max_shift_bp=100 * BIN_BP)
    assert len(bgs) == 16 and sorted({b.run_idx for b in bgs}) == list(range(8))
    assert {b.reverse for b in bgs} == {False, True}
    assert all(-100 <= b.tile_offset_bins <= 100 for b in bgs)  # symmetric, within +-max_shift/BIN_BP
    assert any(b.tile_offset_bins < 0 for b in bgs)
    bgs2 = random_backgrounds("chr1", 1_000_000, runs=[2, 5, 9], rng=rng)
    assert sorted(b.run_idx for b in bgs2) == [2, 5, 9]


def test_freeze_inactivate_gives_fresh_replace_sequences():
    muts = (("inactivate", 1_000_000, 1_000_400),)
    frozen = [freeze_mutations(muts, rng=np.random.default_rng(i)) for i in range(4)]
    seqs = [f[0][2] for f in frozen]
    assert all(f[0][0] == "replace" and len(f[0][2]) == 400 for f in frozen)
    assert len(set(seqs)) == 4 and all(set(s) <= set("ACGT") for s in seqs)


def test_freeze_shuffle_needs_fasta_and_preserves_length():
    class FakeFasta:
        def fetch(self, chrom, lo, hi):
            return ("ACGT" * (hi - lo))[: hi - lo]  # real fasta returns exactly hi-lo bases

    frozen = freeze_mutations(
        (("shuffle2", 1000, 1040),), fasta=FakeFasta(), chrom="chr1", rng=np.random.default_rng(0)
    )
    assert frozen[0][0] == "replace" and len(frozen[0][2]) == 40
    with pytest.raises(ValueError, match="fasta"):
        freeze_mutations((("shuffle2", 1000, 1040),))


def test_freeze_passes_through_deterministic_ops():
    muts = (("replace", 100, "ACGT"), ("invert", 200, 300))
    assert freeze_mutations(muts) == muts


def test_build_specs_matrix():
    del_E = ("inactivate", 10_200_000, 10_205_000)
    del_P = ("inactivate", 10_470_000, 10_475_000)
    sets = {"wt": [], "dE": [del_E], "dP": [del_P], "dEP": [del_E, del_P]}
    bgs = random_backgrounds(
        "chr1", 10_000_000, runs=3, reverse="both", mutations_superset={del_E, del_P}, rng=np.random.default_rng(0)
    )
    specs = build_specs(
        bgs, sets, rng=np.random.default_rng(1), background_meta=[{"run": b.run_idx, "rc": b.reverse} for b in bgs]
    )
    assert len(specs) == len(bgs) * len(sets)  # 6 backgrounds x 4 sets = 24
    assert {"background", "mutation_set", "run", "rc"} <= set(specs[5].tags)
    assert all(x.mutations == () for x in specs if x.tags["mutation_set"] == "wt")
    dep = [x for x in specs if x.tags["mutation_set"] == "dEP"]
    assert all(len(x.mutations) == 2 and all(m[0] == "replace" for m in x.mutations) for x in dep)
    dE_seqs = {x.mutations[0][2] for x in specs if x.tags["mutation_set"] == "dE"}
    assert len(dE_seqs) == len(bgs)  # fresh inert realization per background
    b0 = {x.tags["mutation_set"]: x for x in specs if x.tags["background"] == 0}
    assert b0["wt"].bg is b0["dE"].bg  # matched pair shares the background


def test_build_specs_requires_tiles_for_mutations():
    bg = background_for("chr1", 10_000_000, run_idx=0)  # no superset -> tiles is None
    with pytest.raises(ValueError, match="tiles"):
        build_specs([bg], {"mut": [("replace", 10_500_000, "ACGT")]})
