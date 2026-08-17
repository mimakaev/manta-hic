"""Tests for the streaming predicted-cooler writer (io/predicted_cooler.py)."""

import cooler
import numpy as np
import pandas as pd
import pytest

from manta_hic.io.predicted_cooler import BandAccumulator, CoolerWriterPool, assemble_mcool


def _naive_band(adds, n_channels, n_diag, chrom_nbins):
    """Dense reference: mean of the upper triangles of every added square, in (x, d) band coordinates."""
    sums = np.zeros((n_channels, chrom_nbins, n_diag))
    cnts = np.zeros((chrom_nbins, n_diag))
    for start, sq in adds:
        for r in range(n_diag):
            for c in range(r, n_diag):
                sums[:, start + r, c - r] += sq[:, r, c]
                cnts[start + r, c - r] += 1
    return sums, cnts


def _collect(chunks_lists):
    """Flatten emitted chunks into one (bin1, bin2, counts[C, n]) triple sorted by (bin1, bin2)."""
    b1 = np.concatenate([c[0] for cl in chunks_lists for c in cl] or [np.zeros(0, np.int64)])
    b2 = np.concatenate([c[1] for cl in chunks_lists for c in cl] or [np.zeros(0, np.int64)])
    cnt = np.concatenate([c[2] for cl in chunks_lists for c in cl] or [np.zeros((1, 0), np.float32)], axis=1)
    order = np.lexsort((b2, b1))
    return b1[order], b2[order], cnt[:, order]


@pytest.mark.parametrize("stride", [3, 8, 17])
def test_band_accumulator_matches_naive(stride):
    rng = np.random.default_rng(0)
    C, n_diag, nbins, offset = 3, 16, 200, 1000
    starts = list(range(0, nbins - n_diag + 1, stride))
    if starts[-1] != nbins - n_diag:
        starts.append(nbins - n_diag)  # tail window, like the sweep
    adds = [(s, rng.random((C, n_diag, n_diag)).astype(np.float32)) for s in starts]

    acc = BandAccumulator(C, n_diag, nbins, chrom_offset=offset, min_flush_rows=8)
    emitted = [acc.add(s, sq) for s, sq in adds] + [acc.finalize()]
    b1, b2, counts = _collect(emitted)

    sums, cnts = _naive_band(adds, C, n_diag, nbins)
    x, d = np.nonzero(cnts)
    order = np.lexsort((x + d, x))
    assert np.array_equal(b1, x[order] + offset)
    assert np.array_equal(b2, (x + d)[order] + offset)
    np.testing.assert_allclose(counts, (sums[:, x, d] / cnts[x, d])[:, order], rtol=1e-5)


def test_band_accumulator_arm_gap_jump():
    """A jump far beyond the buffer (next arm) must flush cleanly and keep global coordinates right."""
    rng = np.random.default_rng(1)
    C, n_diag, nbins = 2, 8, 500
    adds = [
        (0, rng.random((C, n_diag, n_diag)).astype(np.float32)),
        (400, rng.random((C, n_diag, n_diag)).astype(np.float32)),
    ]
    acc = BandAccumulator(C, n_diag, nbins, min_flush_rows=4)
    emitted = [acc.add(s, sq) for s, sq in adds] + [acc.finalize()]
    b1, b2, counts = _collect(emitted)
    sums, cnts = _naive_band(adds, C, n_diag, nbins)
    x, d = np.nonzero(cnts)
    order = np.lexsort((x + d, x))
    assert np.array_equal(b1, x[order])
    np.testing.assert_allclose(counts, (sums[:, x, d] / cnts[x, d])[:, order], rtol=1e-5)


def test_add_triu_matches_add():
    rng = np.random.default_rng(3)
    C, n_diag, nbins = 2, 8, 64
    adds = [(s, rng.random((C, n_diag, n_diag)).astype(np.float32)) for s in range(0, nbins - n_diag + 1, 3)]
    a1 = BandAccumulator(C, n_diag, nbins, min_flush_rows=4)
    a2 = BandAccumulator(C, n_diag, nbins, min_flush_rows=4)
    e1 = [a1.add(s, sq) for s, sq in adds] + [a1.finalize()]
    e2 = [a2.add_triu(s, sq.reshape(C, -1)[:, a2.triu_src]) for s, sq in adds] + [a2.finalize()]
    x1, y1, c1 = _collect(e1)
    x2, y2, c2 = _collect(e2)
    assert np.array_equal(x1, x2) and np.array_equal(y1, y2)
    np.testing.assert_array_equal(c1, c2)


def test_band_accumulator_rejects_bad_adds():
    acc = BandAccumulator(1, 8, 100, min_flush_rows=4)
    acc.add(50, np.zeros((1, 8, 8), np.float32))
    with pytest.raises(ValueError, match="backwards"):
        acc.add(10, np.zeros((1, 8, 8), np.float32))
    with pytest.raises(ValueError, match="overflows"):
        acc.add(95, np.zeros((1, 8, 8), np.float32))
    with pytest.raises(ValueError, match="shape"):
        acc.add(60, np.zeros((1, 4, 4), np.float32))


def test_writer_pool_roundtrip(tmp_path):
    """Chunks fed through the writer processes come back verbatim from the finished coolers."""
    chromsizes = {"chrA": 1000, "chrB": 600}
    resolution, C = 100, 2
    paths = [str(tmp_path / f"ch{i}.cool") for i in range(C)]
    rng = np.random.default_rng(2)
    # two chunks on chrA (10 bins) + one on chrB (global bins 10..15), bin1 <= bin2, ascending
    chunks = [
        (np.array([0, 0, 1]), np.array([0, 3, 2]), rng.random((C, 3)).astype(np.float32)),
        (np.array([4, 5]), np.array([6, 5]), rng.random((C, 2)).astype(np.float32)),
        (np.array([10, 12]), np.array([11, 15]), rng.random((C, 2)).astype(np.float32)),
    ]
    with CoolerWriterPool(paths, chromsizes, resolution, assembly="testasm") as pool:
        for c in chunks:
            pool.put_chunk(*c)
        pool.finish()
    b1 = np.concatenate([c[0] for c in chunks])
    b2 = np.concatenate([c[1] for c in chunks])
    for i, path in enumerate(paths):
        clr = cooler.Cooler(path)
        assert clr.binsize == resolution and clr.info["genome-assembly"] == "testasm"
        assert (clr.bins()["weight"][:] == 1.0).all()
        pix = clr.pixels()[:]
        assert np.array_equal(pix["bin1_id"].values, b1) and np.array_equal(pix["bin2_id"].values, b2)
        np.testing.assert_allclose(pix["count"].values, np.concatenate([c[2][i] for c in chunks]), rtol=1e-6)


def _make_cool(path, resolution, chromsizes, seed):
    bins = cooler.util.binnify(pd.Series(chromsizes), resolution)
    b1, b2 = np.triu_indices(len(bins))
    counts = np.random.default_rng(seed).random(len(b1)).astype(np.float32)
    pix = pd.DataFrame({"bin1_id": b1, "bin2_id": b2, "count": counts})
    cooler.create_cooler(str(path), bins, pix, dtypes={"count": "float32"}, ordered=True)


def test_assemble_mcool(tmp_path):
    chromsizes = {"chrA": 80_000}
    for res in (1000, 2000, 4000):
        _make_cool(tmp_path / f"t_{res}.cool", res, chromsizes, seed=res)
    out = str(tmp_path / "t.mcool")
    resolutions = assemble_mcool([str(tmp_path / f"t_{r}.cool") for r in (1000, 2000, 4000)], out, min_zoom_bins=5)
    listed = sorted(int(u.rsplit("/", 1)[1]) for u in cooler.fileops.list_coolers(out))
    assert listed == resolutions and resolutions[:4] == [1000, 2000, 4000, 8000]
    # finer levels are the *predicted* inputs copied verbatim, not aggregates of the base
    for res in (1000, 2000):
        got = cooler.Cooler(f"{out}::resolutions/{res}").pixels()[:]["count"].values
        want = cooler.Cooler(str(tmp_path / f"t_{res}.cool")).pixels()[:]["count"].values
        np.testing.assert_array_equal(got, want)
    with pytest.raises(ValueError, match="doubling"):
        assemble_mcool([str(tmp_path / "t_1000.cool"), str(tmp_path / "t_4000.cool")], str(tmp_path / "bad.mcool"))
