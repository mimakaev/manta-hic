"""
Tests for the whole-file read layer (:class:`manta_hic.io.banded.BandedHicFile`): metadata, eligibility, and window
reconstruction.

A minimal but structurally faithful ``.bhic.h5`` is written by hand (same layout as
``io.banded_write.coolers_to_banded``) so these tests need no coolers / bioframe / cooltools.
"""

import h5py
import hdf5plugin
import numpy as np
import pytest
import torch

from manta_hic.io.banded import (
    BandedHicFile,
    band_from_dense,
    square_from_band,
    square_from_block,
)


def _symmetric_counts(L, C, seed):
    rng = np.random.default_rng(seed)
    out = np.zeros((C, L, L), dtype=np.float32)
    for c in range(C):
        a = rng.poisson(3.0, size=(L, L)).astype(np.float32)
        m = (a + a.T) / 2
        dist = np.abs(np.subtract.outer(np.arange(L), np.arange(L)))
        out[c] = m / (1 + dist)
    return out


def _write_min_banded(path, *, C=2, n_diag=24, res=1000, genome="hg38"):
    """Write a two-chromosome banded file by hand; return (chrom_specs) for cross-checking."""
    rng = np.random.default_rng(7)
    n_arms = 2  # chr1 -> arm 0, chr2 -> arm 1 (one arm each, for simplicity)
    specs = {"chr1": 120, "chr2": 90}
    exp = rng.uniform(0.1, 1.0, size=(C, n_arms, n_diag)).astype(np.float32)
    exp[:, :, :2] = 0.0
    mats = {}
    with h5py.File(path, "w") as f:
        f.attrs.update(
            dict(
                format="manta-banded-v1",
                genome=genome,
                resolution=res,
                n_diag=n_diag,
                group_name="g",
                n_channels=C,
                accepted_fraction=0.5,
                n_eligible=1,
                n_candidate=2,
                accept_max_bad_fraction=0.1,
            )
        )
        sdt = h5py.string_dtype()
        prov = f.create_group("provenance")
        prov.create_dataset("shortnames", data=np.array([f"ch{c}" for c in range(C)], dtype=object), dtype=sdt)
        prov.create_dataset("uris", data=np.array([f"u{c}.cool" for c in range(C)], dtype=object), dtype=sdt)
        ch = f.create_group("chroms")
        ch.create_dataset("name", data=np.array(list(specs) + ["chrM"], dtype=object), dtype=sdt)  # chrM has no band
        ch.create_dataset("length", data=np.array([n * res for n in specs.values()] + [1000], dtype=np.int64))
        av = f.create_group("arms")
        av.create_dataset("name", data=np.array(["p", "p"], dtype=object), dtype=sdt)
        av.create_dataset("chrom", data=np.array(["chr1", "chr2"], dtype=object), dtype=sdt)
        av.create_dataset("start", data=np.array([0, 0], dtype=np.int64))
        av.create_dataset("end", data=np.array([specs["chr1"] * res, specs["chr2"] * res], dtype=np.int64))
        f.create_dataset("exp", data=exp)
        for ai, (chrom, nb) in enumerate(specs.items()):
            M = _symmetric_counts(nb, C, seed=ai)
            mats[chrom] = M
            band = band_from_dense(M, n_diag).astype(np.int16)
            weights = rng.uniform(0.2, 1.0, size=(C, nb)).astype(np.float32)
            bad = rng.random((C, nb)) < 0.05
            weights[bad] = 0.0
            arm_id = np.full(nb, ai, np.int32)
            fold_id = np.zeros(nb, np.int32)
            fold_id[nb // 2 :] = 1  # two contiguous folds per chrom
            g = f.create_group(chrom)
            g.create_dataset(  # Blosc-zstd+bitshuffle: the production band codec, exercised through BandedHicFile
                "band",
                data=band,
                chunks=(1, min(32, nb), n_diag),
                **hdf5plugin.Blosc(cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.BITSHUFFLE),
            )
            g.create_dataset("weights", data=weights)
            g.create_dataset("bad", data=bad)
            g.create_dataset("arm_id", data=arm_id)
            g.create_dataset("fold_id", data=fold_id)
        f.attrs["complete"] = True
    return specs, mats, exp


@pytest.fixture()
def banded_path(tmp_path):
    p = tmp_path / "mini.bhic.h5"
    _write_min_banded(p)
    return str(p)


# --------------------------------------------------------------------------- #
# BandedHicFile                                                               #
# --------------------------------------------------------------------------- #
def test_file_metadata_and_chroms(banded_path):
    bf = BandedHicFile(banded_path)
    assert bf.genome == "hg38" and bf.resolution == 1000 and bf.n_channels == 2
    assert bf.chroms == ["chr1", "chr2"]  # chrM has no band group -> excluded
    assert bf.shortnames == ["ch0", "ch1"] and bf.present_folds() == [0, 1]
    bf.close()


def test_file_window_matches_direct_band(banded_path):
    bf = BandedHicFile(banded_path)
    hic, w, exp = bf.get_window("chr1", 10 * bf.resolution, 16)  # start_bp -> bin 10
    ref = square_from_band(bf._band["chr1"][:], 10, 16)  # via the fully-materialized band
    np.testing.assert_array_equal(hic, ref)
    assert w.shape == (2, 16) and exp.shape == (2, bf.n_diag)
    bf.close()


def test_file_incomplete_rejected(tmp_path):
    p = tmp_path / "partial.bhic.h5"
    _write_min_banded(p)
    with h5py.File(p, "a") as f:
        del f.attrs["complete"]  # simulate a killed write
    with pytest.raises(ValueError, match="not marked complete"):
        BandedHicFile(str(p))


def test_is_eligible_matches_eligible_starts(banded_path):
    bf = BandedHicFile(banded_path)
    n = 16
    for chrom in bf.chroms:
        starts = set(int(a) for a in bf.eligible_starts(chrom, n))
        for a in range(bf.chrom_nbins[chrom] - n + 1):  # O(1) is_eligible must agree with the vectorized set
            assert bf.is_eligible(chrom, a * bf.resolution, n) == (a in starts)
    bf.close()


# --------------------------------------------------------------------------- #
# square_from_block equals square_from_band                                    #
# --------------------------------------------------------------------------- #
def test_square_from_block_equals_band():
    M = _symmetric_counts(60, 2, seed=3)
    band = band_from_dense(M, 24)
    a, n = 7, 16
    np.testing.assert_array_equal(square_from_block(band[:, a : a + n, :n], n), square_from_band(band, a, n))
