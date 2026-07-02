#!/usr/bin/env python
"""
Repack a MicroZoi activation cache (or any manta HDF5) to the Blosc-zstd+shuffle codec.

The old caches store their float16 activations with plain Zstandard level 9 (single-threaded). Re-encoding
them with Blosc-zstd + byte shuffle makes them ~10-15% smaller and, because Blosc decompresses
multi-threaded (BLOSC_NTHREADS, default 4), a training/inference fetch reads ~30% faster -- the cache-read
IO is the training/mutation hot path.

By default it also rounds the activations to ``--round-bits 4`` mantissa bits (lossy): a further ~1.7x
shrink for a predicted Hi-C map still correlated 0.999998 with full precision (the model sees f16/bf16 and
averages runs). Pass ``--round-bits 10`` for a lossless, bit-identical repack.

Design for a slow, one-file-at-a-time 5 TB sweep:
  * streams each dataset in blocks (bounded memory, ~`--block-mb` per block), never the whole [1032, ~1M]
    float16 array (2 GB) at once;
  * only recompresses the big chunked datasets (the activations); copies small/contiguous ones
    (``model_blob``, all attrs, group structure) verbatim;
  * writes to ``<dst>.tmp`` and atomically renames on success, so a killed run never leaves a file that
    looks complete;
  * ``--verify`` re-reads random slabs and asserts destination == round(source) (bit-identical if lossless);
  * skips a destination that already exists (unless ``--overwrite``), so you can just re-run the sweep.

Usage:
    python scripts/repack_cache.py SRC.h5 DST.h5 [--verify]
    # sweep a directory, one by one:
    for f in /path/microzoi_caches/*.h5; do
        python scripts/repack_cache.py "$f" "/path/repacked/$(basename "$f")" --verify || break
    done

It does not delete the source -- swap files yourself once you trust the output.
"""

import argparse
import os
import time

# Blosc filter reads BLOSC_NTHREADS from the environment; set a default before hdf5plugin/h5py touch a file.
os.environ.setdefault("BLOSC_NTHREADS", "4")

import h5py  # noqa: E402
import hdf5plugin  # noqa: E402
import numpy as np  # noqa: E402

from manta_hic.ops.tensor_ops import round_mantissa  # noqa: E402

RECOMPRESS_MIN_BYTES = 1 << 20  # only recompress chunked datasets bigger than this (the activation arrays)
CACHE_CHUNK_BINS = 4096  # rechunk activation datasets to (n_channels, 4096): faster large-window reads +
# ~50% faster multi-threaded writes than the old 1024, at ~same size (see manta.CACHE_CHUNK_BINS)


def _codec(clevel):
    return hdf5plugin.Blosc(cname="zstd", clevel=clevel, shuffle=hdf5plugin.Blosc.BITSHUFFLE)


def _target_chunk(din):
    """Rechunk 2D activation arrays [channels, bins] to (channels, CACHE_CHUNK_BINS); keep others as-is."""
    if din.ndim == 2:
        return (din.shape[0], min(CACHE_CHUNK_BINS, din.shape[1]))
    return din.chunks


def _maybe_round(block, round_bits):
    """Round float16 activation blocks to `round_bits` mantissa bits; leave other dtypes untouched."""
    if round_bits is not None and round_bits < 10 and block.dtype == np.float16:
        return round_mantissa(block, round_bits)
    return block


def _block_step(dset, axis, block_mb):
    """How many indices along ``axis`` fit in a ~block_mb slab."""
    bytes_per_index = dset.dtype.itemsize * (dset.size // max(1, dset.shape[axis]))
    return max(1, int(block_mb * (1 << 20) // max(1, bytes_per_index)))


def _recompress(din, gout, name, clevel, block_mb, round_bits):
    """Create ``name`` in ``gout`` with the Blosc codec and stream-copy ``din`` into it block by block,
    optionally rounding float16 activation blocks to ``round_bits`` mantissa bits (lossy)."""
    dout = gout.create_dataset(name, shape=din.shape, dtype=din.dtype, chunks=_target_chunk(din), **_codec(clevel))
    for k, v in din.attrs.items():
        dout.attrs[k] = v
    axis = int(np.argmax(din.shape))  # stream along the longest axis (bins, for [channels, bins])
    step = _block_step(din, axis, block_mb)
    for b0 in range(0, din.shape[axis], step):
        sl = tuple(slice(b0, min(b0 + step, din.shape[axis])) if i == axis else slice(None) for i in range(din.ndim))
        dout[sl] = _maybe_round(din[sl], round_bits)
    return dout, axis


def _verify(src_ds, dst_ds, axis, round_bits, n_trials=5, win=4096):
    """Assert several random slabs match: destination == round_mantissa(source) (== source when lossless)."""
    rng = np.random.default_rng(0)
    n = src_ds.shape[axis]
    w = min(n, win)
    for _ in range(n_trials):
        a = int(rng.integers(0, max(1, n - w)))
        sl = tuple(slice(a, a + w) if i == axis else slice(None) for i in range(src_ds.ndim))
        if not np.array_equal(_maybe_round(src_ds[sl], round_bits), dst_ds[sl]):
            return False
    return True


def _walk(gin, gout, clevel, block_mb, verify, round_bits, path=""):
    for name in gin:
        obj = gin[name]
        p = f"{path}/{name}"
        if isinstance(obj, h5py.Group):
            sub = gout.create_group(name)
            for k, v in obj.attrs.items():
                sub.attrs[k] = v
            _walk(obj, sub, clevel, block_mb, verify, round_bits, p)
        elif obj.chunks is not None and obj.nbytes > RECOMPRESS_MIN_BYTES:
            t0 = time.time()
            dout, axis = _recompress(obj, gout, name, clevel, block_mb, round_bits)
            src_mb = obj.id.get_storage_size() / 1e6
            dst_mb = dout.id.get_storage_size() / 1e6
            tag = ""
            if verify:
                if not _verify(obj, dout, axis, round_bits):
                    raise RuntimeError(f"VERIFY FAILED for {p} -- destination differs from expected")
                tag = " verified"
            print(
                f"    {p}: {src_mb:8.1f} -> {dst_mb:8.1f} MB ({src_mb/max(dst_mb,1e-9):.2f}x, "
                f"{time.time()-t0:.1f}s){tag}",
                flush=True,
            )
        else:
            gin.copy(name, gout)  # small / contiguous (e.g. model_blob) -- copy verbatim


def repack(src, dst, *, clevel=5, block_mb=128, verify=False, overwrite=False, round_bits=4):
    if os.path.exists(dst) and not overwrite:
        print(f"skip (exists): {dst}")
        return
    tmp = dst + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    t0 = time.time()
    lossy = "lossless" if round_bits is None or round_bits >= 10 else f"round to {round_bits} mantissa bits"
    print(
        f"repacking {src}\n       -> {dst}  (clevel={clevel}, threads={os.environ['BLOSC_NTHREADS']}, {lossy})",
        flush=True,
    )
    with h5py.File(src, "r") as fin, h5py.File(tmp, "w") as fout:
        for k, v in fin.attrs.items():
            fout.attrs[k] = v
        _walk(fin, fout, clevel, block_mb, verify, round_bits)
    os.replace(tmp, dst)  # atomic: only appears complete once fully written
    s, d = os.path.getsize(src) / 1e9, os.path.getsize(dst) / 1e9
    print(f"done: {s:.1f} -> {d:.1f} GB ({s/max(d,1e-9):.2f}x smaller) in {time.time()-t0:.0f}s\n", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--clevel", type=int, default=5, help="zstd level (5 fast+small; 9 ~22%% smaller, slower)")
    ap.add_argument("--block-mb", type=float, default=128, help="streaming block size per dataset (MB)")
    ap.add_argument("--threads", type=int, default=int(os.environ["BLOSC_NTHREADS"]), help="BLOSC_NTHREADS")
    ap.add_argument(
        "--round-bits",
        type=int,
        default=4,
        help="keep this many of float16's 10 mantissa bits (lossy, ~1.7x smaller at 4; "
        "10 = lossless). Predicted Hi-C map corr with full precision is 0.999998 at 4 bits.",
    )
    ap.add_argument("--verify", action="store_true", help="re-read random slabs and assert dst == round(src)")
    ap.add_argument("--overwrite", action="store_true", help="overwrite an existing destination")
    args = ap.parse_args()
    os.environ["BLOSC_NTHREADS"] = str(args.threads)
    repack(
        args.src,
        args.dst,
        clevel=args.clevel,
        block_mb=args.block_mb,
        verify=args.verify,
        overwrite=args.overwrite,
        round_bits=args.round_bits,
    )


if __name__ == "__main__":
    main()
