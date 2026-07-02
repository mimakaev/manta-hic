#!/usr/bin/env python
"""
Repack banded ``.bhic.h5`` Hi-C input files to the Blosc-zstd+shuffle codec with a small read chunk.

The first en-masse run wrote each chromosome's ``band`` with lzf and a ``(1, 8192, 1024)`` chunk. Re-encoding
``band`` with Blosc-zstd + byte shuffle and a ``(1, 512, n_diag)`` chunk makes it ~2.7x smaller and reads a
training/inference window faster: a 1024-bin window no longer decompresses an 8192-row chunk, and Blosc
decompresses multi-threaded (BLOSC_NTHREADS, default 4). This is lossless -- the int16 Hi-C counts go in and
come out bit-identical (no rounding; that trick is only for the float16 activation caches).

Everything else in the file (``weights``/``bad``/``arm_id``/``fold_id`` per chromosome, the ``exp`` table, and
the ``provenance``/``chroms``/``arms`` groups) is small and copied verbatim, so the repacked file is a
drop-in replacement readable by ``BandedHicFile`` exactly like the original.

Design (same as scripts/repack_cache.py): streams each band in blocks (bounded memory, never the whole
multi-GB chromosome band), writes to ``<dst>.tmp`` and atomically renames on success, ``--verify`` re-reads
random windows and asserts the band is bit-identical, and it skips a destination that already exists.

Usage:
    python scripts/repack_banded.py SRC.bhic.h5 DST.bhic.h5 --verify
    # sweep the whole directory (writing to a *writable* location -- the mount may be read-only):
    for f in /path/2024_manta_inputs/banded/*.bhic.h5; do
        python scripts/repack_banded.py "$f" "/writable/banded/$(basename "$f")" --verify || break
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

# Band chunk rows. Under Blosc (multi-threaded de/compress) bigger chunks are faster on BOTH axes than the
# tiny 128 that lzf preferred: 512 reads a 1024-bin window in ~3.2 ms (vs 4.7 at 128) and compresses at
# ~330 MB/s multi-threaded (vs ~120 single-threaded at 128 -- the small chunk starves Blosc's per-chunk
# block parallelism), at identical file size. 512 divides the 8192 write block.
READ_CHUNK = 512


def _codec(clevel):
    return hdf5plugin.Blosc(cname="zstd", clevel=clevel, shuffle=hdf5plugin.Blosc.BITSHUFFLE)


def _repack_band(din, gout, clevel, block_mb, verify):
    """Recreate ``band`` in ``gout`` with the Blosc codec + a (1, READ_CHUNK, n_diag) chunk, streamed."""
    C, nb, n_diag = din.shape
    chunk = (1, min(READ_CHUNK, nb), n_diag)
    dout = gout.create_dataset("band", shape=din.shape, dtype=din.dtype, chunks=chunk, **_codec(clevel))
    for k, v in din.attrs.items():
        dout.attrs[k] = v
    bytes_per_bin = C * n_diag * din.dtype.itemsize
    step = max(chunk[1], int(block_mb * (1 << 20)) // max(1, bytes_per_bin))
    step = max(chunk[1], (step // chunk[1]) * chunk[1])  # whole-chunk-aligned block along the bin axis
    for b0 in range(0, nb, step):
        b1 = min(b0 + step, nb)
        dout[:, b0:b1, :] = din[:, b0:b1, :]
    if verify:
        rng = np.random.default_rng(0)
        w = min(nb, 1024)
        for _ in range(4):
            a = int(rng.integers(0, max(1, nb - w)))
            if not np.array_equal(din[:, a : a + w, :], dout[:, a : a + w, :]):
                raise RuntimeError("band VERIFY FAILED -- repacked band differs from source")
    return dout


def repack(src, dst, *, clevel=5, block_mb=256, verify=False, overwrite=False):
    if os.path.exists(dst) and not overwrite:
        print(f"skip (exists): {dst}")
        return
    tmp = dst + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    t0 = time.time()
    print(f"repacking {src}\n       -> {dst}  (clevel={clevel}, threads={os.environ['BLOSC_NTHREADS']})", flush=True)
    with h5py.File(src, "r") as fin, h5py.File(tmp, "w") as fout:
        for k, v in fin.attrs.items():
            fout.attrs[k] = v
        for name in fin:
            obj = fin[name]
            if isinstance(obj, h5py.Group) and "band" in obj:  # a chromosome group: recompress band, copy rest
                gout = fout.create_group(name)
                for k, v in obj.attrs.items():
                    gout.attrs[k] = v
                for dsname in obj:
                    if dsname == "band":
                        t1 = time.time()
                        dnew = _repack_band(obj["band"], gout, clevel, block_mb, verify)
                        s = obj["band"].id.get_storage_size() / 1e6
                        d = dnew.id.get_storage_size() / 1e6
                        tag = " verified" if verify else ""
                        print(
                            f"    {name}/band: {s:8.1f} -> {d:8.1f} MB ({s/max(d,1e-9):.2f}x, "
                            f"{time.time()-t1:.1f}s){tag}",
                            flush=True,
                        )
                    else:
                        obj.copy(dsname, gout)  # weights / bad / arm_id / fold_id -- verbatim
            else:
                fin.copy(name, fout)  # exp dataset, or provenance/chroms/arms groups -- verbatim
    os.replace(tmp, dst)  # atomic: only appears complete once fully written
    s, d = os.path.getsize(src) / 1e9, os.path.getsize(dst) / 1e9
    print(f"done: {s:.2f} -> {d:.2f} GB ({s/max(d,1e-9):.2f}x smaller) in {time.time()-t0:.0f}s\n", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--clevel", type=int, default=5, help="zstd level (5 fast+small; 9 ~smaller, slower)")
    ap.add_argument("--block-mb", type=float, default=256, help="streaming block size per band (MB)")
    ap.add_argument("--threads", type=int, default=int(os.environ["BLOSC_NTHREADS"]), help="BLOSC_NTHREADS")
    ap.add_argument("--verify", action="store_true", help="re-read random windows and assert bit-identity")
    ap.add_argument("--overwrite", action="store_true", help="overwrite an existing destination")
    args = ap.parse_args()
    os.environ["BLOSC_NTHREADS"] = str(args.threads)
    repack(args.src, args.dst, clevel=args.clevel, block_mb=args.block_mb, verify=args.verify, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
