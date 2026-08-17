"""
Streaming cooler output for genome-wide *predicted* Hi-C maps.

The prediction sweep (``nn/save_cooler.py``) walks each chromosome with overlapping ``n x n`` windows whose
starts only ever increase. That monotonicity is the whole trick here: once the sweep has moved past a bin row,
no future window can touch it, so the row's accumulated mean is final and can leave RAM immediately.
:class:`BandAccumulator` keeps only a small rolling band buffer (``O(n_bins^2)`` per channel, not
``O(chrom_bins * n_bins)``) and emits finalized pixel chunks as it goes.

Those chunks stream straight into ``cooler.create_cooler``, which accepts an *iterator* of pixel chunks when
``ordered=True`` -- so each channel's genome-wide cooler is written in a single pass by a
:class:`CoolerWriterPool` worker process fed from a queue. No per-chromosome temporary coolers, no
``cooler.merge_coolers`` step, and compression runs in parallel across channels while the GPU keeps predicting.

This module is imported by the writer worker processes (``spawn``), so it must stay free of torch imports.
"""

from __future__ import annotations

import multiprocessing as mp
from queue import Full

import cooler
import numpy as np
import pandas as pd

__all__ = ["BandAccumulator", "CoolerWriterPool", "assemble_mcool"]


class BandAccumulator:
    """
    Rolling mean-accumulator for overlapping ``n_diag x n_diag`` prediction squares along one chromosome's
    diagonal.

    Cells live in banded coordinates ``(x, d) = (row bin, diagonal)``; a square added at ``start_bin``
    contributes its upper triangle (``M[r, c]`` with ``c >= r`` -> ``(start_bin + r, c - r)``). ``add`` requires
    non-decreasing ``start_bin`` and returns finalized pixel chunks (rows strictly below the new start, once at
    least ``min_flush_rows`` have accumulated); ``finalize`` flushes the rest. A chunk is
    ``(bin1, bin2, counts)`` with global bin ids (``chrom_offset`` added) and ``counts`` of shape
    ``[n_channels, n_pixels]`` -- the caller fans the per-channel rows out to the per-channel writers.
    """

    def __init__(self, n_channels: int, n_diag: int, chrom_nbins: int, *, chrom_offset: int = 0, min_flush_rows=1024):
        self.n_channels = int(n_channels)
        self.n_diag = int(n_diag)
        self.chrom_nbins = int(chrom_nbins)
        self.chrom_offset = int(chrom_offset)
        self.min_flush_rows = int(min_flush_rows)
        self._cap = self.n_diag + self.min_flush_rows  # rows in RAM; add() flushes before this overflows
        self._origin = 0  # bin index of buffer row 0 (rows below are already flushed)
        self._sums = np.zeros((self.n_channels, self._cap * self.n_diag), dtype=np.float32)
        self._cnts = np.zeros(self._cap * self.n_diag, dtype=np.uint16)
        # upper-triangle gather: square flat index -> band flat index (relative to the square's start row)
        r, c = np.triu_indices(self.n_diag)
        self._src = r * self.n_diag + c
        self._dst = r * self.n_diag + (c - r)

    @property
    def triu_src(self) -> np.ndarray:
        """Flat indices of the upper triangle in a ``[n_diag, n_diag]`` square -- the gather that turns a square
        into the ``vals`` accepted by :meth:`add_triu` (callers on a GPU can apply it there and ship half the
        bytes)."""
        return self._src

    def add(self, start_bin: int, square: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Fold one ``[n_channels, n_diag, n_diag]`` square in; return any newly finalized pixel chunks."""
        if square.shape != (self.n_channels, self.n_diag, self.n_diag):
            raise ValueError(f"square shape {square.shape} != ({self.n_channels}, {self.n_diag}, {self.n_diag})")
        return self.add_triu(start_bin, square.reshape(self.n_channels, -1)[:, self._src])

    def add_triu(self, start_bin: int, vals: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Like :meth:`add`, but takes the square's upper triangle already gathered via :attr:`triu_src`."""
        start_bin = int(start_bin)
        if start_bin < self._origin:
            raise ValueError(f"start_bin {start_bin} went backwards (already flushed up to {self._origin})")
        if start_bin + self.n_diag > self.chrom_nbins:
            raise ValueError(f"square at bin {start_bin} overflows the chromosome ({self.chrom_nbins} bins)")
        if vals.shape != (self.n_channels, len(self._src)):
            raise ValueError(f"vals shape {vals.shape} != ({self.n_channels}, {len(self._src)})")
        chunks = []
        if start_bin - self._origin >= self.min_flush_rows:
            chunks = self._emit(start_bin)
            if start_bin > self._origin:  # jump beyond the buffer (e.g. the next arm): it is empty now, skip ahead
                self._origin = start_bin
        base = (start_bin - self._origin) * self.n_diag
        self._sums[:, base + self._dst] += vals
        self._cnts[base + self._dst] += 1
        return chunks

    def finalize(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Flush everything still buffered (call once, after the last ``add`` of the chromosome)."""
        return self._emit(self._origin + self._cap)

    def _emit(self, upto_bin: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Finalize rows ``[origin, upto_bin)``: build their pixel chunk and slide the buffer window."""
        k = min(upto_bin, self._origin + self._cap) - self._origin
        if k <= 0:
            return []
        flat = min(k * self.n_diag, self._cap * self.n_diag)
        idx = np.nonzero(self._cnts[:flat])[0]
        chunks = []
        if idx.size:
            x, d = np.divmod(idx, self.n_diag)
            bin1 = (x + self._origin + self.chrom_offset).astype(np.int64)
            counts = self._sums[:, idx] / self._cnts[idx].astype(np.float32)
            chunks.append((bin1, bin1 + d.astype(np.int64), counts))
        keep = self._cap * self.n_diag - flat
        self._sums[:, :keep] = self._sums[:, flat:]
        self._sums[:, keep:] = 0.0
        self._cnts[:keep] = self._cnts[flat:]
        self._cnts[keep:] = 0
        self._origin += k
        return chunks


def _write_cooler_from_queue(queue, cool_path, chromsizes_items, resolution, assembly):
    """Worker-process target: stream pixel chunks from ``queue`` into one cooler (``None`` ends the stream)."""
    bins = cooler.util.binnify(pd.Series(dict(chromsizes_items)), resolution)
    bins["weight"] = 1.0  # predictions are already balanced; a unit weight keeps balanced-fetch APIs working

    def chunks():
        while True:
            item = queue.get()
            if item is None:
                return
            yield item

    cooler.create_cooler(
        cool_path,
        bins,
        chunks(),
        dtypes={"count": "float32"},
        assembly=assembly,
        ordered=True,
        symmetric_upper=True,
        boundscheck=False,
        dupcheck=False,
        triucheck=False,
        ensure_sorted=False,
    )


class CoolerWriterPool:
    """
    One writer process + queue per output cooler (channel), so HDF5 compression overlaps the GPU sweep.

    ``spawn`` is used (workers import only this torch-free module, and a fork after CUDA init is unsafe).
    Feed it with :meth:`put_chunk` (fans a :class:`BandAccumulator` chunk out to every channel) and call
    :meth:`finish`; use as a context manager so workers are terminated if the sweep dies mid-way.
    """

    def __init__(
        self, cool_paths: list[str], chromsizes: dict[str, int], resolution: int, *, assembly=None, queue_depth: int = 4
    ):
        ctx = mp.get_context("spawn")
        items = tuple(chromsizes.items())
        self.paths = list(cool_paths)
        self.queues, self.procs = [], []
        for path in self.paths:
            q = ctx.Queue(queue_depth)
            p = ctx.Process(target=_write_cooler_from_queue, args=(q, path, items, int(resolution), assembly))
            p.start()
            self.queues.append(q)
            self.procs.append(p)
        self._done = False

    def put_chunk(self, bin1: np.ndarray, bin2: np.ndarray, counts: np.ndarray) -> None:
        """Send one accumulator chunk: channel ``i``'s pixels ``(bin1, bin2, counts[i])`` go to writer ``i``."""
        for i, q in enumerate(self.queues):
            payload = {"bin1_id": bin1, "bin2_id": bin2, "count": np.ascontiguousarray(counts[i])}
            while True:  # bounded queue gives backpressure; poll liveness so a dead writer can't hang the sweep
                if not self.procs[i].is_alive():
                    raise RuntimeError(f"cooler writer for {self.paths[i]} died (exitcode {self.procs[i].exitcode})")
                try:
                    q.put(payload, timeout=5)
                    break
                except Full:
                    continue

    def finish(self) -> None:
        """Signal end-of-stream and wait for every writer to close its file."""
        for q in self.queues:
            q.put(None)
        for path, p in zip(self.paths, self.procs):
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"cooler writer for {path} failed (exitcode {p.exitcode})")
        self._done = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self._done:
            for p in self.procs:
                p.terminate()
            for p in self.procs:
                p.join()


def assemble_mcool(
    cool_paths: list[str], out_path: str, *, min_zoom_bins: int = 256, chunksize: int = 20_000_000, nproc: int = 8
) -> list[int]:
    """
    Combine same-track predicted coolers (one per resolution) into one ``.mcool``.

    The *coarsest* input is zoomified upward (doubling until the genome has fewer than ``min_zoom_bins`` bins),
    which also copies it in as the base layer; the finer inputs -- genuine predictions, better than aggregating
    a finer one -- are then copied in as-is. Returns the sorted resolution list of the output.
    """
    by_res = {}
    for path in cool_paths:
        res = cooler.Cooler(path).binsize
        if res in by_res:
            raise ValueError(f"two inputs at resolution {res}: {by_res[res]} and {path}")
        by_res[res] = path
    resolutions = sorted(by_res)
    coarsest = resolutions[-1]
    for fine, coarse in zip(resolutions, resolutions[1:]):
        if coarse != 2 * fine:
            raise ValueError(f"input resolutions must form a doubling chain, got {resolutions}")
    genome_bp = cooler.Cooler(by_res[coarsest]).chromsizes.sum()
    zooms = []
    res = coarsest * 2
    while genome_bp // res >= min_zoom_bins:
        zooms.append(int(res))
        res *= 2
    zooms = zooms or [int(coarsest * 2)]  # zoomify_cooler needs >= 1 level (it also copies in the base layer)
    cooler.zoomify_cooler(by_res[coarsest], out_path, zooms, chunksize=chunksize, nproc=nproc)
    for res in resolutions[:-1]:
        cooler.fileops.cp(by_res[res], f"{out_path}::resolutions/{res}")
    return sorted(resolutions + zooms)
