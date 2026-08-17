"""
Genome-wide Manta prediction -> cooler export (the ``manta_hic infer`` CLI).

``save-cooler`` sweeps every chromosome arm of a trained checkpoint's genome with overlapping windows,
averages the overlaps, multiplies the model's observed-over-expected output by the banded target file's
per-arm expected (so the maps read like real balanced Hi-C), and streams one genome-wide ``.cool`` per output
channel -- in a single pass, with per-channel writer processes compressing while the GPU predicts (see
``manta_hic/io/predicted_cooler.py``). ``make-mcool`` then stacks per-resolution predictions of one track
into a browsable ``.mcool`` (the coarsest level is additionally zoomified upward).

Averaging is done consistently everywhere: a fixed set of cache runs (``--n-runs``, default 4, runs
``0..n-1``) plus the reverse-complement pass, for every window.
"""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import click
import numpy as np
import torch

from manta_hic.io.banded import BandedHicFile
from manta_hic.io.predicted_cooler import BandAccumulator, CoolerWriterPool, assemble_mcool
from manta_hic.nn.fetchers import CachedMicrozoiFetcher
from manta_hic.nn.inference import MantaInference


def _arm_segments(target: BandedHicFile, chrom: str) -> list[tuple[int, int, int]]:
    """Contiguous same-arm runs of ``chrom``'s bins as ``(lo_bin, hi_bin, arm_id)``, excluded regions dropped."""
    lo = target.chrom_start[chrom]
    arm_id = target.arm_id[lo : lo + target.chrom_nbins[chrom]]
    (breaks,) = np.nonzero(arm_id[1:] != arm_id[:-1])
    edges = np.concatenate([[0], breaks + 1, [len(arm_id)]])
    return [(int(a), int(b), int(arm_id[a])) for a, b in zip(edges[:-1], edges[1:]) if arm_id[a] != -1]


def _window_starts(target: BandedHicFile, chrom: str, n_bins: int, stride: int) -> list[tuple[int, int]]:
    """Ascending ``(start_bin, arm_id)`` prediction windows: each arm on a ``stride`` grid plus its tail window."""
    out = []
    for lo, hi, arm in _arm_segments(target, chrom):
        if hi - lo < n_bins:
            continue  # arm shorter than one window
        starts = list(range(lo, hi - n_bins + 1, stride))
        if starts[-1] != hi - n_bins:
            starts.append(hi - n_bins)  # cover the arm tail
        out.extend((s, arm) for s in starts)
    return out


def _group_starts(starts: list[tuple[int, int]], n_bins: int, group_size: int) -> list[list[tuple[int, int]]]:
    """Chunk the window list into fetch groups of overlapping neighbors (split where windows stop overlapping,
    e.g. across an arm gap, so a union read never spans a centromere)."""
    groups = []
    for s, arm in starts:
        if groups and len(groups[-1]) < group_size and s - groups[-1][-1][0] < n_bins:
            groups[-1].append((s, arm))
        else:
            groups.append([(s, arm)])
    return groups


@torch.no_grad()
def predict_genome_to_coolers(
    checkpoint,
    cache_path,
    target,
    out_dir,
    *,
    n_runs: int = 4,
    steps: int = 8,
    batch_windows: int = 2,
    fetch_group: int = 8,
    reverse_average: bool = True,
    device: str = "cuda:0",
    chroms=None,
    overwrite: bool = False,
    progress=print,
) -> dict[str, str]:
    """
    Predict the whole genome with a trained Manta checkpoint and write one ``.cool`` per output channel.

    Overlapping windows (``steps`` per window length) are averaged; each window averages ``n_runs`` fixed
    cache runs and, with ``reverse_average``, the reverse-complement pass. Consecutive windows overlap by
    ``(steps-1)/steps``, so the cache is read once per *group* of ``fetch_group`` neighboring windows (the
    union span, per run and orientation) and the windows are sliced out on the GPU -- without this the same
    activations would be decompressed ``steps`` times, and the cache read is the sweep's dominant cost.
    Output pixels are ``prediction * expected`` (per-arm expected from the banded ``target``), i.e.
    balanced-count scale. Returns ``{channel_name: cool_path}`` and prints a stage-timing summary
    (fetch / model / postprocess / writer backpressure) so the current bottleneck is visible.
    """
    target = BandedHicFile(target) if isinstance(target, (str, os.PathLike)) else target
    fetcher = CachedMicrozoiFetcher(cache_path)
    infer = MantaInference(checkpoint, fetcher, device=device, target=target)
    res, n_bins, pad, C = infer.resolution, infer.n_bins, infer.bins_pad, infer.output_channels
    stride = n_bins // steps
    run_idx = tuple(range(min(int(n_runs), int(fetcher.N_runs))))

    names = infer.channel_names or [f"ch{i}" for i in range(C)]
    paths = {n: os.path.join(out_dir, f"{re.sub(r'[^A-Za-z0-9._-]', '_', n)}_{res}.cool") for n in names}
    exists = [p for p in paths.values() if os.path.exists(p)]
    if exists and not overwrite:
        raise FileExistsError(f"output exists (pass overwrite): {exists[0]}")
    os.makedirs(out_dir, exist_ok=True)

    chroms = list(chroms) if chroms else target.chroms
    if unknown := [c for c in chroms if c not in target.chrom_start]:
        raise ValueError(f"no band for chromosome(s) {unknown} in {target.path} (available: {target.chroms})")
    chromsizes = {c: target.chrom_lengths[c] for c in target.chroms}  # full band chrom table, sweep or not

    # Per-arm expected, distances 0..n_bins-1. The first two diagonals are unreliable in cooltools expected
    # (0/NaN); fill them from d=2, as the training-side plots do, so the written map has a complete diagonal.
    exp = np.nan_to_num(np.asarray(target.exp[..., :n_bins], dtype=np.float32))
    exp[..., :2] = exp[..., 2:3]
    ii = torch.arange(n_bins)
    dist = (ii[:, None] - ii[None, :]).abs().to(device)

    win_bins = (n_bins + 2 * pad) * res // int(fetcher.bin_bp)  # model-input length in cache bins
    tasks, chrom_nstarts = [], {}
    for chrom in chroms:
        starts = _window_starts(target, chrom, n_bins, stride)
        chrom_nstarts[chrom] = len(starts)
        tasks += [(chrom, group) for group in _group_starts(starts, n_bins, fetch_group)]

    def fetch_task(task):
        """One group's cache read: the union span of its overlapping windows, per orientation (run-averaged)."""
        chrom, group = task
        fs = (group[0][0] - pad) * res
        fe = (group[-1][0] + n_bins + pad) * res
        fwd = fetcher.fetch(chrom, fs, fe, reverse=False, run_idx=run_idx, device=device)
        rev = fetcher.fetch(chrom, fs, fe, reverse=True, run_idx=run_idx, device=device) if reverse_average else None
        return fs, fwd, rev

    t_fetch = t_model = t_post = t_write = 0.0
    n_windows = 0
    t0 = time.perf_counter()
    acc, cur_chrom, done, triu_src = None, None, 0, None

    def _finalize_chrom(pool):
        nonlocal t_write
        t = time.perf_counter()
        for c in acc.finalize():
            pool.put_chunk(*c)
        t_write += time.perf_counter() - t
        progress(f"{cur_chrom}: done ({chrom_nstarts[cur_chrom]} windows)          ")

    # one-slot prefetch: the next group's cache read (h5py + Blosc release the GIL) overlaps the GPU forward
    with (
        CoolerWriterPool(list(paths.values()), chromsizes, res, assembly=infer.genome) as pool,
        ThreadPoolExecutor(max_workers=1) as ex,
    ):
        fut = ex.submit(fetch_task, tasks[0]) if tasks else None
        for k, (chrom, group) in enumerate(tasks):
            t = time.perf_counter()
            fs, fwd, rev = fut.result()
            t_fetch += time.perf_counter() - t  # time actually *blocked* on the read, net of the overlap
            fut = ex.submit(fetch_task, tasks[k + 1]) if k + 1 < len(tasks) else None
            if chrom != cur_chrom:
                if acc is not None:
                    _finalize_chrom(pool)
                acc = BandAccumulator(C, n_bins, target.chrom_nbins[chrom], chrom_offset=target.chrom_start[chrom])
                exp_mat = {}  # arm_id -> [C, n_bins, n_bins] expected matrix on device (a chromosome has ~2 arms)
                cur_chrom, done = chrom, 0

            for j in range(0, len(group), batch_windows):
                sub = group[j : j + batch_windows]
                t = time.perf_counter()
                acts = []
                for s, _arm in sub:
                    i0 = ((s - pad) * res - fs) // int(fetcher.bin_bp)
                    acts.append(fwd[:, i0 : i0 + win_bins])
                    if reverse_average:  # rev is the union span flipped, so the window slice flips too
                        acts.append(rev[:, rev.shape[1] - i0 - win_bins : rev.shape[1] - i0])
                with torch.autocast(infer._device_type()):
                    out = infer.model(torch.stack(acts).float()).float()
                if reverse_average:  # undo the RC map flip, average with the forward pass
                    out = 0.5 * (out[0::2] + torch.flip(out[1::2], dims=(-2, -1)))
                torch.cuda.synchronize(device) if str(device).startswith("cuda") else None
                t_model += time.perf_counter() - t

                t = time.perf_counter()
                if triu_src is None:
                    triu_src = torch.from_numpy(acc.triu_src).to(device)
                for (s, arm), m in zip(sub, out):
                    if arm not in exp_mat:
                        e = torch.from_numpy(exp[:, arm] if exp.ndim == 3 else exp).to(device)
                        exp_mat[arm] = e[:, dist]
                    # gather the upper triangle on-GPU: halves the device->host transfer and skips a numpy gather
                    vals = (m * exp_mat[arm]).reshape(C, -1)[:, triu_src].cpu().numpy()
                    chunks = acc.add_triu(s, vals)
                    t_post += time.perf_counter() - t
                    t = time.perf_counter()
                    for c in chunks:
                        pool.put_chunk(*c)
                    t_write += time.perf_counter() - t
                    t = time.perf_counter()
                done += len(sub)
                n_windows += len(sub)
                progress(f"{chrom}: {done}/{chrom_nstarts[chrom]} windows", end="\r")
        if acc is not None:
            _finalize_chrom(pool)
        t = time.perf_counter()
        pool.finish()
        t_write += time.perf_counter() - t

    wall = time.perf_counter() - t0
    progress(
        f"{n_windows} windows in {wall:.1f}s ({n_windows / max(wall, 1e-9):.2f}/s) -- "
        f"cache fetch {t_fetch:.1f}s, model {t_model:.1f}s, expected+aggregate {t_post:.1f}s, "
        f"cooler write wait {t_write:.1f}s"
    )
    return paths


@click.command(name="save-cooler", context_settings={"show_default": True})
@click.option("--checkpoint", "-m", type=click.Path(exists=True), required=True, help="Trained Manta checkpoint.")
@click.option("--cache", "-c", type=click.Path(exists=True), required=True, help="MicroZoi activation cache.")
@click.option(
    "--target",
    "-t",
    type=click.Path(exists=True),
    required=True,
    help="Banded .bhic.h5 the model was trained on (supplies expected + chromosome table).",
)
@click.option("--out-dir", "-o", type=click.Path(), required=True, help="Output directory for the .cool files.")
@click.option("--n-runs", "-n", default=4, help="Cache runs to average (runs 0..n-1, consistently everywhere).")
@click.option("--steps", default=8, help="Overlapping windows per window length (stride = n_bins/steps).")
@click.option("--batch-size", "-b", default=2, help="Windows per model forward (doubled by the reverse pass).")
@click.option("--fetch-group", default=8, help="Neighboring windows sharing one cache read (union span).")
@click.option("--no-reverse", is_flag=True, help="Skip the reverse-complement averaging pass.")
@click.option("--device", "-d", default="cuda:0", help="Torch device.")
@click.option("--chrom", multiple=True, help="Restrict to these chromosomes (default: all with a band).")
@click.option("--overwrite", is_flag=True, help="Overwrite existing output .cool files.")
def save_cooler_cli(
    checkpoint, cache, target, out_dir, n_runs, steps, batch_size, fetch_group, no_reverse, device, chrom, overwrite
):
    """Predict genome-wide Hi-C with a trained Manta model and stream one .cool per output channel."""
    paths = predict_genome_to_coolers(
        checkpoint,
        cache,
        target,
        out_dir,
        n_runs=n_runs,
        steps=steps,
        batch_windows=batch_size,
        fetch_group=fetch_group,
        reverse_average=not no_reverse,
        device=device,
        chroms=chrom or None,
        overwrite=overwrite,
        progress=lambda msg, end="\n": click.echo(msg, nl=(end == "\n")),
    )
    for name, path in paths.items():
        click.echo(f"{name}: {path}")


@click.command(name="make-mcool", context_settings={"show_default": True})
@click.argument("coolers", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), required=True, help="Output .mcool path.")
@click.option("--min-zoom-bins", default=256, help="Stop zoomifying once the genome has fewer bins than this.")
@click.option("--chunksize", default=20_000_000, help="Zoomify chunk size (pixels).")
@click.option("--nproc", "-p", default=8, help="Zoomify worker processes.")
def make_mcool_cli(coolers, output, min_zoom_bins, chunksize, nproc):
    """Stack per-resolution predicted COOLERS of one track into an .mcool (zoomifying the coarsest upward)."""
    resolutions = assemble_mcool(list(coolers), output, min_zoom_bins=min_zoom_bins, chunksize=chunksize, nproc=nproc)
    click.echo(f"{output}: resolutions {resolutions}")
