"""
Mixed-precision, multi-model Manta trainer.

One invocation co-trains **N Manta models** (one per Hi-C dataset) that share a single MicroZoi activation
cache, genome and resolution. Every training step fetches the shared activations for a window **once** and each
model that is *eligible* for that window trains on them, so 1 or 30 models cost almost the same in I/O. Models
are otherwise independent (own weights, optimizer, output channels).

How the data pipeline is organized (all precomputed up front, pure numpy):

* **Eligibility is a genome-wide mask, per (model, n_bins).** ``BandedHicFile.eligible_mask`` gives, over the
  global-position axis, the full inclusion mask (arm + bad-fraction + fold-fraction) for a fold set. We stack it
  across models, yielding, per n_bins, a *superset* of train (and val) global positions plus an
  ``(n_windows x n_models)`` eligibility matrix. A window is just one global bin index; no per-chromosome
  bookkeeping, no per-model random draws that would break the shared fetch.
* **Windows are sampled uniformly** over the superset (coverage proportional to the genome, not per-chromosome),
  with random reverse-complement.
* **One epoch = the genome seen ~once in both directions**, i.e. ``~2 * n_eligible`` bins of total window
  coverage, regardless of how many ``n_bins`` sizes are trained: each size draws the same window count
  (``2 * n_eligible / sum(n_bins)``), so variable-window training costs the same per epoch as single-window and
  the RES_EPOCHS schedule (calibrated at ``n_bins=[1024]``) transfers directly. A batch is homogeneous in
  n_bins so activations stack.
* **No activation cache.** Train and val are the same loop minus gradients; activations are fetched from disk,
  overlapped with GPU compute by a background prefetch thread. Validation windows are sampled once and frozen.

Precision: params are always ``float32``; ``--compute-dtype`` picks the autocast math dtype -- ``bfloat16``
(default, no scaler), ``float16`` (fp16 mixed, GradScaler on cuda/mps), or ``float32`` (plain fp32, no autocast).

Default ``--n-bins`` is ``512,768,896,960,1024`` -- every distributed model is trained on all of these (equal
window counts per size, sharing one genome-pass epoch) so it can be asked for any of them, and in particular for
a 512-bin map (laptop / fast sweeps). Pass a single value (e.g. ``--n-bins 512``) to train one size.

The checkpoint is the log: each model is one ``<output-dir>/<name>.pth`` re-saved as it trains, carrying its
per-epoch ``history`` (mean train/val loss + mean of each correlation) in the config. No side-car folders.

Specify models three interchangeable ways (they merge): ``--input-file`` (single); repeated ``--model
name=path``; or ``--models manifest.json`` (list of ``{name, input_file, [cache_path], [genome], [params]}``).
"""

import json
import os
import queue
import threading
import time

import click
import numpy as np
import torch
import torch.optim as optim

from manta_hic.io.banded import BandedHicFile
from manta_hic.nn.fetchers import CachedMicrozoiFetcher
from manta_hic.nn.manta import MANTA_PRESETS, Manta, save_manta_checkpoint
from manta_hic.ops.hic_ops import coarsegrained_hic_corrs, create_expected_matrix, hic_hierarchical_loss
from manta_hic.ops.tensor_ops import torch_device_type

CORR_NAMES = ["spearman", "pearson", "msd", "spearman_bm", "pearson_bm", "msd_bm"]
# How long to train the flagship per Hi-C resolution, from the 4096/8192/2048 convergence study (val 'combined'
# stops growing -- there is no overfitting decay, it just plateaus). Coarser resolution needs many more epochs
# (same number of *snippets* seen -- an epoch has fewer windows at coarse res). Two schedules:
#   RES_EPOCHS       -- full convergence (~99% of peak).
#   RES_EPOCHS_EARLY -- economical early stop (~95% of peak; where the curve visibly flattens).
# 256/512/1024 are provisional (under test); 2048..16384 are measured.
RES_EPOCHS = {256: 10, 512: 20, 1024: 30, 2048: 50, 4096: 80, 8192: 120, 16384: 160}
RES_EPOCHS_EARLY = {256: 7, 512: 14, 1024: 20, 2048: 30, 4096: 40, 8192: 50, 16384: 60}
N_BINS_DEFAULT = (512, 768, 896, 960, 1024)  # every distributed model sees these; equal window counts per size
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def resolve_specs(input_file, model, models, cache_path, genome, params):
    """Merge the three ways of naming models into a list of {name, input_file, cache_path, genome, params}."""
    specs = []

    def add(name, path, cache=None, gen=None, prm=None):
        specs.append(
            {
                "name": name,
                "input_file": path,
                "cache_path": cache or cache_path,
                "genome": gen or genome,
                "params": prm if prm is not None else params,
            }
        )

    if models:
        entries = json.load(open(models))
        if not isinstance(entries, list):
            raise ValueError("--models manifest must be a JSON list of objects")
        for e in entries:
            add(e["name"], e["input_file"], e.get("cache_path"), e.get("genome"), e.get("params"))
    for m in model or ():
        if "=" not in m:
            raise ValueError(f"--model must be 'name=path', got {m!r}")
        name, path = m.split("=", 1)
        add(name, path)
    if input_file:
        add(os.path.splitext(os.path.basename(input_file))[0], input_file)

    if not specs:
        raise ValueError("no models: pass --input-file, one/more --model name=path, or --models manifest.json")
    for s in specs:
        if s["cache_path"] is None:
            raise ValueError(f"model {s['name']!r} has no cache: pass --cache-path or set it in the manifest")
    names = [s["name"] for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate model names: {names}")
    return specs


class _Model:
    """Lightweight per-model holder: network + optimizer + scaler + banded target file + log. Eligibility lives in
    the shared per-n_bins index, not here. Params are always float32 (autocast handles the low-precision math)."""

    def __init__(self, spec, *, max_n_bins, bins_pad, device, lr, tower_h):
        self.name = spec["name"]
        self.banded = BandedHicFile(spec["input_file"])
        self.nch = self.banded.n_channels
        prm = dict(spec.get("params") or {})
        for k in ("tower_height", "n_bins", "bins_pad"):
            prm.pop(k, None)
        self.model_arch = prm
        # NOTE: ``.to(device)`` only -- never ``.to(dtype=...)`` on the module. Params are already float32 at
        # init, and ``Module.to(dtype)`` also casts complex buffers, silently turning the rotary ``freqs_cis``
        # table (complex64 e^{i*theta}) into its real part cos(theta) -- rotation degrades to scaling.
        self.model = Manta(
            n_bins=max_n_bins, bins_pad=bins_pad, tower_height=tower_h, output_channels=self.nch, **prm
        ).to(device)
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.history = []


def sample_n_runs(prob=0.1, lo=2, hi=6):
    """MicroZoi run-averaging depth for a training batch: usually 1 (fast), occasionally a random 2..hi (a
    smoother, more expensive activation) so the model sees both."""
    return int(np.random.randint(lo, hi + 1)) if np.random.rand() < prob else 1


def reduce_means(acc):
    """``acc[mi] = list of (loss, corr6|None)`` -> ``{mi: {'loss':, <corr means>...}}`` (epoch means per model)."""
    out = {}
    for mi, rows in acc.items():
        if not rows:
            continue
        d = {"loss": float(np.mean([r[0] for r in rows]))}
        corrs = [r[1] for r in rows if r[1] is not None]
        if corrs:
            mean = np.nanmean(np.array(corrs), axis=0)
            d.update({CORR_NAMES[k]: float(mean[k]) for k in range(len(CORR_NAMES))})
        out[mi] = d
    return out


def make_batches(sub, nb, batch_size, rng, sum_nb):
    """Sample this size's share of a genome-once-in-both-directions epoch: ``2 * n_eligible / sum_nb`` windows
    (rounded up to a batch multiple), where ``sum_nb`` is the total of ALL n_bins sizes trained in this run.
    Every size draws the same window count, and summed over sizes the epoch covers ``~2 * n_eligible`` bins --
    one forward + one reverse pass -- no matter how many sizes are trained (with a single size this reduces to
    the classic ``2 * n_eligible / n_bins``). Windows are drawn uniformly from a superset ``{pos, elig}`` and cut
    into homogeneous batches. Returns a list of ``(nb, positions, elig, reverse)`` tuples of length
    ``batch_size``."""
    nwin = int(np.ceil(2 * len(sub["pos"]) / sum_nb / batch_size) * batch_size)
    if nwin == 0:  # empty pool (e.g. no-holdout run's val split)
        return []
    rows = rng.integers(0, len(sub["pos"]), size=nwin)
    positions, elig, rev = sub["pos"][rows], sub["elig"][rows], (rng.random(nwin) < 0.5)
    return [
        (nb, positions[i : i + batch_size], elig[i : i + batch_size], rev[i : i + batch_size])
        for i in range(0, nwin, batch_size)
    ]


class _Prefetcher:
    """Single background thread (so the one h5py file set is touched by one thread only) that materializes each
    batch -- shared activation fetch + per-eligible-model target windows -- while the main thread runs the GPU.

    Failure handling (both directions, so neither side can ever hang the other):

    * Worker fails (missing chromosome, out-of-range fetch, arm-boundary window, ...): the exception object is
      put on the queue instead of a batch, and ``__iter__`` re-raises it in the main thread -- training dies
      loudly with the worker's traceback instead of blocking forever on an empty queue.
    * Consumer exits early (exception in the training step, generator closed): ``__iter__``'s ``finally`` sets
      a stop event; the worker's queue puts time out, notice it, and the thread returns instead of blocking
      forever on a full queue.
    """

    def __init__(self, batches, models, fetcher, *, bins_pad, res, n_runs_fn, queue_size=3):
        self.batches, self.models, self.fetcher = batches, models, fetcher
        self.bins_pad, self.res, self.n_runs_fn = bins_pad, res, n_runs_fn
        self.q = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()

    def _put(self, item):
        """Blocking put that gives up (returns False) once the consumer is gone."""
        while not self._stop.is_set():
            try:
                self.q.put(item, timeout=1.0)
                return True
            except queue.Full:
                continue
        return False

    def _work(self):
        try:
            for nb, positions, elig, rev in self.batches:
                acts = []
                for pos, rc in zip(positions, rev):
                    chrom, start_bp = self.models[0].banded.pos_to_coord(int(pos))  # same coord for all models
                    acts.append(
                        self.fetcher.fetch(
                            chrom,
                            start_bp - self.bins_pad * self.res,
                            start_bp + (nb + self.bins_pad) * self.res,
                            reverse=bool(rc),
                            n_runs=self.n_runs_fn(),
                            device="cpu",
                        )
                    )
                targets = {}
                for mi, m in enumerate(self.models):
                    rows = np.nonzero(elig[:, mi])[0]
                    if not len(rows):
                        continue
                    hs, ws, es = [], [], []
                    for r in rows:
                        hic, weight, exp = m.banded.window_at(int(positions[r]), nb)
                        if rev[r]:  # reverse-complement flips the map on both axes and the weights on one
                            hic, weight = hic[:, ::-1, ::-1], weight[:, ::-1]
                        hs.append(np.ascontiguousarray(hic))
                        ws.append(np.ascontiguousarray(weight))
                        es.append(np.ascontiguousarray(exp))
                    targets[mi] = (rows, np.stack(hs), np.stack(ws), np.stack(es))
                if not self._put((nb, torch.stack(acts), elig, targets)):
                    return
            self._put(None)  # normal end-of-batches sentinel
        except BaseException as e:  # noqa: B036 -- deliver ANY worker failure to the main thread
            self._put(e)

    def __iter__(self):
        t = self._thread = threading.Thread(target=self._work, daemon=True)
        t.start()
        try:
            while True:
                item = self.q.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item  # re-raise in the main thread; carries the worker's traceback
                yield item
        finally:
            self._stop.set()  # unblock the worker if we exited early (exception / generator close)
            t.join(timeout=60)  # bounded by one in-flight fetch; a daemon thread can't outlive the process anyway


def train_manta_multi(
    specs,
    output_dir,
    *,
    device="cuda:0",
    n_bins=N_BINS_DEFAULT,
    bins_pad=64,
    batch_size=8,
    n_epochs=0,
    epoch_multiplier=1.0,
    lr=2e-4,
    val_fold=3,
    test_fold=4,
    save_every=5,
    compute_dtype="bfloat16",
    max_bad_fraction=0.1,
    overlap_threshold=0.9,
    train_corr=False,
):
    os.makedirs(output_dir, exist_ok=True)
    n_bins = sorted({int(x) for x in ([n_bins] if isinstance(n_bins, int) else n_bins)})
    bad_nb = [nb for nb in n_bins if nb % 16]
    if bad_nb:  # hic_hierarchical_loss aggregates 2x2 blocks over 4 levels; Manta itself needs even n_bins
        raise ValueError(f"--n-bins values must be divisible by 16 (hierarchical loss), got {bad_nb}")
    max_nb = max(n_bins)
    dev_type = torch_device_type(device)
    # params are float32; compute_dtype is the autocast math dtype. fp32 -> no autocast; fp16 -> GradScaler.
    cdt = DTYPES[compute_dtype]
    autocast_on = cdt is not torch.float32
    scaler_on = cdt is torch.float16 and dev_type in ("cuda", "mps")
    mode = "fp32" if not autocast_on else ("mixed-fp16(scaler)" if scaler_on else "mixed-bf16")
    print(f"[train] precision: params float32, compute {compute_dtype} -> {mode}", flush=True)

    caches = set(s["cache_path"] for s in specs)
    genomes = set(s["genome"] for s in specs)
    if len(caches) != 1 or len(genomes) != 1:
        raise ValueError(
            f"all models in one run must share cache+genome (caches={caches}, genomes={genomes}); "
            "run separate invocations per cache/genome"
        )
    fetcher = CachedMicrozoiFetcher(next(iter(caches)))
    genome = next(iter(genomes))
    if fetcher.genome is not None and fetcher.genome != genome:
        raise ValueError(f"cache genome {fetcher.genome!r} != requested {genome!r}")

    probe = BandedHicFile(specs[0]["input_file"])  # resolution + tower height (all models must match)
    res = probe.resolution
    tower_h = int(np.round(np.log2(res))) - 9
    probe.close()

    models = [_Model(s, max_n_bins=max_nb, bins_pad=bins_pad, device=device, lr=lr, tower_h=tower_h) for s in specs]
    for m in models:
        if m.banded.resolution != res:
            raise ValueError(f"model {m.name!r} resolution {m.banded.resolution} != run resolution {res}")
        if m.banded.genome != genome:
            raise ValueError(f"model {m.name!r} genome {m.banded.genome!r} != run genome {genome!r}")
        # The genome axis must match EXACTLY (same chromosomes, same order, same per-chrom bin counts): the
        # prefetcher resolves every window's coordinate through models[0] while each model slices its own file at
        # the same global pos, so a reordered/subset chrom axis would silently pair activations with the wrong
        # targets. total_bins equality alone cannot catch a pure reordering.
        if m.banded.chroms != models[0].banded.chroms or m.banded.chrom_nbins != models[0].banded.chrom_nbins:
            raise ValueError(
                f"model {m.name!r} does not share the genome axis with {models[0].name!r}: "
                f"chroms {m.banded.chroms} (bins {m.banded.chrom_nbins}) != "
                f"{models[0].banded.chroms} (bins {models[0].banded.chrom_nbins})"
            )

    # train/val fold split (folds are integer Borzoi ids and must be present in the file -- else error, no silent
    # skip). --val-fold -1 = NO holdout: train on every fold, no validation (production models on an 'all' cache).
    all_folds = val_fold < 0
    present = set(models[0].banded.present_folds())
    if not all_folds and (val_fold not in present or test_fold not in present):
        raise ValueError(f"val_fold {val_fold} / test_fold {test_fold} not among the file's folds {sorted(present)}")
    train_folds = None if all_folds else present - {val_fold, test_fold}  # None -> eligible_mask skips fold rule
    val_folds = set() if all_folds else {val_fold}

    if n_epochs == 0:
        n_epochs = RES_EPOCHS.get(res, 50)
    n_epochs = max(1, int(n_epochs * epoch_multiplier))
    print(
        f"[train] res={res}bp n_bins={n_bins} bins_pad={bins_pad} genome={genome} epochs={n_epochs} "
        f"batch={batch_size} lr={lr} train_corr={train_corr}",
        flush=True,
    )

    # -- eligibility index: per n_bins, stack each model's eligible_mask for the train & val fold sets --------- #
    index = {}
    for nb in n_bins:
        entry = {}
        for split, fold_set in (("train", train_folds), ("val", val_folds)):
            if split == "val" and not val_folds:  # no-holdout run: empty val pool -> zero val batches downstream
                entry["val"] = {"pos": np.zeros(0, np.int64), "elig": np.zeros((0, len(models)), bool)}
                continue
            E = np.stack(
                [
                    m.banded.eligible_mask(
                        nb, max_bad_fraction=max_bad_fraction, fold=fold_set, overlap_threshold=overlap_threshold
                    )
                    for m in models
                ],
                axis=1,
            )
            pos = np.nonzero(E.any(axis=1))[0].astype(np.int64)
            entry[split] = {"pos": pos, "elig": E[pos]}  # elig: [S, M]
        index[nb] = entry
    for m in models:
        print(f"  - {m.name}: {m.nch}ch {m.n_params / 1e6:.2f}M", flush=True)
    for nb in n_bins:
        tr, va = index[nb]["train"], index[nb]["val"]
        print(
            f"  n_bins={nb}: train superset={len(tr['pos'])} val superset={len(va['pos'])} "
            f"(avg models/window {tr['elig'].mean() * len(models):.1f})",
            flush=True,
        )

    # frozen validation batches (sampled once)
    val_rng = np.random.default_rng(12345)
    val_batches = []
    for nb in n_bins:
        val_batches += make_batches(index[nb]["val"], nb, batch_size, val_rng, sum(n_bins))
    print(
        f"[train] frozen val: {sum(b[2].shape[0] for b in val_batches)} windows in {len(val_batches)} batches",
        flush=True,
    )

    scalers = [torch.GradScaler(dev_type, enabled=scaler_on) for _ in models]
    rng = np.random.default_rng(0)

    def run_batch(nb, acts, elig, targets, *, train):
        """One shared batch across all eligible models. Returns {mi: (loss, corr6|None)} of per-window means."""
        acts = acts.to(device=device, dtype=torch.float32)
        out = {}
        for mi, m in enumerate(models):
            if mi not in targets:
                continue
            rows, hic, weight, exp = targets[mi]
            sub = acts[rows]
            to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=torch.float32)
            hic_t, weight_t, exp_t = to_t(hic), to_t(weight), to_t(exp)
            target, weightmat = create_expected_matrix(hic_t, weight_t, exp_t)
            if train:
                m.model.train()
                m.opt.zero_grad()
            else:
                m.model.eval()
            with torch.set_grad_enabled(train), torch.autocast(dev_type, dtype=cdt, enabled=autocast_on):
                pred = m.model(sub)
                loss = hic_hierarchical_loss(pred, target, weightmat)
            if train:
                scalers[mi].scale(loss).backward()
                scalers[mi].step(m.opt)
                scalers[mi].update()
            corr = None
            if not train or train_corr:
                # Correlate over valid-bin pixels only: zero the bad bins in pred to match target (which
                # create_expected_matrix already zeroed there), so hic_corrs' nonzero mask drops them on both
                # sides. Done explicitly here -- the loss is pure and no longer zeroes pred as a side effect.
                pred_c = pred.detach().float().masked_fill(weightmat == 0, 0)
                cc = coarsegrained_hic_corrs(pred_c, target, weight_t, exp_t, also_divide_by_mean=True)
                corr = [float(np.nanmean(x.cpu().numpy())) for x in cc]
            out[mi] = (loss.detach().item(), corr)  # detach first: float() on a grad-carrying tensor warns
        return out

    meta = dict(
        lr=lr,
        batch_size=batch_size,
        n_bins=n_bins,
        bins_pad=bins_pad,
        n_epochs=n_epochs,
        compute_dtype=compute_dtype,
        precision_mode=mode,
        max_bad_fraction=max_bad_fraction,
        overlap_threshold=overlap_threshold,
        val_fold=val_fold,
        test_fold=test_fold,
        max_nb=max_nb,
    )

    for epoch in range(n_epochs):
        train_batches = []  # re-sampled each epoch; homogeneous-n_bins batches, shuffled together
        for nb in n_bins:
            train_batches += make_batches(index[nb]["train"], nb, batch_size, rng, sum(n_bins))
        rng.shuffle(train_batches)

        t0 = time.time()
        tr_acc = {mi: [] for mi in range(len(models))}
        for nb, acts, elig, targets in _Prefetcher(
            train_batches, models, fetcher, bins_pad=bins_pad, res=res, n_runs_fn=sample_n_runs
        ):
            for mi, r in run_batch(nb, acts, elig, targets, train=True).items():
                tr_acc[mi].append(r)
        t_train = time.time() - t0

        t0 = time.time()
        va_acc = {mi: [] for mi in range(len(models))}
        for nb, acts, elig, targets in _Prefetcher(
            val_batches, models, fetcher, bins_pad=bins_pad, res=res, n_runs_fn=lambda: 1
        ):
            for mi, r in run_batch(nb, acts, elig, targets, train=False).items():
                va_acc[mi].append(r)
        t_val = time.time() - t0

        tr_means, va_means = reduce_means(tr_acc), reduce_means(va_acc)
        line = [f"ep{epoch + 1}/{n_epochs} t_train={t_train:.0f}s t_val={t_val:.0f}s"]
        for mi, m in enumerate(models):
            rec = {"epoch": epoch}
            if mi in tr_means:
                rec["train"] = tr_means[mi]
            if mi in va_means:
                rec["val"] = va_means[mi]
            m.history.append(rec)
            vl = va_means.get(mi, {}).get("loss", float("nan"))
            vs = va_means.get(mi, {}).get("spearman", float("nan"))
            line.append(f"{m.name}:vl={vl:.3f}/vsp={vs:.3f}")
        print("[train] " + "  ".join(line), flush=True)

        if (epoch + 1) % save_every == 0 or (epoch + 1) == n_epochs:
            for m in models:
                save_manta_checkpoint(
                    m.model,
                    os.path.join(output_dir, f"{m.name}.pth"),
                    channel_names=m.banded.shortnames,
                    model_params=m.model_arch,
                    genome=m.banded.genome,
                    history=m.history,
                    train_meta=meta,
                )

    for m in models:
        m.banded.close()
    print("[train] DONE", flush=True)


def train_manta(
    input_file,
    cache_path,
    output_folder,
    *,
    device="cuda:0",
    genome="hg38",
    params=None,
    batch_size=8,
    n_epochs=0,
    lr=2e-4,
    save_every=5,
    n_bins=N_BINS_DEFAULT,
    bins_pad=64,
    val_fold=3,
    test_fold=4,
    epoch_multiplier=1.0,
    compute_dtype="bfloat16",
):
    """Train a single Manta model (thin wrapper over :func:`train_manta_multi`)."""
    name = os.path.splitext(os.path.basename(input_file))[0]
    specs = [{"name": name, "input_file": input_file, "cache_path": cache_path, "genome": genome, "params": params}]
    train_manta_multi(
        specs,
        output_folder,
        device=device,
        n_bins=n_bins,
        bins_pad=bins_pad,
        batch_size=batch_size,
        n_epochs=n_epochs,
        epoch_multiplier=epoch_multiplier,
        lr=lr,
        save_every=save_every,
        val_fold=val_fold,
        test_fold=test_fold,
        compute_dtype=compute_dtype,
    )


file = click.Path(exists=True, dir_okay=False)


@click.command(name="manta", context_settings={"show_default": True})
@click.option("--input-file", "-i", type=file, default=None, help="Single dataset (convenience; name from filename).")
@click.option("--model", "-m", multiple=True, help="Repeatable 'name=path'. Add several to co-train.")
@click.option("--models", type=file, default=None, help="JSON manifest: list of {name, input_file, [cache_path]...}.")
@click.option(
    "--cache-path", "-c", type=file, default=None, help="Shared MicroZoi cache (SSD). Default for all models."
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False),
    required=True,
    help="Directory for the per-model <name>.pth checkpoints (each carries its own history).",
)
@click.option("--device", "-d", default="cuda:0", help="Torch device")
@click.option("--genome", "-g", default="hg38", help="Genome (all models must share it).")
@click.option(
    "--preset",
    type=click.Choice(sorted(MANTA_PRESETS)),
    default="opt1M",
    help="Architecture size preset (see manta.MANTA_PRESETS); opt1M is the recommended small model.",
)
@click.option(
    "--params",
    type=click.Path(exists=True),
    default=None,
    help="Architecture params JSON, merged on top of --preset (overrides individual keys).",
)
@click.option(
    "--n-bins",
    default=",".join(map(str, N_BINS_DEFAULT)),
    help="Hi-C map size(s) in bins, comma-separated for variable-window training. Pass one value to train one size.",
)
@click.option("--bins-pad", default=64, help="Padding bins.")
@click.option("--batch-size", default=8, help="Batch size (shared fetch; one n_bins per batch, sized for max n_bins).")
@click.option("--n-epochs", "-e", default=0, help="Epochs (0 = auto by resolution).")
@click.option("--epoch-multiplier", default=1.0, type=float, help="Scale the epoch count.")
@click.option("--lr", default=2e-4, help="Learning rate.")
@click.option("--val-fold", default=3, type=int,
              help="Validation fold (integer Borzoi id, must be in the file). -1 = NO holdout: train on every "
                   "fold with no validation (production models on an 'all' cache; --test-fold is then ignored).")
@click.option("--test-fold", default=4, type=int, help="Test fold (integer Borzoi id, held out, not evaluated here).")
@click.option("--save-every", default=5, help="Re-save each <name>.pth every N epochs (always at the end).")
@click.option(
    "--compute-dtype",
    type=click.Choice(["float16", "bfloat16", "float32"]),
    default="bfloat16",
    help="Autocast math dtype (params are always float32).",
)
@click.option("--max-bad-fraction", default=0.1, help="Reject windows with RMS-over-channels bad fraction >= this.")
@click.option(
    "--overlap-threshold",
    default=0.9,
    help="Min fraction of a window's bins in the fold set to keep it (fold-boundary tolerance).",
)
@click.option("--train-corr", is_flag=True, help="Also compute (expensive) train-set correlations each epoch.")
def train_manta_click(
    input_file,
    model,
    models,
    cache_path,
    output_dir,
    device,
    genome,
    preset,
    params,
    n_bins,
    bins_pad,
    batch_size,
    n_epochs,
    epoch_multiplier,
    lr,
    val_fold,
    test_fold,
    save_every,
    compute_dtype,
    max_bad_fraction,
    overlap_threshold,
    train_corr,
):
    arch = dict(MANTA_PRESETS[preset])  # size preset, then --params JSON overrides individual keys
    if params:
        arch.update(json.load(open(params)))
    specs = resolve_specs(input_file, model, models, cache_path, genome, arch)
    sizes = tuple(int(x) for x in str(n_bins).split(","))
    train_manta_multi(
        specs,
        output_dir,
        device=device,
        n_bins=sizes,
        bins_pad=bins_pad,
        batch_size=batch_size,
        n_epochs=n_epochs,
        epoch_multiplier=epoch_multiplier,
        lr=lr,
        val_fold=val_fold,
        test_fold=test_fold,
        save_every=save_every,
        compute_dtype=compute_dtype,
        max_bad_fraction=max_bad_fraction,
        overlap_threshold=overlap_threshold,
        train_corr=train_corr,
    )
