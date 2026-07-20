"""
Mixed-precision, multi-model Manta trainer.

One invocation co-trains **N Manta models** (one per Hi-C dataset) that share a single MicroZoi activation
cache, genome and resolution. Every training step fetches the shared activations for a window **once** and each
model that is *eligible* for that window trains on them, so 1 or 30 models cost almost the same in I/O. Models
are otherwise independent (own weights, optimizer, output channels).

How the data pipeline is organized (all precomputed up front, pure numpy):

* **Eligibility is a genome-wide array, per (model, n_bins).** ``BandedHicFile.eligible_mask`` gives, over the
  genome-wide global-position axis, the full inclusion mask (arm + bad-fraction + fold-fraction) for a fold set.
  We stack it across models on that shared axis, yielding, per n_bins, a *superset* of train (and val) global
  positions plus an ``(n_windows x n_models)`` eligibility matrix. No round-robin over chromosomes, no per-model
  random draws that would break the shared fetch; a window is just one global bin index.
* **Windows are sampled uniformly** over the superset (so coverage is proportional to genome, not per-chromosome
  -- big chromosomes are not under-sampled), with random reverse-complement.
* **One epoch = the genome seen ~once in both directions** = ``2 * n_eligible / n_bins`` windows (half the old
  stride-n_bins/4 epoch). With several ``n_bins`` values (variable-window training) each contributes its own
  windows; a batch is homogeneous in n_bins (so activations stack), and the per-batch size is drawn in
  batch-size blocks.
* **No activation cache.** Train and val are the same loop minus gradients; activations are always fetched from
  disk, overlapped with GPU compute by a background prefetch thread. Validation windows are sampled once and
  frozen (deterministic curves); training windows are re-sampled each epoch.

Precision: ``--compute-dtype`` (autocast math, default ``bfloat16``) and ``--param-dtype`` (weights, default
``float32``). Equal dtypes => fixed precision (autocast + scaler both no-op). GradScaler is enabled only for
genuine fp16-mixed on cuda/mps; bf16 needs no scaler.

The checkpoint is the log: each model is one ``<output-dir>/<name>.pth`` re-saved as it trains, carrying its
per-epoch ``history`` (mean train/val loss + mean of each correlation) in the config. No side-car folders.

Specify models three interchangeable ways (they merge): ``--input-file`` (single); repeated ``--model
name=path``; or ``--models manifest.json`` (list of ``{name, input_file, [cache_path], [genome], [params]}``).
``--cache-path`` / ``--genome`` / ``--params`` are shared defaults.
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
from manta_hic.nn.manta import Manta, save_manta_checkpoint
from manta_hic.ops.hic_ops import coarsegrained_hic_corrs, create_expected_matrix, hic_hierarchical_loss
from manta_hic.ops.tensor_ops import list_to_tensor_batch, torch_device_type

CORR_NAMES = ["spearman", "pearson", "msd", "spearman_bm", "pearson_bm", "msd_bm"]
RES_EPOCHS = {256: 20, 512: 30, 1024: 50, 2048: 100, 4096: 150, 8192: 200, 16384: 200}

_DTYPES = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "full": torch.float32,
}


def parse_dtype(s):
    key = str(s).lower()
    if key not in _DTYPES:
        raise ValueError(f"unknown dtype {s!r}; choose from {sorted(set(_DTYPES))}")
    return _DTYPES[key]


def amp_settings(compute_dtype, param_dtype, dev_type):
    """Mixed precision unless the two dtypes are equal. Returns (autocast_on, scaler_on) -- both False for fixed
    precision, so autocast and GradScaler degrade to no-ops and one code path serves every mode."""
    autocast_on = compute_dtype != param_dtype
    scaler_on = autocast_on and compute_dtype == torch.float16 and dev_type in ("cuda", "mps")
    return autocast_on, scaler_on


def _parse_fold(fold):
    """Accept a fold as ``int`` or ``"foldN"`` string (the banded file stores integer fold ids)."""
    if isinstance(fold, str):
        return int(fold[4:]) if fold.startswith("fold") else int(fold)
    return int(fold)


def train_val_test_folds(banded_file, val_fold, test_fold):
    """Standard 3-way split of the file's present Borzoi folds: ``(train_set, {val}, {test})`` as int sets."""
    val, test = _parse_fold(val_fold), _parse_fold(test_fold)
    present = set(banded_file.present_folds())
    return present - {val, test}, {val}, {test}


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
    """Lightweight per-model holder: the network + optimizer + scaler + its banded target file + log. Eligibility
    lives in the shared per-n_bins index, not here."""

    def __init__(self, spec, *, max_n_bins, bins_pad, device, param_dtype, lr, tower_h):
        self.name = spec["name"]
        self.banded = BandedHicFile(spec["input_file"])
        self.nch = self.banded.n_channels
        prm = dict(spec.get("params") or {})
        for k in ("tower_height", "n_bins", "bins_pad"):
            prm.pop(k, None)
        self.model_arch = prm
        self.model = Manta(
            n_bins=max_n_bins, bins_pad=bins_pad, tower_height=tower_h, output_channels=self.nch, **prm
        ).to(device=device, dtype=param_dtype)
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.history = []


def _rc_target(banded, pos, n_bins, reverse):
    """(hic, weight, exp) for the window at global ``pos``, reverse-complemented (flip map + weight) when set."""
    hic, weight, exp = banded.window_at(pos, n_bins)
    if reverse:
        hic = hic[:, ::-1, ::-1]
        weight = weight[:, ::-1]
    return np.ascontiguousarray(hic), np.ascontiguousarray(weight), np.ascontiguousarray(exp)


# ---------------------------------------------------------------------------------------------------------- #
# eligibility index: per n_bins, the train/val supersets (global positions) + (S x n_models) elig matrices    #
# ---------------------------------------------------------------------------------------------------------- #
def build_index(models, n_bins, train_folds, val_folds, max_bad_fraction, overlap_threshold):
    """For one ``n_bins`` and one fold split: per model, ``eligible_mask`` (arm + bad + fold-fraction) over the
    shared genome-wide position axis; stack the models into an ``(S x M)`` eligibility matrix and keep the
    positions at least one model is eligible for. Positions are global bin indices (same across the models)."""
    N = models[0].banded.total_bins
    for m in models:
        if m.banded.total_bins != N:
            raise ValueError(f"models do not share a genome axis (total_bins {m.banded.total_bins} != {N}): {m.name}")

    def split(fold_set):
        cols = [
            m.banded.eligible_mask(
                n_bins, max_bad_fraction=max_bad_fraction, fold=fold_set, overlap_threshold=overlap_threshold
            )
            for m in models
        ]
        E = np.stack(cols, axis=1)  # [n_cand, M]
        pos = np.nonzero(E.any(axis=1))[0].astype(np.int64)
        return {"pos": pos, "elig": E[pos]}  # elig: [S, M]

    return {"n_bins": n_bins, "train": split(train_folds), "val": split(val_folds)}


def _sample_windows(sub, n_windows, rng):
    """Draw ``n_windows`` uniform superset rows -> (positions[n], elig[n,M], reverse[n])."""
    rows = rng.integers(0, len(sub["pos"]), size=n_windows)
    return sub["pos"][rows], sub["elig"][rows], (rng.random(n_windows) < 0.5)


def _epoch_windows(n_eligible, n_bins, batch_size):
    """~genome once in both directions = 2 * n_eligible / n_bins windows, rounded up to a batch multiple."""
    w = int(np.ceil(2 * n_eligible / n_bins))
    return int(np.ceil(w / batch_size) * batch_size)


# ---------------------------------------------------------------------------------------------------------- #
# background prefetch: turns a list of batch specs into ready (acts, targets, elig) payloads                   #
# ---------------------------------------------------------------------------------------------------------- #
class _Prefetcher:
    """Single background thread (so the one h5py file set is touched by one thread only) that materializes each
    batch -- shared activation fetch + per-eligible-model target windows -- while the main thread runs the GPU."""

    def __init__(self, batches, models, fetcher, *, bins_pad, res, n_runs_fn, queue_size=3):
        self.batches, self.models, self.fetcher = batches, models, fetcher
        self.bins_pad, self.res, self.n_runs_fn = bins_pad, res, n_runs_fn
        self.q = queue.Queue(maxsize=queue_size)

    def _work(self):
        for spec in self.batches:
            nb, positions, elig, rev = spec
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
                    h, w, e = _rc_target(m.banded, int(positions[r]), nb, bool(rev[r]))
                    hs.append(h)
                    ws.append(w)
                    es.append(e)
                targets[mi] = (rows, np.stack(hs), np.stack(ws), np.stack(es))
            self.q.put((nb, torch.stack(acts), elig, targets))
        self.q.put(None)

    def __iter__(self):
        t = threading.Thread(target=self._work, daemon=True)
        t.start()
        while True:
            item = self.q.get()
            if item is None:
                break
            yield item
        t.join()


# ---------------------------------------------------------------------------------------------------------- #
# core trainer                                                                                                 #
# ---------------------------------------------------------------------------------------------------------- #
def train_manta_multi(
    specs,
    output_dir,
    *,
    device="cuda:0",
    n_bins=(512,),
    bins_pad=64,
    batch_size=8,
    n_epochs=0,
    epoch_multiplier=1.0,
    lr=2e-4,
    val_fold="fold3",
    test_fold="fold4",
    max_val_windows=2000,
    max_train_windows=0,
    save_every=5,
    compute_dtype="bfloat16",
    param_dtype="float32",
    max_bad_fraction=0.1,
    overlap_threshold=0.9,
    train_corr=False,
):
    os.makedirs(output_dir, exist_ok=True)
    n_bins = sorted({int(x) for x in ([n_bins] if isinstance(n_bins, int) else n_bins)})
    max_nb = max(n_bins)
    dev_type = torch_device_type(device)
    cdt, pdt = parse_dtype(compute_dtype), parse_dtype(param_dtype)
    autocast_on, scaler_on = amp_settings(cdt, pdt, dev_type)
    mode = "fixed" if not autocast_on else ("mixed-fp16(scaler)" if scaler_on else f"mixed-{compute_dtype}")
    print(f"[train] precision: compute={compute_dtype} param={param_dtype} -> {mode}", flush=True)

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

    # resolution + tower height from the first banded file (all must match)
    probe = BandedHicFile(specs[0]["input_file"])
    res = probe.resolution
    tower_h = int(np.round(np.log2(res))) - 9
    probe.close()

    models = [
        _Model(s, max_n_bins=max_nb, bins_pad=bins_pad, device=device, param_dtype=pdt, lr=lr, tower_h=tower_h)
        for s in specs
    ]
    for m in models:
        if m.banded.resolution != res:
            raise ValueError(f"model {m.name!r} resolution {m.banded.resolution} != run resolution {res}")
        if m.banded.genome != genome:
            raise ValueError(f"model {m.name!r} genome {m.banded.genome!r} != run genome {genome!r}")
    train_folds, val_folds, _ = train_val_test_folds(models[0].banded, val_fold, test_fold)

    if n_epochs == 0:
        n_epochs = RES_EPOCHS.get(res, 50)
    n_epochs = max(1, int(n_epochs * epoch_multiplier))

    print(
        f"[train] res={res}bp n_bins={n_bins} bins_pad={bins_pad} genome={genome} epochs={n_epochs} "
        f"batch={batch_size} lr={lr} train_corr={train_corr}",
        flush=True,
    )

    # -- eligibility index + frozen validation windows (sampled once) -------------------------------------- #
    index = {nb: build_index(models, nb, train_folds, val_folds, max_bad_fraction, overlap_threshold) for nb in n_bins}
    for m in models:
        print(f"  - {m.name}: {m.nch}ch {m.n_params / 1e6:.2f}M", flush=True)
    for nb in n_bins:
        tr, va = index[nb]["train"], index[nb]["val"]
        print(
            f"  n_bins={nb}: train superset={len(tr['pos'])} val superset={len(va['pos'])} "
            f"(avg models/window {tr['elig'].mean() * len(models):.1f})",
            flush=True,
        )

    val_rng = np.random.default_rng(12345)
    val_batches = []
    for nb in n_bins:
        va = index[nb]["val"]
        if not len(va["pos"]):
            continue
        nwin = min(_epoch_windows(len(va["pos"]), nb, batch_size), max_val_windows)
        positions, elig, rev = _sample_windows(va, nwin, val_rng)
        for i in range(0, nwin, batch_size):
            sl = slice(i, i + batch_size)
            val_batches.append((nb, positions[sl], elig[sl], rev[sl]))
    print(
        f"[train] frozen val: {sum(b[2].shape[0] for b in val_batches)} windows in {len(val_batches)} batches",
        flush=True,
    )

    def sample_n_runs(prob=0.1, lo=2, hi=6):
        return int(np.random.randint(lo, hi + 1)) if np.random.rand() < prob else 1

    scalers = [torch.GradScaler(dev_type, enabled=scaler_on) for _ in models]
    rng = np.random.default_rng(0)

    def run_batch(nb, acts, elig, targets, *, train):
        """One shared batch across all eligible models. Returns {mi: (loss, corr6|None)} of per-window means."""
        acts = acts.to(device=device, dtype=pdt)
        out = {}
        for mi, m in enumerate(models):
            if mi not in targets:
                continue
            rows, hic, weight, exp = targets[mi]
            sub = acts[rows]
            hic = list_to_tensor_batch([h for h in hic.astype(np.float32)], device)
            weight = list_to_tensor_batch([w for w in weight.astype(np.float32)], device)
            exp = list_to_tensor_batch([e for e in exp.astype(np.float32)], device)
            target, weightmat = create_expected_matrix(hic, weight, exp)
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
                cc = coarsegrained_hic_corrs(pred.detach().float(), target, weight, exp, also_divide_by_mean=True)
                corr = [float(np.nanmean(x.cpu().numpy())) for x in cc]
            out[mi] = (float(loss), corr)
        return out

    def reduce_means(acc):
        """acc[mi] = list of (loss, corr|None) -> {mi: {'loss':, corr means...}}."""
        res = {}
        for mi, rows in acc.items():
            if not rows:
                continue
            d = {"loss": float(np.mean([r[0] for r in rows]))}
            corrs = [r[1] for r in rows if r[1] is not None]
            if corrs:
                mean = np.nanmean(np.array(corrs), axis=0)
                d.update({CORR_NAMES[k]: float(mean[k]) for k in range(6)})
            res[mi] = d
        return res

    meta = dict(
        lr=lr,
        batch_size=batch_size,
        n_bins=n_bins,
        bins_pad=bins_pad,
        n_epochs=n_epochs,
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
        precision_mode=mode,
        max_bad_fraction=max_bad_fraction,
        overlap_threshold=overlap_threshold,
        val_fold=val_fold,
        test_fold=test_fold,
        max_nb=max_nb,
    )

    for epoch in range(n_epochs):
        # build this epoch's training batches: per n_bins, sample windows, split into batch-size blocks; shuffle blocks
        train_batches = []
        for nb in n_bins:
            tr = index[nb]["train"]
            if not len(tr["pos"]):
                continue
            nwin = _epoch_windows(len(tr["pos"]), nb, batch_size)
            if max_train_windows:
                nwin = min(nwin, int(np.ceil(max_train_windows / batch_size) * batch_size))
            positions, elig, rev = _sample_windows(tr, nwin, rng)
            for i in range(0, nwin, batch_size):
                sl = slice(i, i + batch_size)
                train_batches.append((nb, positions[sl], elig[sl], rev[sl]))
        rng.shuffle(train_batches)

        t0 = time.time()
        tr_acc = {mi: [] for mi in range(len(models))}
        loader = _Prefetcher(train_batches, models, fetcher, bins_pad=bins_pad, res=res, n_runs_fn=sample_n_runs)
        for nb, acts, elig, targets in loader:
            for mi, r in run_batch(nb, acts, elig, targets, train=True).items():
                tr_acc[mi].append(r)
        t_train = time.time() - t0

        t0 = time.time()
        va_acc = {mi: [] for mi in range(len(models))}
        vloader = _Prefetcher(val_batches, models, fetcher, bins_pad=bins_pad, res=res, n_runs_fn=lambda: 1)
        for nb, acts, elig, targets in vloader:
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
    n_bins=512,
    bins_pad=64,
    val_fold="fold3",
    test_fold="fold4",
    epoch_multiplier=1.0,
    compute_dtype="bfloat16",
    param_dtype="float32",
):
    """Train a single Manta model (thin wrapper over :func:`train_manta_multi`)."""
    name = os.path.splitext(os.path.basename(input_file))[0]
    specs = [{"name": name, "input_file": input_file, "cache_path": cache_path, "genome": genome, "params": params}]
    train_manta_multi(
        specs,
        output_folder,
        device=device,
        n_bins=(n_bins,),
        bins_pad=bins_pad,
        batch_size=batch_size,
        n_epochs=n_epochs,
        epoch_multiplier=epoch_multiplier,
        lr=lr,
        save_every=save_every,
        val_fold=val_fold,
        test_fold=test_fold,
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
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
@click.option("--params", type=click.Path(exists=True), default=None, help="Default architecture params JSON.")
@click.option(
    "--n-bins", default="512", help="Hi-C map size(s) in bins, comma-separated for variable-window (e.g. 512,768,1024)."
)
@click.option("--bins-pad", default=64, help="Padding bins.")
@click.option("--batch-size", default=8, help="Batch size (shared fetch; one n_bins per batch).")
@click.option("--n-epochs", "-e", default=0, help="Epochs (0 = auto by resolution).")
@click.option("--epoch-multiplier", default=1.0, type=float, help="Scale the epoch count.")
@click.option("--lr", default=2e-4, help="Learning rate.")
@click.option("--val-fold", default="fold3", help="Validation fold.")
@click.option("--test-fold", default="fold4", help="Test fold (held out, not evaluated here).")
@click.option("--max-val-windows", default=2000, help="Cap on frozen validation windows per n_bins.")
@click.option(
    "--max-train-windows", default=0, help="Cap on train windows per n_bins per epoch (0 = full auto-sized epoch)."
)
@click.option("--save-every", default=5, help="Re-save each <name>.pth every N epochs (always at the end).")
@click.option("--compute-dtype", default="bfloat16", help="Autocast math dtype (bfloat16/float16/float32).")
@click.option("--param-dtype", default="float32", help="Weight dtype. Equal to compute-dtype => fixed precision.")
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
    params,
    n_bins,
    bins_pad,
    batch_size,
    n_epochs,
    epoch_multiplier,
    lr,
    val_fold,
    test_fold,
    max_val_windows,
    max_train_windows,
    save_every,
    compute_dtype,
    param_dtype,
    max_bad_fraction,
    overlap_threshold,
    train_corr,
):
    default_params = json.load(open(params)) if params else None
    specs = resolve_specs(input_file, model, models, cache_path, genome, default_params)
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
        max_val_windows=max_val_windows,
        max_train_windows=max_train_windows,
        save_every=save_every,
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
        max_bad_fraction=max_bad_fraction,
        overlap_threshold=overlap_threshold,
        train_corr=train_corr,
    )
