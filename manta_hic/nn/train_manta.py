"""
Mixed-precision, multi-model Manta trainer.

One invocation co-trains **N Manta models** (one per Hi-C dataset) that share a single MicroZoi activation
cache, genome and resolution: each training step fetches the shared activations **once** and every eligible
model trains on them, so training 1 or 30 models costs almost the same in I/O. Each model is fully
independent (its own weights, optimizer, output-channel count) -- nothing is shared across models except the
read-only activation fetch.

Design choices
--------------
* **Mixed precision by default.** Two dtype knobs, ``--compute-dtype`` (autocast math, default ``float16``)
  and ``--param-dtype`` (weight/optimizer storage, default ``float32``). Setting them **equal** turns
  autocast off and gives plain fixed-precision training -- and it never mis-fires: the GradScaler is enabled
  only for genuine fp16-mixed on cuda/mps, and both autocast and the scaler collapse to no-ops otherwise, so
  the *same* step code runs in every mode.
* **The checkpoint is the log.** No per-epoch folders or corr pickles: each model is a single
  ``<output-dir>/<name>.pth`` re-saved as it trains, carrying its whole per-epoch ``history`` (mean train/val
  loss + mean of each correlation) inside the checkpoint config. Tiny even at hundreds of epochs.
* **Deterministic eval.** A frozen validation set and a frozen held-in train-probe (same construction, train
  folds) are precomputed once -- activations fetched once, decompressed Hi-C cached -- and evaluated the same
  way every epoch, so the train-vs-val gap (overfitting) is measured on fixed snippets with no resampling noise.

Specify the models three interchangeable ways (they merge): a single ``--input-file``; repeated
``--model name=path`` options; or a ``--models manifest.json`` file (a list of ``{"name","input_file",...}``
objects) -- the manifest is the clean way to launch many. ``--cache-path`` / ``--genome`` / ``--params`` are
shared defaults any entry may override.
"""

import json
import os

import click
import numpy as np
import torch
import torch.optim as optim

from manta_hic.io.banded import BandedHicFile
from manta_hic.nn.dataset import train_val_test_folds
from manta_hic.nn.fetchers import CachedMicrozoiFetcher
from manta_hic.nn.manta import Manta, save_manta_checkpoint
from manta_hic.ops.hic_ops import coarsegrained_hic_corrs, create_expected_matrix, hic_hierarchical_loss
from manta_hic.ops.tensor_ops import list_to_tensor_batch, torch_device_type

# means of these six are logged (order matches coarsegrained_hic_corrs(..., also_divide_by_mean=True))
CORR_NAMES = ["spearman", "pearson", "msd", "spearman_bm", "pearson_bm", "msd_bm"]
# auto epochs per Hi-C resolution (bp) when --n-epochs 0
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
    """Mixed precision unless the two dtypes are equal. Returns (autocast_on, scaler_on) -- both False for
    fixed precision, so autocast and GradScaler degrade to no-ops and one code path serves every mode."""
    autocast_on = compute_dtype != param_dtype
    scaler_on = autocast_on and compute_dtype == torch.float16 and dev_type in ("cuda", "mps")
    return autocast_on, scaler_on


# ---------------------------------------------------------------------------------------------------------- #
# model spec resolution                                                                                       #
# ---------------------------------------------------------------------------------------------------------- #
def resolve_specs(input_file, model, models, cache_path, genome, params):
    """Merge the three ways of naming models into a list of dicts {name, input_file, cache_path, genome, params}."""
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

    if models:  # manifest file: list of {name, input_file, [cache_path], [genome], [params]}
        entries = json.load(open(models))
        if not isinstance(entries, list):
            raise ValueError("--models manifest must be a JSON list of objects")
        for e in entries:
            add(e["name"], e["input_file"], e.get("cache_path"), e.get("genome"), e.get("params"))
    for m in model or ():  # repeated --model name=path
        if "=" not in m:
            raise ValueError(f"--model must be 'name=path', got {m!r}")
        name, path = m.split("=", 1)
        add(name, path)
    if input_file:  # single --input-file convenience
        add(os.path.splitext(os.path.basename(input_file))[0], input_file)

    if not specs:
        raise ValueError("no models given: pass --input-file, one/more --model name=path, or --models manifest.json")
    for s in specs:
        if s["cache_path"] is None:
            raise ValueError(f"model {s['name']!r} has no cache: pass --cache-path or set it in the manifest")
    names = [s["name"] for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate model names: {names}")
    return specs


# ---------------------------------------------------------------------------------------------------------- #
# per-model state                                                                                             #
# ---------------------------------------------------------------------------------------------------------- #
class ModelState:
    """One dataset + its independent model, optimizer, scaler, eligibility sets and frozen eval sets."""

    def __init__(
        self,
        spec,
        *,
        n_bins,
        bins_pad,
        device,
        param_dtype,
        lr,
        val_fold,
        test_fold,
        val_per_size,
        probe_per_size,
        min_fraction,
        use_all_data,
    ):
        self.name = spec["name"]
        self.banded = BandedHicFile(spec["input_file"])
        self.res = self.banded.resolution
        self.nch = self.banded.n_channels
        self.bins_pad = bins_pad
        self.min_fraction = min_fraction
        tower_h = int(np.round(np.log2(self.res))) - 9
        prm = dict(spec.get("params") or {})
        prm.pop("tower_height", None)
        prm.pop("n_bins", None)
        prm.pop("bins_pad", None)
        self.model_arch = prm
        self.model = Manta(n_bins=n_bins, bins_pad=bins_pad, tower_height=tower_h, output_channels=self.nch, **prm).to(
            device=device, dtype=param_dtype
        )
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.history = []

        tr, va, te = train_val_test_folds(self.banded, val_fold, test_fold)
        self.train_folds = None if use_all_data else tr
        self.heldout = va | te
        # eligible train starts (bins): array for sampling + set for O(1) routing eligibility
        self.train_starts = {
            c: self.banded.eligible_starts(c, n_bins, min_fraction=min_fraction, fold=self.train_folds)
            for c in self.banded.chroms
        }
        self.train_set = {c: set(int(x) for x in self.train_starts[c]) for c in self.banded.chroms}
        if sum(len(v) for v in self.train_starts.values()) == 0:
            raise ValueError(f"model {self.name!r}: no eligible training windows at n_bins={n_bins}")
        # frozen deterministic window lists (bins), spread across the genome
        self.val_windows = self._frozen(n_bins, self.heldout, val_per_size) if not use_all_data else []
        self.probe_windows = self._frozen(n_bins, self.train_folds, probe_per_size)

    def _frozen(self, n_bins, folds, per_size):
        pool = [
            (c, int(x))
            for c in self.banded.chroms
            for x in self.banded.eligible_starts(c, n_bins, min_fraction=self.min_fraction, fold=folds)
        ]
        if not pool or per_size <= 0:
            return []
        idx = np.linspace(0, len(pool) - 1, min(per_size, len(pool))).astype(int)
        return [pool[i] for i in idx]

    def is_eligible(self, chrom, start):
        return start in self.train_set[chrom]

    def target_cpu(self, chrom, start, n_bins):
        """Decompressed (hic, weight, exp) for one window, on CPU as fp16 (cached; re-cloned per eval epoch)."""
        hic, w, e = self.banded.store(chrom).get_window(start, n_bins)
        return (torch.from_numpy(hic).half(), torch.from_numpy(w).half(), torch.from_numpy(e).half())


def fetch_acts(fetcher, chrom, start_bin, n_bins, bins_pad, res, device, n_runs=1):
    ms = start_bin * res
    return fetcher.fetch(
        chrom, ms - bins_pad * res, ms + (n_bins + bins_pad) * res, reverse=False, n_runs=n_runs, device=device
    )


# ---------------------------------------------------------------------------------------------------------- #
# core trainer                                                                                                #
# ---------------------------------------------------------------------------------------------------------- #
def train_manta_multi(
    specs,
    output_dir,
    *,
    device="cuda:0",
    n_bins=512,
    bins_pad=64,
    batch_size=8,
    n_epochs=0,
    epoch_multiplier=1.0,
    steps_per_epoch=0,
    lr=2e-4,
    val_fold="fold3",
    test_fold="fold4",
    val_per_size=100,
    probe_per_size=100,
    save_every=5,
    compute_dtype="float16",
    param_dtype="float32",
    min_fraction=0.1,
    use_all_data=False,
):
    os.makedirs(output_dir, exist_ok=True)
    dev_type = torch_device_type(device)
    cdt, pdt = parse_dtype(compute_dtype), parse_dtype(param_dtype)
    autocast_on, scaler_on = amp_settings(cdt, pdt, dev_type)
    mode = "fixed" if not autocast_on else ("mixed-fp16(scaler)" if scaler_on else f"mixed-{compute_dtype}")
    print(f"[train] precision: compute={compute_dtype} param={param_dtype} -> {mode}", flush=True)

    # all models in a run MUST share cache + genome + resolution (that's what lets one fetch feed all)
    caches = set(s["cache_path"] for s in specs)
    genomes = set(s["genome"] for s in specs)
    if len(caches) != 1 or len(genomes) != 1:
        raise ValueError(
            f"all models in one run must share cache+genome (got caches={caches}, genomes={genomes}); "
            "run separate invocations per cache/genome"
        )
    fetcher = CachedMicrozoiFetcher(next(iter(caches)))
    genome = next(iter(genomes))
    if fetcher.genome is not None and fetcher.genome != genome:
        raise ValueError(f"cache genome {fetcher.genome!r} != requested {genome!r}")

    states = []
    for s in specs:
        st = ModelState(
            s,
            n_bins=n_bins,
            bins_pad=bins_pad,
            device=device,
            param_dtype=pdt,
            lr=lr,
            val_fold=val_fold,
            test_fold=test_fold,
            val_per_size=val_per_size,
            probe_per_size=probe_per_size,
            min_fraction=min_fraction,
            use_all_data=use_all_data,
        )
        if st.banded.genome != genome:
            raise ValueError(f"model {st.name!r} genome {st.banded.genome!r} != run genome {genome!r}")
        states.append(st)
    res = states[0].res
    if any(st.res != res for st in states):
        raise ValueError(f"all models must share resolution; got {[st.res for st in states]}")

    if n_epochs == 0:
        n_epochs = RES_EPOCHS.get(res, 50)
    n_epochs = max(1, int(n_epochs * epoch_multiplier))

    print(
        f"[train] res={res}bp n_bins={n_bins} bins_pad={bins_pad} genome={genome} epochs={n_epochs} "
        f"batch={batch_size} lr={lr}",
        flush=True,
    )
    for st in states:
        print(
            f"  - {st.name}: {st.nch}ch {st.n_params / 1e6:.2f}M  "
            f"train_elig={sum(len(v) for v in st.train_starts.values())} "
            f"val={len(st.val_windows)} probe={len(st.probe_windows)}",
            flush=True,
        )

    # -- one-time frozen eval precompute: acts fetched once (shared by position), Hi-C decompressed once ------ #
    shared_acts = {}  # (chrom, start) -> acts on CPU (fp16)

    def eval_cache(windows, st):
        entries = []
        for c, start in windows:
            if (c, start) not in shared_acts:
                shared_acts[(c, start)] = fetch_acts(fetcher, c, start, n_bins, bins_pad, res, "cpu").half()
            entries.append((c, start, *st.target_cpu(c, start, n_bins)))
        return entries

    n_win = sum(len(st.val_windows) + len(st.probe_windows) for st in states)
    est_gb = n_win * (states[0].nch * n_bins * n_bins * 2) / 1e9  # rough truth-cache upper bound
    print(
        f"[train] precomputing frozen eval: ~{n_win} (val+probe) windows across {len(states)} models "
        f"(~{est_gb:.1f} GB truth cache, plus shared acts)",
        flush=True,
    )
    for st in states:
        st.val_entries = eval_cache(st.val_windows, st)
        st.probe_entries = eval_cache(st.probe_windows, st)
    if dev_type == "cuda":
        torch.cuda.empty_cache()
    print(f"[train] eval ready: {len(shared_acts)} shared activation windows cached", flush=True)

    @torch.no_grad()
    def evaluate(st, entries):
        """Mean loss + mean of each of the 6 correlations over a frozen entry list (eval mode, no grad)."""
        if not entries:
            return None
        st.model.eval()
        losses, corr_rows = [], []
        for i in range(0, len(entries), batch_size):
            chunk = entries[i : i + batch_size]
            acts = torch.stack([shared_acts[(c, s)] for c, s, *_ in chunk]).to(device=device, dtype=pdt)
            hic = list_to_tensor_batch([e[2].float() for e in chunk], device)
            weight = list_to_tensor_batch([e[3].float() for e in chunk], device)
            exp = list_to_tensor_batch([e[4].float() for e in chunk], device)
            target, weightmat = create_expected_matrix(hic, weight, exp)  # mutates hic->target, weight in place
            with torch.autocast(dev_type, dtype=cdt, enabled=autocast_on):
                out = st.model(acts)
                loss = hic_hierarchical_loss(out, target, weightmat)
            out = out.float()
            cc = coarsegrained_hic_corrs(out, target, weight, exp, also_divide_by_mean=True)  # weight now mutated
            losses.append(float(loss))
            corr_rows.append([float(np.nanmean(x.cpu().numpy())) for x in cc])
        means = np.nanmean(np.array(corr_rows), axis=0)
        return {"loss": float(np.mean(losses)), **{CORR_NAMES[k]: float(means[k]) for k in range(6)}}

    # -- steps per epoch: cover the union of eligible windows once (strided by n_bins/4), capped ------------- #
    if steps_per_epoch <= 0:
        union = set()
        for st in states:
            for c in st.banded.chroms:
                union |= {(c, int(x)) for x in st.train_starts[c]}
        stride = max(1, n_bins // 4)
        steps_per_epoch = min(6000, max(1, len({(c, s) for (c, s) in union if s % stride == 0}) // batch_size))
    print(f"[train] steps/epoch={steps_per_epoch}", flush=True)

    def sample_n_runs(prob=0.1, lo=2, hi=6):  # run-averaging augmentation (matches single-model trainer)
        return int(np.random.randint(lo, hi + 1)) if np.random.rand() < prob else 1

    rng = np.random.default_rng(0)
    scalers = {st.name: torch.GradScaler(dev_type, enabled=scaler_on) for st in states}

    import time

    for epoch in range(n_epochs):
        for st in states:
            st.model.train()
        t0 = time.time()
        for step in range(steps_per_epoch):
            lead = states[step % len(states)]  # rotate which model seeds the batch positions
            chroms = [c for c in lead.banded.chroms if len(lead.train_starts[c])]
            batch = []
            while len(batch) < batch_size:
                c = chroms[int(rng.integers(len(chroms)))]
                pool = lead.train_starts[c]
                batch.append((c, int(pool[int(rng.integers(len(pool)))])))
            acts = list_to_tensor_batch(
                [fetch_acts(fetcher, c, s, n_bins, bins_pad, res, device, sample_n_runs()) for c, s in batch], device
            ).to(dtype=pdt)
            for st in states:
                elig = [k for k, (c, s) in enumerate(batch) if st.is_eligible(c, s)]
                if not elig:
                    continue
                sub = acts[elig]
                tt = [st.target_cpu(c, s, n_bins) for (c, s) in (batch[k] for k in elig)]
                hic = list_to_tensor_batch([t[0].float() for t in tt], device)
                weight = list_to_tensor_batch([t[1].float() for t in tt], device)
                exp = list_to_tensor_batch([t[2].float() for t in tt], device)
                target, weightmat = create_expected_matrix(hic, weight, exp)
                st.opt.zero_grad()
                with torch.autocast(dev_type, dtype=cdt, enabled=autocast_on):
                    out = st.model(sub)
                    loss = hic_hierarchical_loss(out, target, weightmat)
                sc = scalers[st.name]
                sc.scale(loss).backward()
                sc.step(st.opt)
                sc.update()

        # -- epoch metrics: frozen train-probe + frozen val (means only) -> checkpoint history --------------- #
        line = [f"ep{epoch + 1}/{n_epochs} t={time.time() - t0:.0f}s"]
        for st in states:
            tr = evaluate(st, st.probe_entries)
            va = evaluate(st, st.val_entries)
            rec = {"epoch": epoch}
            if tr is not None:
                rec["train"] = tr
            if va is not None:
                rec["val"] = va
            st.history.append(rec)
            tc = tr["spearman"] if tr else float("nan")
            vc = va["spearman"] if va else float("nan")
            line.append(f"{st.name}:tr_sp={tc:.3f}/va_sp={vc:.3f}")
        print("[train] " + "  ".join(line), flush=True)

        if (epoch + 1) % save_every == 0 or (epoch + 1) == n_epochs:
            meta = dict(
                lr=lr,
                batch_size=batch_size,
                n_bins=n_bins,
                bins_pad=bins_pad,
                n_epochs=n_epochs,
                compute_dtype=compute_dtype,
                param_dtype=param_dtype,
                precision_mode=mode,
                val_fold=val_fold,
                test_fold=test_fold,
                steps_per_epoch=steps_per_epoch,
            )
            for st in states:
                save_manta_checkpoint(
                    st.model,
                    os.path.join(output_dir, f"{st.name}.pth"),
                    channel_names=st.banded.shortnames,
                    model_params=st.model_arch,
                    genome=st.banded.genome,
                    history=st.history,
                    train_meta=meta,
                )

    for st in states:
        st.banded.close()
    print("[train] DONE", flush=True)


# ---------------------------------------------------------------------------------------------------------- #
# backward-compatible single-model entry point                                                                #
# ---------------------------------------------------------------------------------------------------------- #
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
    use_all_data=False,
    epoch_multiplier=1.0,
    compute_dtype="float16",
    param_dtype="float32",
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
        use_all_data=use_all_data,
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
@click.option("--n-bins", default=512, help="Hi-C map size (bins).")
@click.option("--bins-pad", default=64, help="Padding bins.")
@click.option("--batch-size", default=8, help="Batch size (shared fetch).")
@click.option(
    "--n-epochs", "-e", default=0, help="Epochs (0 = auto by resolution: 1024:50 2048:100 4096:150 8192/16384:200)."
)
@click.option("--epoch-multiplier", default=1.0, type=float, help="Scale the epoch count.")
@click.option("--steps-per-epoch", default=0, help="Steps per epoch (0 = auto from eligible-window count).")
@click.option("--lr", default=2e-4, help="Learning rate.")
@click.option("--val-fold", default="fold3", help="Validation fold.")
@click.option("--test-fold", default="fold4", help="Test fold (held out, not evaluated here).")
@click.option("--val-per-size", default=100, help="Frozen validation windows per model.")
@click.option("--probe-per-size", default=100, help="Frozen held-in train-probe windows per model (overfitting gap).")
@click.option("--save-every", default=5, help="Re-save each <name>.pth every N epochs (always at the end).")
@click.option("--compute-dtype", default="float16", help="Autocast math dtype (float16/bfloat16/float32).")
@click.option(
    "--param-dtype", default="float32", help="Weight/optimizer dtype. Equal to compute-dtype => fixed precision."
)
@click.option("--min-fraction", default=0.1, help="Min valid-bin fraction for an eligible window.")
@click.option("--use-all-data", is_flag=True, help="Train on all folds (no validation split).")
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
    steps_per_epoch,
    lr,
    val_fold,
    test_fold,
    val_per_size,
    probe_per_size,
    save_every,
    compute_dtype,
    param_dtype,
    min_fraction,
    use_all_data,
):
    default_params = json.load(open(params)) if params else None
    specs = resolve_specs(input_file, model, models, cache_path, genome, default_params)
    train_manta_multi(
        specs,
        output_dir,
        device=device,
        n_bins=n_bins,
        bins_pad=bins_pad,
        batch_size=batch_size,
        n_epochs=n_epochs,
        epoch_multiplier=epoch_multiplier,
        steps_per_epoch=steps_per_epoch,
        lr=lr,
        val_fold=val_fold,
        test_fold=test_fold,
        val_per_size=val_per_size,
        probe_per_size=probe_per_size,
        save_every=save_every,
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
        min_fraction=min_fraction,
        use_all_data=use_all_data,
    )
