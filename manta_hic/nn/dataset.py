"""
Training/eval sampling over the banded Hi-C store (``io/banded.py``).

This is the reborn, thin ``HiCDataset``. The old one read a pre-tiled HDF5 (one row per accepted 1.25x square,
folds baked in by tile coordinates, a random sub-offset inside each square). The banded file instead stores
each chromosome's first ``n_diag`` diagonals once, with per-bin metadata (arm/fold/bad/weights), so tile
accept/reject and the fold split are *sampling-time* decisions over prefix sums -- not a baked build. What is
left here is genuinely training-specific: enumerate the eligible windows for a fold set, sample them (strided
for reproducible eval, random for training), reconstruct the target, and fetch the matching activations.

Because a candidate window starts at every bin, at 256 bp there are ~12 M eligible windows genome-wide. We do
not train on all of them: with ``stride`` (default ``n_bins // 4`` == 256 for the standard 1024-bin map) we
either take eligible starts on a ``stride`` grid (``sampling="strided"``, reproducible) or draw
``N_eligible // stride`` uniform-random eligible windows per epoch (``sampling="random"``) -- same epoch size
either way, decoupled from resolution.

The per-sample dict is unchanged (``acts``/``hic_slice``/``weight_slice``/``exp`` + coordinates), so
``run_epoch`` and the training loop are untouched.
"""

import datetime as dt
import queue
import threading
from contextlib import nullcontext

import numpy as np
import torch

from manta_hic.io.banded import BandedHicFile
from manta_hic.ops.hic_ops import (
    coarsegrained_hic_corrs,
    create_expected_matrix,
    hic_hierarchical_loss,
)
from manta_hic.ops.tensor_ops import list_to_tensor_batch, torch_device_type


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


class HiCDataset:
    """
    Sample training/eval windows from a banded Hi-C file, returning activations + reconstructed target.

    Parameters
    ----------
    banded_file : BandedHicFile | str
        An open banded file or a path to one (opened read-only).
    fetcher : CachedMicrozoiFetcher
        Supplies ``fetch(chrom, start_bp, end_bp, reverse, n_runs, device)`` activations for the padded window.
    n_bins : int
        Hi-C map side length in bins (the model's ``n_bins``).
    bins_pad : int
        Padding (in bins) added on each side of the map when fetching activations (the model's ``bins_pad``).
    folds : int | Iterable[int] | None
        Which Borzoi folds to sample. ``None`` uses all data with no fold constraint (windows may cross fold
        boundaries); an int/set requires each window to lie within a single fold that is in the set (the
        no-leakage train/val/test split). Use :func:`train_val_test_folds` to build the sets.
    min_fraction : float
        Max RMS-over-channels windowed bad-bin fraction for a window to be eligible.
    stride : int | None
        Grid step / epoch-size divisor in bins. Default ``n_bins // 4``.
    sampling : {"random", "strided"}
        ``"random"`` draws ``N_eligible // stride`` uniform-random eligible windows per epoch (training);
        ``"strided"`` yields the eligible starts on a ``stride`` grid, deterministically (eval/plotting).
    stochastic_reverse : bool | None
        Randomly reverse-complement windows. Defaults to ``True`` for random sampling, ``False`` for strided.
    n_runs : int | Callable[[], int]
        MicroZoi run-averaging depth per sample (an int, or a zero-arg callable evaluated per sample so the
        training loop owns the augmentation policy).
    device : str
        Device for the fetcher's activation tensor (keep ``"cpu"`` for a threaded loader).
    chroms : Iterable[str] | None
        Restrict to these chromosomes (default: all with a band group).
    """

    def __init__(
        self,
        banded_file,
        fetcher,
        *,
        n_bins=1024,
        bins_pad=128,
        folds=None,
        min_fraction=0.1,
        stride=None,
        sampling="random",
        stochastic_reverse=None,
        n_runs=1,
        device="cpu",
        chroms=None,
    ):
        if sampling not in ("random", "strided"):
            raise ValueError(f"sampling must be 'random' or 'strided', got {sampling!r}")
        self.file = BandedHicFile(banded_file) if isinstance(banded_file, (str, bytes)) else banded_file
        self.fetcher = fetcher
        self.n_bins = int(n_bins)
        self.bins_pad = int(bins_pad)
        self.min_fraction = float(min_fraction)
        self.stride = int(stride) if stride else max(1, self.n_bins // 4)
        self.sampling = sampling
        self.stochastic_reverse = (sampling == "random") if stochastic_reverse is None else bool(stochastic_reverse)
        self.n_runs = n_runs
        self.device = device
        self.n_channels = self.file.n_channels
        self.hic_res = self.file.resolution

        if folds is None:
            fold_arg = None
        elif np.isscalar(folds):
            fold_arg = _parse_fold(folds)
        else:
            fold_arg = {_parse_fold(x) for x in folds}
        self.folds = fold_arg

        self._chroms = list(chroms) if chroms is not None else list(self.file.chroms)

        # Build the eligible-window pool once: one eligible_starts pass per chromosome (the fold set is applied
        # inside that pass). Store as parallel (chrom_id, start_bin) int32 arrays -- ~48 MB each at 256 bp.
        chrom_ids, starts = [], []
        for ci, chrom in enumerate(self._chroms):
            s = self.file.eligible_starts(chrom, self.n_bins, min_fraction=self.min_fraction, fold=self.folds)
            if len(s):
                chrom_ids.append(np.full(len(s), ci, dtype=np.int32))
                starts.append(s.astype(np.int32))
        self._chrom_ids = np.concatenate(chrom_ids) if chrom_ids else np.empty(0, np.int32)
        self._starts = np.concatenate(starts) if starts else np.empty(0, np.int32)
        self.n_eligible = int(len(self._starts))

        if self.sampling == "strided":
            self._pool = np.nonzero(self._starts % self.stride == 0)[0]  # indices into the eligible pool
        else:
            self._pool = None  # random draws over the whole pool

    # -- one window -> the training dict ------------------------------------- #
    def _sample(self, chrom, start_bin, reverse, n_runs):
        hic, weight, exp = self.file.store(chrom).get_window(start_bin, self.n_bins)
        if reverse:  # reverse-complement flips the map on both axes (weights on one); distance-exp is unchanged
            hic = hic[:, ::-1, ::-1].copy()
            weight = weight[:, ::-1].copy()
        res = self.hic_res
        map_start_bp = start_bin * res
        map_end_bp = map_start_bp + self.n_bins * res
        fetch_start_bp = map_start_bp - self.bins_pad * res
        fetch_end_bp = map_end_bp + self.bins_pad * res
        acts = self.fetcher.fetch(
            chrom, fetch_start_bp, fetch_end_bp, reverse=reverse, n_runs=n_runs, device=self.device
        )
        return {
            "acts": acts,
            "hic_slice": hic,
            "weight_slice": weight,
            "exp": exp,
            "chrom": chrom,
            "start_bin": int(start_bin),
            "map_start_bp": map_start_bp,
            "map_end_bp": map_end_bp,
            "fetch_start_bp": fetch_start_bp,
            "fetch_end_bp": fetch_end_bp,
            "reverse": reverse,
        }

    def _resolve_n_runs(self):
        return self.n_runs() if callable(self.n_runs) else self.n_runs

    def get_slice_by_coords(self, chrom, start_bp, reverse=False):
        """Fetch the window at an exact ``(chrom, start_bp)`` (for eval/inference). Raises if the window is not
        a single-arm, in-bounds region (``get_window`` guards arm-crossing / excluded starts)."""
        return self._sample(chrom, self.file.bp_to_bin(start_bp), reverse, self._resolve_n_runs())

    def __len__(self):
        if self.sampling == "strided":
            return int(len(self._pool))
        return max(1, self.n_eligible // self.stride) if self.n_eligible else 0

    def __getitem__(self, idx):
        if self.sampling == "strided":
            p = int(self._pool[idx])
            reverse = False
        else:
            p = np.random.randint(self.n_eligible)  # idx is only an epoch counter in random mode
            reverse = self.stochastic_reverse and np.random.rand() > 0.5
        chrom = self._chroms[int(self._chrom_ids[p])]
        return self._sample(chrom, int(self._starts[p]), reverse, self._resolve_n_runs())


class ThreadedDataLoader:
    """
    Loads batches in a background thread through a bounded queue (so disk IO / activation reads overlap the
    main-thread GPU compute).

    Parameters
    ----------
    dataset : HiCDataset
    batch_size : int
    shuffle : bool
        Shuffle the drawn indices (irrelevant for random-sampling datasets, where each index is a fresh draw).
    fraction : float
        Fraction of ``len(dataset)`` to iterate per epoch (the dataset already sizes an epoch, so ``1.0`` is
        the natural default now; lower it to shorten epochs).
    queue_size : int
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, fraction=1.0, queue_size=3):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.fraction = fraction
        self.q = queue.Queue(maxsize=queue_size)

    def _loader_thread(self):
        need = int(len(self.dataset) * self.fraction)
        indices = np.random.choice(len(self.dataset), need, replace=False)
        if not self.shuffle:
            indices = np.sort(indices)
        for i in range(0, len(indices), self.batch_size):
            batch_idx = indices[i : i + self.batch_size]
            batch = [self.dataset[j] for j in batch_idx]
            if len(batch) < self.batch_size:
                break
            self.q.put(tuple(batch))
        self.q.put(None)

    def __iter__(self):
        thread = threading.Thread(target=self._loader_thread)
        thread.start()
        while True:
            data = self.q.get()
            if data is None:
                break
            yield data
        thread.join()


def run_epoch(model, dataloader, device, is_train=True, optimizer=None, scaler=None):
    corrs = []

    model.train() if is_train else model.eval()

    for batch in dataloader:
        t0 = dt.datetime.now()

        acts = list_to_tensor_batch([i["acts"] for i in batch], device)
        target = list_to_tensor_batch([i["hic_slice"] for i in batch], device)
        weight = list_to_tensor_batch([i["weight_slice"] for i in batch], device)
        exp = list_to_tensor_batch([i["exp"] for i in batch], device)
        target, weightmat = create_expected_matrix(target, weight, exp)

        with torch.autocast(torch_device_type(device)):
            if is_train:
                acts.requires_grad = True
                optimizer.zero_grad()

            with torch.no_grad() if not is_train else nullcontext():
                output = model(acts)
                if is_train:
                    loss = hic_hierarchical_loss(output, target, weightmat)
                corr = [i for i in coarsegrained_hic_corrs(output, target, weight, exp, also_divide_by_mean=True)]
                corr = np.array([i.detach().cpu().numpy() for i in corr])

            corrs.append(corr)

        if is_train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        duration = (dt.datetime.now() - t0).total_seconds()
        cr = ", ".join([f"{i.mean():.4f}" for i in corr])
        print(f"[{'Train' if is_train else 'Val'}] spearm/pears/msd = {cr}, duration={duration:.3f} s   ", end="\r")

    return np.array(corrs)
