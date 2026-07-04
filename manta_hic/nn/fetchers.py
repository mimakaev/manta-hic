"""
Read side of the MicroZoi activation cache: turn cached activations -- or, for mutations, a local MicroZoi
recompute spliced into the cache -- into Manta inputs.

- :class:`CachedMicrozoiFetcher` reads the HDF5 activation cache, averages cached runs (a training
  augmentation), and patches mutations. Its :meth:`fetch_activations_batch` is the **L1** engine of the
  spec-based inference stack (see docs/INFERENCE_SPEC_PLAN.md).
- :class:`SequenceFetcher` fetches one-hot sequence instead, for Akita-style one-shot models.

The offline builder that *writes* this cache is in ``nn/fill_cache.py``.
"""

from __future__ import annotations

import io
import json

import h5py
import hdf5plugin  # noqa: F401 -- registers the Blosc filter so caches written with it are readable
import numpy as np
import torch

from manta_hic.nn.microzoi import Microzoi
from manta_hic.nn.specs import Spec, mutation_span
from manta_hic.ops.seq_ops import make_seq_1hot
from manta_hic.ops.tensor_ops import list_to_tensor_batch

MICROZOI_RECEPTIVE_FIELD = 2**19 + 2**18
BIN_BP = 256
CACHE_OVERHANG_BP = 2**22


def create_microzoi_model_from_cache(cache_path, device="cuda", return_type="mha", **kwargs):
    """
    Create a MicroZoi model from the 'model_blob' recorded in the HDF5 cache file.
    This re-creates the exact Torch model used to produce the cached activations.

    Parameters
    ----------
    cache_path : str
        Path to the HDF5 file created by populate_microzoi_cache.
    device : str
        Torch device.
    kwargs : dict
        Additional keyword arguments to pass to the Microzoi constructor.

    Returns
    -------
    model : Microzoi
        The MicroZoi model, loaded from the blob, set to eval mode.
    """
    with h5py.File(cache_path, "r") as f:
        model_bytes = f["model_blob"][:].tobytes()
        params = json.loads(f.attrs["model_params"])

    # combine model parameters
    mod_args = params["model"]
    mod_args.update(kwargs)
    mod_args.update({"return_type": return_type})

    model = Microzoi(**mod_args).to(device)
    model.load_state_dict(torch.load(io.BytesIO(model_bytes), map_location=device, weights_only=True))
    model.eval()
    return model


class CachedMicrozoiFetcher(object):
    """
    Reads MicroZoi activations from a precomputed HDF5 cache, and can locally recompute MicroZoi to patch in
    mutations (the L1 spec engine :meth:`fetch_activations_batch`).

    The cache stores ``N_runs`` runs (each a different sub-bp shift / crop / tile offset) per chromosome and
    orientation, over [-CACHE_OVERHANG_BP, chrom_len_rounded + CACHE_OVERHANG_BP). Requested regions must be
    bin-aligned and inside that range.

    Spec inference pins one run per spec (``run_idx``). The plain :meth:`fetch` can also average ``n_runs``
    distinct runs, which the training loop uses as an augmentation (see ``train_manta``'s ``sample_n_runs``).
    """

    def __init__(self, cache_path, fasta_open=None, batch_size=4):
        """
        Parameters
        ----------
        cache_path : str
            Path to the HDF5 file with cached activations.
        fasta_open : pysam.FastaFile or compatible, optional
            FASTA handle; required only for the mutation-patching methods.
        batch_size : int
            Batch size for the MicroZoi recompute during mutation patching.
        """
        self.cache_path = cache_path
        self.fasta_open = fasta_open
        self.batch_size = batch_size
        with h5py.File(self.cache_path, "r") as f:
            self.N_runs = f.attrs["N_runs"]
            self.cache_overhang_bp = f.attrs["CACHE_OVERHANG_BP"]
            self.bin_bp = f.attrs["BIN_BP"]
            genome = f.attrs.get("genome")  # caches written before genome-baking lack it -> None
            self.genome = genome.decode() if isinstance(genome, bytes) else genome

    def _read_runs(self, chrom, start_bp, end_bp, reverse, run_indices, device):
        """
        Read cached activations for one or more runs and return their elementwise mean as a float16 torch
        tensor on ``device``.

        h5py hands us float16 numpy (unavoidable); everything after is torch on ``device`` -- on GPU
        float16 is native, so averaging is cheap and the result can be fed straight into Manta. A single
        run is returned as-is; several runs are averaged in float32 then cast back to float16 (the model
        sees float16 under autocast anyway, and on real data the float32-vs-float16 averaging difference is
        ~400x below the genuine run-to-run variation).
        """
        if (start_bp % self.bin_bp) != 0 or (end_bp % self.bin_bp) != 0:
            raise ValueError("start_bp and end_bp must be multiples of BIN_BP")
        orientation = "reverse" if reverse else "forward"
        region_start_bp = -self.cache_overhang_bp
        tensors = []
        with h5py.File(self.cache_path, "r") as f:
            for run_idx in run_indices:
                run_group = f[f"run{int(run_idx)}"]
                ds_name = f"{chrom}_{orientation}"
                if ds_name not in run_group:
                    raise KeyError(f"Dataset not found: run{int(run_idx)}/{ds_name}")
                dset = run_group[ds_name]
                total_bins = dset.shape[1]
                region_end_bp = region_start_bp + total_bins * self.bin_bp
                if start_bp < region_start_bp or end_bp > region_end_bp:
                    raise ValueError(
                        f"Requested region [{start_bp}, {end_bp}) is outside stored range "
                        f"[{region_start_bp}, {region_end_bp})."
                    )
                i0 = (start_bp - region_start_bp) // self.bin_bp
                i1 = (end_bp - region_start_bp) // self.bin_bp
                arr = dset[:, i0:i1]
                if reverse:  # cache is stored forward; flip (copy: torch.from_numpy needs positive strides)
                    arr = arr[:, ::-1]
                tensors.append(torch.from_numpy(np.ascontiguousarray(arr)).to(device))
        if len(tensors) == 1:
            return tensors[0]
        return torch.stack(tensors).float().mean(dim=0).half()

    def fetch(
        self,
        chrom: str,
        start_bp: int,
        end_bp: int,
        reverse: bool = False,
        run_idx: int | None = None,
        n_runs: int = 1,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Fetch cached activations for [start_bp, end_bp) (bin-aligned, within the stored range) as a float16
        torch tensor on ``device`` (default "cpu").

        - ``run_idx`` given: that exact run (what L1 / spec inference uses -- one deterministic run per spec).
        - otherwise: average ``n_runs`` distinct random runs (``n_runs=1`` = one random run). The training
          loop uses this run-averaging as an augmentation (see ``train_manta``'s ``sample_n_runs``).
        """
        if run_idx is not None:
            return self._read_runs(chrom, start_bp, end_bp, reverse, [run_idx], device)
        return self._read_runs(chrom, start_bp, end_bp, reverse, self._pick_runs(n_runs), device)

    def _fetch_microzoi_model(self, device):
        """The MicroZoi model that produced this cache (built once, on first use, from the embedded blob)."""
        if not hasattr(self, "_model"):
            self._model = create_microzoi_model_from_cache(self.cache_path, device=device, return_type="mha")
        return self._model

    # ------------------------------------------------------------------ #
    # Recompute primitives (used by the L1 spec engine below)             #
    # ------------------------------------------------------------------ #
    #
    # Recompute MicroZoi over a window from mutated sequence and splice it into a cached background. The
    # matched-pair invariant is that outside the recomputed window WT and mutant share the identical cached
    # background, and inside both come from the same recompute, so ``mutant - WT`` is clean and the seam
    # cancels. Only length-preserving ops (``replace``/``invert``/``shuffle``) reach here; indels are
    # rejected upstream (they would break the fixed cache bin grid).

    def _splice(self, acts, win_lo, win_hi, patch, start_bp, end_bp, reverse):
        """Splice a recomputed window ``patch`` into a full-window activation tensor, in place."""
        if reverse:
            i0, i1 = (end_bp - win_hi) // BIN_BP, (end_bp - win_lo) // BIN_BP
        else:
            i0, i1 = (win_lo - start_bp) // BIN_BP, (win_hi - start_bp) // BIN_BP
        acts[:, i0:i1] = patch

    def _pick_runs(self, n_runs, run_idx=None):
        """Pick ``n_runs`` distinct cached run indices (or the pinned ``run_idx`` when n_runs == 1)."""
        if n_runs <= 1:
            return [int(run_idx) if run_idx is not None else int(np.random.randint(self.N_runs))]
        n = min(int(n_runs), int(self.N_runs))
        return [int(i) for i in np.random.choice(self.N_runs, size=n, replace=False)]

    # ------------------------------------------------------------------ #
    # L1: batched spec -> activations (the spec-based inference entry)     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _merge_runs(tiles) -> list[tuple[int, int]]:
        """Merge a Background's brick tiles into contiguous ``(lo, hi)`` recompute runs."""
        ts = sorted((int(lo), int(hi)) for lo, hi in tiles)
        runs = [list(ts[0])]
        for lo, hi in ts[1:]:
            if lo <= runs[-1][1]:
                runs[-1][1] = max(runs[-1][1], hi)
            else:
                runs.append([lo, hi])
        return [(lo, hi) for lo, hi in runs]

    def _recompute_jobs(self, model, jobs, crop_mha_bins, device):
        """
        Recompute a set of DISTINCT tile-run jobs, **pooling all their MicroZoi tiles into shared forward
        batches**. Each job is ``(chrom, win_lo, win_hi, mutate, reverse)`` (bin-aligned window, frozen
        mutation tuple or ``None``, orientation); ``crop_mha_bins`` is uniform across the jobs. Returns
        ``{job: patch}`` with ``patch`` a ``[n_channels, (win_hi-win_lo)//BIN_BP]`` float16 tensor on
        ``device``, spliceable at ``win_lo``. Geometry matches ``fetch_tile_microzoi_activations`` with no tile
        shift/offset. A recompute reads the fasta (not the cache), so a job is run-independent -- callers
        dedup jobs so a wild-type tile shared by several conditions is computed once.
        """
        tile_size_bins = MICROZOI_RECEPTIVE_FIELD // BIN_BP - 2 * crop_mha_bins
        tile_step_bp = tile_size_bins * BIN_BP

        tiles, metas = [], []  # metas: (num_tiles, end_off, reverse) per job, in order
        for chrom, win_lo, win_hi, mutate, reverse in jobs:
            num_bins = (win_hi - win_lo) // BIN_BP
            num_tiles = (num_bins + tile_size_bins - 1) // tile_size_bins
            seq_start = win_lo - crop_mha_bins * BIN_BP
            seq_end = win_lo + num_tiles * tile_step_bp + crop_mha_bins * BIN_BP
            seq = make_seq_1hot(self.fasta_open, chrom, seq_start, seq_end, reverse, mutate=mutate)
            job_tiles = [
                seq[i : i + MICROZOI_RECEPTIVE_FIELD]
                for i in range(0, len(seq) - (MICROZOI_RECEPTIVE_FIELD - tile_step_bp), tile_step_bp)
            ]
            assert len(job_tiles) == num_tiles
            tiles.extend(job_tiles)
            metas.append((num_tiles, num_tiles * tile_size_bins - num_bins, reverse))

        device_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
        outs = []
        for i in range(0, len(tiles), self.batch_size):  # one pooled pass through MicroZoi (+8 positional ch)
            batch = list_to_tensor_batch(tiles[i : i + self.batch_size], device)
            with torch.no_grad(), torch.autocast(device_type):
                act = model(batch.permute(0, 2, 1), genome="hg38", offset=0, crop_mha=crop_mha_bins)
                lin = torch.linspace(-1, 1, act.shape[2], device=device).unsqueeze(0).unsqueeze(0)
                lin = torch.cat([lin.repeat(act.shape[0], 1, 1).pow(p) for p in range(8)], dim=1)
                act = torch.cat([act, lin], dim=1)
            outs.append(act.detach())
        outs = torch.cat(outs, dim=0)  # [total_tiles, C+8, tile_size_bins]

        patches, pos = {}, 0
        for job, (num_tiles, end_off, reverse) in zip(jobs, metas):
            arr = outs[pos : pos + num_tiles].permute(1, 0, 2).reshape(outs.shape[1], -1)  # [C, num_tiles*N]
            pos += num_tiles
            arr = arr[:, end_off:] if reverse else arr[:, : arr.shape[1] - end_off]  # crop to the window
            patches[job] = arr.half()
        return patches

    @torch.no_grad()
    def fetch_activations_batch(
        self,
        specs: list[Spec],
        *,
        resolution: int,
        n_bins: int = 1024,
        bins_pad: int = 128,
        device: str = "cuda:0",
    ) -> list[torch.Tensor]:
        """
        Turn a list of specs (``manta_hic.nn.specs.Spec``) into their activation tensors on ``device``, ready
        for Manta -- the L1 engine of the spec-based inference stack (see docs/INFERENCE_SPEC_PLAN.md).

        Two levels of dedup within the batch: identical **cache reads** (specs whose Background shares a cache
        window, ``run_idx`` and orientation) happen once; and identical **recompute jobs** -- a tile window +
        the frozen mutations that land in it + orientation -- are pooled and computed once (a recompute is
        run-independent), so e.g. wild type and ``dE`` share the recompute of the promoter tile. Per spec the
        offset-0 tile pattern is rigidly shifted by ``tile_offset_bins``; each run is recomputed from only the
        mutations inside it (a run with none -> wild-type recompute, which makes a matched pair cancel) and
        spliced into a private copy of the cached background. ``bg.tiles is None`` -> pure cache read.

        Returns one ``[n_channels, n_bins_window]`` tensor per spec, in input order.
        """
        res = int(resolution)

        def cache_key(bg):
            fs, fe = bg.map_start_bp - bins_pad * res, bg.map_start_bp + (n_bins + bins_pad) * res
            return bg.chrom, fs, fe, bool(bg.reverse), int(bg.run_idx)

        # --- plan: collect distinct cache reads and distinct recompute jobs across the whole batch --------- #
        reads, jobs, crops = {}, {}, set()
        plan = []  # per spec: (cache_key, [(win_lo, win_hi, job)] | None)
        for spec in specs:
            bg = spec.bg
            ck = cache_key(bg)
            reads[ck] = None
            if bg.tiles is None:
                plan.append((ck, None))
                continue
            crops.add(bg.crop_mha_bins)
            shift = int(bg.tile_offset_bins) * BIN_BP
            _chrom, fs, fe = ck[0], ck[1], ck[2]
            runs = []
            for run_lo, run_hi in self._merge_runs(bg.tiles):
                win_lo, win_hi = run_lo + shift, run_hi + shift
                if win_lo < fs or win_hi > fe:  # a mutation too near the map edge pushes its tile out of range
                    raise ValueError(
                        f"recompute tile [{win_lo}, {win_hi}) falls outside the cached map window [{fs}, {fe}) "
                        f"(map_start_bp={bg.map_start_bp}); the mutation is too close to the map edge -- move "
                        f"the map window or the mutation inward."
                    )
                run_muts = tuple(
                    m for m in spec.mutations if run_lo <= mutation_span(m)[0] and mutation_span(m)[1] <= run_hi
                )
                job = (bg.chrom, win_lo, win_hi, run_muts or None, bool(bg.reverse))
                jobs[job] = None
                runs.append((win_lo, win_hi, job))
            plan.append((ck, runs))

        # --- do the reads and the single pooled recompute -------------------------------------------------- #
        reads = {ck: self.fetch(ck[0], ck[1], ck[2], reverse=ck[3], run_idx=ck[4], device=device) for ck in reads}
        patches = {}
        if jobs:
            assert len(crops) == 1, f"mixed crop_mha_bins {crops} not supported in one batch"
            patches = self._recompute_jobs(self._fetch_microzoi_model(device), list(jobs), crops.pop(), device)

        # --- assemble each spec's activations -------------------------------------------------------------- #
        out = []
        for ck, runs in plan:
            if runs is None:  # pure cache read
                out.append(reads[ck].clone())
                continue
            acts = reads[ck].clone()
            for win_lo, win_hi, job in runs:
                self._splice(acts, win_lo, win_hi, patches[job], ck[1], ck[2], ck[3])
            out.append(acts)
        return out


class SequenceFetcher(object):
    """
    A class that fetches sequences from a FASTA file, rather than intermediate activations.

    It can be used for "one shot" models that do not use transfer learning, like Akita.
    Accepts optional "max_shift_bp" argument to apply a random shift in basepairs.
    """

    def __init__(self, fasta_open, max_shift_bp=128):
        self.fasta_open = fasta_open
        self.max_shift_bp = max_shift_bp

    def fetch(self, chrom, start_bp, end_bp, reverse=True):
        """
        Fetch a sequence from a FASTA file, optionally reversed.
        """

        shift_bp = np.random.randint(-self.max_shift_bp, self.max_shift_bp + 1)
        return make_seq_1hot(self.fasta_open, chrom, start_bp + shift_bp, end_bp + shift_bp, reverse).T
