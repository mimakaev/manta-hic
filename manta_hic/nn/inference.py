"""
Turnkey Manta inference (``nn/inference.py``).

Ties a trained ``Manta`` checkpoint to a :class:`CachedMicrozoiFetcher` and hides the padding /
windowing / reverse-complement bookkeeping that callers used to redo by hand. One object predicts a Hi-C map
for a region, optionally averages the forward + reverse-complement passes, runs a wild-type-vs-mutant matched
pair, and (if you point it at the banded target file) hands back the observed map for the same window so you
can plot prediction and target side by side.

    infer = MantaInference("…/saved_model.pth", fetcher, target="…/4dn-diff_2048.bhic.h5")
    pred = infer.predict("chr1", 20_000_000)              # [output_channels, n_bins, n_bins] torch tensor
    obs  = infer.target("chr1", 20_000_000)               # observed-over-expected, same window (or None)
    wt, mut = infer.predict_pair("chr1", 20_000_000, [("replace", 20_500_000, "ACGT...")])

``output_channels`` is read from ``final_conv.weight`` and the resolution from the checkpoint's tower height,
so a checkpoint + fetcher is usually all you need. The one exception is the 256/512/1024 bp floor, which the
checkpoint alone can't tell apart -- there, attach a ``target`` (its resolution is authoritative) or pass
``tower_height`` (see :class:`MantaInference`).
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence

import numpy as np
import torch

from manta_hic.io.banded import BandedHicFile
from manta_hic.nn.manta import Manta
from manta_hic.nn.specs import Spec
from manta_hic.ops.hic_ops import create_expected_matrix


def tower_height_from_state(state: dict) -> int:
    """Manta's ``tower_height`` from a checkpoint = the number of ``conv_blocks_1d`` blocks.

    Ambiguous at the floor: ``Manta`` builds ``max(tower_height, 1)`` blocks, so tower_height -1/0/1 (256/512/
    1024 bp) all yield one block and are indistinguishable from the state dict. A count of 1 is reported as 1
    (1024 bp); pass ``tower_height`` or attach a ``target`` for a 256/512 bp model.
    """
    idxs = {int(m.group(1)) for k in state if (m := re.match(r"conv_blocks_1d\.(\d+)\.", k))}
    if not idxs:
        raise ValueError("checkpoint has no conv_blocks_1d.* keys; cannot infer the tower height")
    return max(idxs) + 1


def resolution_for_tower_height(tower_height: int) -> int:
    """Hi-C bin size in bp for a ``tower_height`` (microzoi 256 bp, +1 maxpool): ``2 ** (tower_height + 9)``."""
    return 2 ** (tower_height + 9)


def tower_height_for_resolution(resolution: int) -> int:
    """Inverse of :func:`resolution_for_tower_height`: ``round(log2(resolution)) - 9``."""
    return int(round(np.log2(resolution))) - 9


class MantaInference:
    """
    Predict Hi-C maps from a trained Manta model + a MicroZoi activation fetcher.

    Parameters
    ----------
    checkpoint : str | os.PathLike | dict
        Path to a ``saved_model.pth`` (a bare ``state_dict``) or an already-loaded state dict. ``output_channels``
        is read from it, and the resolution from its tower height -- except at the 256/512/1024 bp floor, which
        the checkpoint can't disambiguate (pass ``target`` or ``tower_height`` there).
    fetcher : CachedMicrozoiFetcher
        Supplies activations; must cover the queried chromosome/genome (and hold a fasta handle for mutations).
    device : str
        Device to run on (e.g. ``"cuda:0"``).
    n_bins, bins_pad : int
        Map side length and activation padding (must match training; the pretrained models use 1024 / 128).
    channel_names : list[str] | None
        Optional per-output-channel names. Defaults to the ``target`` file's shortnames when a target is given.
    target : BandedHicFile | str | None
        Optional banded file with the observed maps, enabling :meth:`target` / :meth:`is_eligible`. **Its
        resolution is authoritative** (it also disambiguates the 256/512/1024 bp floor); channel count is
        checked against the model.
    tower_height : int | None
        Override the tower height (else taken from ``target``, else the checkpoint). Needed for a 256/512 bp
        checkpoint loaded without a target.
    model_params : dict | None
        Extra ``Manta`` kwargs if the checkpoint was trained with non-default architecture params.
    """

    def __init__(
        self,
        checkpoint,
        fetcher,
        *,
        device: str = "cuda:0",
        n_bins: int = 1024,
        bins_pad: int = 128,
        channel_names: list[str] | None = None,
        target=None,
        tower_height: int | None = None,
        model_params: dict | None = None,
    ):
        obj = (
            checkpoint
            if isinstance(checkpoint, dict)
            else torch.load(checkpoint, map_location=device, weights_only=True)
        )

        self.target_file = None
        if target is not None:
            self.target_file = BandedHicFile(target) if isinstance(target, (str, bytes)) else target

        config_channel_names = None
        if "config" in obj and "state_dict" in obj:
            # self-describing checkpoint (save_manta_checkpoint): its config is authoritative -- no shape guessing
            state, cfg = obj["state_dict"], obj["config"]
            th, resolution = int(cfg["tower_height"]), int(cfg["resolution"])
            n_bins, bins_pad = int(cfg.get("n_bins", n_bins)), int(cfg.get("bins_pad", bins_pad))
            output_channels = int(cfg["output_channels"])
            config_channel_names = cfg.get("channel_names")
            model_params = model_params or cfg.get("model_params")
        else:
            # bare state dict: infer the shape. Resolution is ambiguous below 1024 bp, so a target (authoritative)
            # or an explicit tower_height disambiguates the 256/512/1024 floor (see tower_height_from_state).
            state = obj
            output_channels = int(state["final_conv.weight"].shape[0])
            if tower_height is not None:
                th = tower_height
            elif self.target_file is not None:
                th = tower_height_for_resolution(self.target_file.resolution)
            else:
                th = tower_height_from_state(state)
                if th == 1:  # one conv block -> could be tower_height -1/0/1 (256/512/1024 bp)
                    warnings.warn(
                        "bare checkpoint has one conv_blocks_1d block; 256/512/1024 bp models are structurally "
                        "identical, so resolution is assumed to be 1024 bp. Pass tower_height=/target=, or re-save "
                        "with save_manta_checkpoint, for a 256/512 bp model.",
                        stacklevel=2,
                    )
            resolution = resolution_for_tower_height(th)

        self.model = (
            Manta(
                n_bins=n_bins,
                bins_pad=bins_pad,
                tower_height=th,
                output_channels=output_channels,
                **(model_params or {}),
            )
            .to(device)
            .eval()
        )
        self.model.load_state_dict(state)

        if self.target_file is not None:
            if self.target_file.resolution != resolution:
                raise ValueError(
                    f"target resolution {self.target_file.resolution} bp != model resolution {resolution} bp "
                    f"(tower_height {th})"
                )
            if self.target_file.n_channels != output_channels:
                raise ValueError(
                    f"target has {self.target_file.n_channels} channels but the model outputs {output_channels}"
                )

        self.fetcher = fetcher
        self.resolution = resolution
        self.device = device
        self.n_bins = int(n_bins)
        self.bins_pad = int(bins_pad)
        self.output_channels = output_channels
        self.channel_names = (
            channel_names or config_channel_names or (self.target_file.shortnames if self.target_file else None)
        )

    # -- coordinate helper: model input window (padded) for a map at start_bp -- #
    def _fetch_span(self, start_bp):
        map_end = start_bp + self.n_bins * self.resolution
        return start_bp - self.bins_pad * self.resolution, map_end + self.bins_pad * self.resolution

    def _device_type(self):
        """``"cuda"``/``"cpu"`` for ``torch.autocast`` (which wants a device *type*), accepting either a
        device string (``"cuda:0"``) or a ``torch.device``."""
        return self.device.type if isinstance(self.device, torch.device) else str(self.device).split(":")[0]

    def _run(self, acts):
        """Run the model on a ``[in_channels, L]`` activation tensor -> ``[output_channels, n_bins, n_bins]``."""
        with torch.autocast(self._device_type()):
            return self.model(acts.unsqueeze(0).float().to(self.device))[0]

    @torch.no_grad()
    def predict(self, chrom, start_bp, *, n_runs=1, average_reverse=False, run_idx=None):
        """
        Predict the Hi-C map for the window ``[start_bp, start_bp + n_bins*resolution)``.

        Parameters
        ----------
        n_runs : int
            MicroZoi run-averaging depth (deeper = smoother activations, slower).
        average_reverse : bool
            Also run the reverse-complement pass and average (test-time augmentation).
        run_idx : int | None
            Pin a specific cached run (deterministic; ``n_runs`` is ignored). Use when only some runs exist.

        Returns
        -------
        torch.Tensor
            ``[output_channels, n_bins, n_bins]`` on ``device`` (float32).
        """
        fs, fe = self._fetch_span(start_bp)
        acts = self.fetcher.fetch(chrom, fs, fe, reverse=False, n_runs=n_runs, run_idx=run_idx, device=self.device)
        out = self._run(acts).float()
        if average_reverse:
            acts_r = self.fetcher.fetch(chrom, fs, fe, reverse=True, n_runs=n_runs, run_idx=run_idx, device=self.device)
            out_r = torch.flip(self._run(acts_r).float(), dims=(-2, -1))  # undo the RC map flip before averaging
            out = 0.5 * (out + out_r)
        return out

    @torch.no_grad()
    def predict_pair(self, chrom, start_bp, mutations, *, run_idx=0):
        """
        Predict ``(wt_map, mut_map)`` for the same window from one shared cache read (for mutation screens).

        ``mutations`` is one list of length-preserving edits applied together; random ops
        (``shuffle``/``inactivate``) are frozen here. ``run_idx`` pins the cached run. Built on the spec
        engine (:meth:`CachedMicrozoiFetcher.fetch_activations_batch`): WT and mutant share a
        Background, so their difference is clean.
        """
        from manta_hic.nn.specs import background_for, variant_specs
        from manta_hic.ops.seq_ops import freeze_mutations

        frozen = freeze_mutations(tuple(mutations), fasta=self.fetcher.fasta_open, chrom=chrom)
        bg = background_for(chrom, start_bp, run_idx=run_idx, mutations_superset=frozen)
        wt_acts, mut_acts = self.fetcher.fetch_activations_batch(
            variant_specs(bg, {"wt": None, "mut": frozen}),
            resolution=self.resolution,
            n_bins=self.n_bins,
            bins_pad=self.bins_pad,
            device=self.device,
        )
        return self._run(wt_acts).float(), self._run(mut_acts).float()

    # -- L2: spec-based inference (batched, forward-oriented, channel-sliced) - #
    def _resolve_channels(self, channels: Sequence[int | str] | None) -> tuple[list[int], list]:
        """Map a spec's ``channels`` (``None`` | ints | names) to (index list, name list)."""
        if channels is None:
            idx = list(range(self.output_channels))
        elif all(isinstance(c, str) for c in channels):
            lut = {n: i for i, n in enumerate(self.channel_names or [])}
            idx = [lut[c] for c in channels]
        else:
            idx = [int(c) for c in channels]
        names = [self.channel_names[i] if self.channel_names else i for i in idx]
        return idx, names

    @torch.no_grad()
    def _forward_flip_slice(
        self, specs: list[Spec], acts_list: list[torch.Tensor], batch_size: int
    ) -> list[tuple[torch.Tensor, list]]:
        """Manta-forward the specs' activations (in sub-batches), flip RC maps back to forward on-GPU, and
        slice to each spec's channels. Returns a list of ``(map, names)`` parallel to ``specs``."""
        dev_type = self._device_type()
        out = []
        for i in range(0, len(specs), batch_size):
            chunk = specs[i : i + batch_size]
            batch = torch.stack([a.float() for a in acts_list[i : i + batch_size]]).to(self.device)
            with torch.autocast(dev_type):
                maps = self.model(batch).float()  # [b, output_channels, n, n], native (maybe RC) orientation
            for spec, m in zip(chunk, maps):
                if spec.bg.reverse:  # RC map -> flip both axes back to canonical forward before it leaves here
                    m = torch.flip(m, dims=(-2, -1))
                idx, names = self._resolve_channels(spec.channels)
                out.append((m[idx].contiguous(), names))
        return out

    @torch.no_grad()
    def infer(self, specs: list[Spec], *, batch_size: int = 4) -> tuple[list[torch.Tensor], list]:
        """
        Predict a forward-oriented Hi-C map per spec: L1 activations -> Manta -> RC-flip -> channel slice.

        Returns ``(maps, names)``: ``maps`` a list of ``[C, n_bins, n_bins]`` tensors on ``device`` (one per
        spec, in input order); ``names`` the channel names (assumes the specs share a channel selection). Use
        this for small/explicit sweeps where you want every map; use :meth:`infer_grouped` to average over a
        big sweep without holding every map.
        """
        acts = self.fetcher.fetch_activations_batch(
            specs, resolution=self.resolution, n_bins=self.n_bins, bins_pad=self.bins_pad, device=self.device
        )
        results = self._forward_flip_slice(specs, acts, batch_size)
        return [m for m, _ in results], (results[0][1] if results else [])

    @torch.no_grad()
    def infer_grouped(
        self, specs: list[Spec], group_by: Sequence[str], *, batch_size: int = 4
    ) -> list[tuple[dict, torch.Tensor, list]]:
        """
        Average maps over a sweep, streaming so memory is O(#groups), not O(#specs).

        Processes one :class:`Background` at a time (so each cache read happens once and the whole sweep is
        never materialized), Manta-forwards + RC-flips + slices, then folds each forward-oriented map into an
        accumulator keyed by the spec's ``group_by`` tags. Everything not in ``group_by`` (runs, tile shifts,
        orientation, realizations) is averaged over -- "average the maps first, compare after". ``group_by=[]``
        averages everything into one map.

        Returns ``[(key, mean_map, names), ...]`` sorted by key, where ``key`` is a dict of the ``group_by``
        tags and ``mean_map`` is ``[C, n_bins, n_bins]`` on ``device``.
        """
        by_bg = {}  # Background -> its specs; groups the reads so each is done once, and lets us stream
        for spec in specs:
            by_bg.setdefault(spec.bg, []).append(spec)

        accum = {}  # group key -> [sum_map, count, names]
        for bg_specs in by_bg.values():
            acts = self.fetcher.fetch_activations_batch(
                bg_specs, resolution=self.resolution, n_bins=self.n_bins, bins_pad=self.bins_pad, device=self.device
            )
            for spec, (m, names) in zip(bg_specs, self._forward_flip_slice(bg_specs, acts, batch_size)):
                key = tuple((k, spec.tags.get(k)) for k in group_by)
                if key in accum:
                    accum[key][0] += m
                    accum[key][1] += 1
                else:
                    accum[key] = [m.clone(), 1, names]
        out = []
        for key in sorted(accum, key=str):  # deterministic group order
            total, count, names = accum[key]
            out.append((dict(key), total / count, names))
        return out

    # -- L3: one friendly call for a region ---------------------------------- #
    @torch.no_grad()
    def predict_region(
        self,
        chrom: str,
        start_bp: int,
        conditions: dict[str, list] | list,
        *,
        runs: int = 8,
        max_shift_bins: int = 100,
        rc: bool | str = "average",
        backgrounds: str = "matched",
        channels: Sequence[int | str] | None = None,
        rng_seed: int | None = None,
        batch_size: int = 4,
    ) -> tuple[dict[str, torch.Tensor], list]:
        """
        Predict one averaged, forward-oriented Hi-C map per condition at ``(chrom, start_bp)``.

        ``conditions`` is a dict ``{name: mutation_list}`` (or a list, auto-named ``wt``/``cond1``/...; a
        ``None`` / empty list is wild type). Each condition's map is averaged over ``runs`` cached runs,
        a random tile shift up to ``max_shift_bins`` bins, and (per ``rc``) orientation:

        - ``rc=False`` forward only; ``rc=True`` reverse only (returned forward); ``rc="average"`` averages
          forward + reverse.
        - ``backgrounds="matched"`` (default): all conditions share each background -- synchronized run / tile
          shift / tiling, so the per-replicate difference is the mutation alone. A polished comparison.
        - ``backgrounds="random"``: each condition draws its own independent backgrounds (and tiles only what
          it must -- wild type stays a pure cache read). Nothing is synchronized, so the spread across
          conditions shows the model's natural variability rather than an isolated mutation effect.

        Returns ``(maps, names)``: ``maps`` a dict ``{name: [C, n_bins, n_bins]}`` on ``device``, ``names``
        the channel names.
        """
        from manta_hic.nn.specs import BIN_BP as _BIN
        from manta_hic.nn.specs import build_specs, random_backgrounds

        if not isinstance(conditions, dict):
            conditions = {("wt" if m is None else f"cond{i}"): list(m or []) for i, m in enumerate(conditions)}
        else:
            conditions = {k: list(v or []) for k, v in conditions.items()}

        rng = np.random.default_rng(rng_seed)
        runs = min(int(runs), int(self.fetcher.N_runs))
        reverse = {False: False, True: True, "average": "both"}[rc]
        max_shift_bp = int(max_shift_bins) * _BIN
        kw = dict(runs=runs, rng=rng, reverse=reverse, max_shift_bp=max_shift_bp)

        if backgrounds == "matched":
            superset = {m for muts in conditions.values() for m in muts}
            bgs = random_backgrounds(chrom, start_bp, mutations_superset=superset, **kw)
            specs = build_specs(bgs, conditions, rng=rng, fasta=self.fetcher.fasta_open, channels=channels)
        elif backgrounds == "random":
            specs = []
            for name, muts in conditions.items():
                bgs = random_backgrounds(chrom, start_bp, mutations_superset=set(muts), **kw)
                specs += build_specs(bgs, {name: muts}, rng=rng, fasta=self.fetcher.fasta_open, channels=channels)
        else:
            raise ValueError(f"backgrounds must be 'matched' or 'random', got {backgrounds!r}")

        groups = self.infer_grouped(specs, ["mutation_set"], batch_size=batch_size)
        maps = {key["mutation_set"]: m for key, m, _ in groups}
        return maps, (groups[0][2] if groups else [])

    # -- observed target (needs a banded file) ------------------------------- #
    def is_eligible(self, chrom, start_bp, *, min_fraction=0.1, fold=None) -> bool:
        """O(1) check that the observed window is a clean training/eval target (requires a ``target`` file)."""
        if self.target_file is None:
            raise ValueError("no target file attached; pass target=… to check eligibility")
        return self.target_file.is_eligible(chrom, start_bp, self.n_bins, min_fraction=min_fraction, fold=fold)

    @torch.no_grad()
    def target(self, chrom, start_bp, *, observed_over_expected=True):
        """
        Observed Hi-C map for the same window from the banded ``target`` file, as ``[output_channels, n, n]``.

        With ``observed_over_expected`` (default) the map is divided by the per-arm distance expectation via
        :func:`create_expected_matrix` (matching the training target and the model's output convention);
        otherwise raw balanced counts are returned. Returns ``None`` if no target file is attached.
        """
        if self.target_file is None:
            return None
        hic, weight, exp = self.target_file.get_window(chrom, start_bp, self.n_bins)
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).float().unsqueeze(0).to(self.device)
        snippet, expmat = create_expected_matrix(t(hic), t(weight), t(exp))
        if not observed_over_expected:
            return snippet[0]
        ooe = torch.where(expmat > 0, snippet / expmat.clamp_min(1e-9), torch.zeros_like(snippet))
        return ooe[0]
