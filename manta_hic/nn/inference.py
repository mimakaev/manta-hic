"""
Turnkey Manta inference (``nn/inference.py``).

Ties a trained ``Manta2`` checkpoint to a :class:`CachedStochasticActivationFetcher` and hides the padding /
windowing / reverse-complement bookkeeping that callers used to redo by hand. One object predicts a Hi-C map
for a region, optionally averages the forward + reverse-complement passes, runs a wild-type-vs-mutant matched
pair, and (if you point it at the banded target file) hands back the observed map for the same window so you
can plot prediction and target side by side.

    infer = MantaInference("…/saved_model.pth", fetcher, resolution=2048, target="…/4dn-diff_2048.bhic.h5")
    pred = infer.predict("chr1", 20_000_000)              # [output_channels, n_bins, n_bins] torch tensor
    obs  = infer.target("chr1", 20_000_000)               # observed-over-expected, same window (or None)
    wt, mut = infer.predict_pair("chr1", 20_000_000, [("sub", 20_500_000, "ACGT...")])

Checkpoints are bare ``state_dict``s: ``output_channels`` is read from ``final_conv.weight`` and the tower
height from the resolution (``round(log2(resolution)) - 9``), so only the resolution needs to be supplied.
"""

import numpy as np
import torch

from manta_hic.io.banded import BandedHicFile
from manta_hic.nn.manta import Manta2
from manta_hic.ops.hic_ops import create_expected_matrix


def tower_height_for_resolution(resolution: int) -> int:
    """Manta's ``tower_height`` for a Hi-C bin size (microzoi 256 bp -> log2(256)=8, +1 for the first maxpool)."""
    return int(round(np.log2(resolution))) - 9


class MantaInference:
    """
    Predict Hi-C maps from a trained Manta model + a MicroZoi activation fetcher.

    Parameters
    ----------
    checkpoint : str | os.PathLike | dict
        Path to a ``saved_model.pth`` (a bare ``state_dict``) or an already-loaded state dict.
    fetcher : CachedStochasticActivationFetcher
        Supplies activations; must cover the queried chromosome/genome (and hold a fasta handle for mutations).
    resolution : int
        Hi-C bin size in bp of the trained model (also sets ``tower_height`` unless ``tower_height`` is given).
    device : str
        Device to run on (e.g. ``"cuda:0"``).
    n_bins, bins_pad : int
        Map side length and activation padding (must match training; the pretrained models use 1024 / 128).
    channel_names : list[str] | None
        Optional per-output-channel names. Defaults to the ``target`` file's shortnames when a target is given.
    target : BandedHicFile | str | None
        Optional banded file with the observed maps, enabling :meth:`target` / :meth:`is_eligible`.
    tower_height : int | None
        Override the tower height (else derived from ``resolution``).
    model_params : dict | None
        Extra ``Manta2`` kwargs if the checkpoint was trained with non-default architecture params.
    """

    def __init__(
        self,
        checkpoint,
        fetcher,
        *,
        resolution,
        device="cuda:0",
        n_bins=1024,
        bins_pad=128,
        channel_names=None,
        target=None,
        tower_height=None,
        model_params=None,
    ):
        state = (
            checkpoint
            if isinstance(checkpoint, dict)
            else torch.load(checkpoint, map_location=device, weights_only=True)
        )
        output_channels = int(state["final_conv.weight"].shape[0])
        th = tower_height if tower_height is not None else tower_height_for_resolution(resolution)
        self.model = (
            Manta2(
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

        self.fetcher = fetcher
        self.resolution = int(resolution)
        self.device = device
        self.n_bins = int(n_bins)
        self.bins_pad = int(bins_pad)
        self.output_channels = output_channels

        self.target_file = None
        if target is not None:
            self.target_file = BandedHicFile(target) if isinstance(target, (str, bytes)) else target
        self.channel_names = channel_names or (self.target_file.shortnames if self.target_file else None)

    # -- coordinate helper: model input window (padded) for a map at start_bp -- #
    def _fetch_span(self, start_bp):
        map_end = start_bp + self.n_bins * self.resolution
        return start_bp - self.bins_pad * self.resolution, map_end + self.bins_pad * self.resolution

    def _run(self, acts):
        """Run the model on a ``[in_channels, L]`` activation tensor -> ``[output_channels, n_bins, n_bins]``."""
        with torch.autocast(self.device.split(":")[0]):
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
    def predict_pair(self, chrom, start_bp, mutations, *, n_runs=1, run_idx=None):
        """
        Predict ``(wt_map, mut_map)`` for the same window, sharing one MicroZoi batch (for mutation screens).

        ``mutations`` is one list of length-preserving edits applied together (see
        :meth:`CachedStochasticActivationFetcher.fetch_matched_pair`). ``run_idx`` pins a specific cached run.
        """
        fs, fe = self._fetch_span(start_bp)
        wt_acts, mut_acts = self.fetcher.fetch_matched_pair(
            chrom, fs, fe, self.device, mutations, reverse=False, n_runs=n_runs, run_idx=run_idx
        )
        return self._run(wt_acts).float(), self._run(mut_acts).float()

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
