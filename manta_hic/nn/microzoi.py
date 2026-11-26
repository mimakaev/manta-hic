"""
A "micro Borzoi" model for predicting gene expression from DNA sequences.

Main differences from Borzoi are:
* Model operates at a single 256bp resolution
* There is no U-net
* Using modern transformer architecture with rotary embeddings and SDPA
"""

import h5py
import numpy as np
import polars as pl
import pysam
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from manta_hic.nn.layers import ConvolutionalBlock1d, TransformerTower
from manta_hic.nn.training_meta import get_seqs_targs
from manta_hic.ops.seq_ops import make_seq_1hot

# Constructor arguments that determine the weights or the inference-time math. A state dict alone does not pin
# all of them down: ``n_heads`` leaves every tensor shape unchanged and ``seq_length`` owns no tensor at all, so a
# wrong value loads strictly and silently gives wrong outputs. That is why checkpoints carry this config.
# Everything else (dropouts, gradient checkpointing, ``return_type``) only affects training or which output is
# returned.
MICROZOI_ARCH_KEYS = (
    "seq_conv_width",
    "tower_mults",
    "groups",
    "width",
    "base_channels",
    "n_heads",
    "ff_mult",
    "transf_layers",
    "seq_length",
    "last_channels_mult",
    "output_channels_human",
    "output_channels_mouse",
    "qk_norm",
)
MICROZOI_CHECKPOINT_FORMAT = "microzoi-checkpoint"
# Head layers, absent from a ``return_type="mha"`` model (and from the weights stored in activation caches).
_HEAD_PREFIXES = ("output_conv_block.", "output_conv_block2.", "final_conv_human.", "final_conv_mouse.")


def microzoi_version(config):
    """Architecture generation of a MicroZoi config: 1 = first generation (every model and cache in the paper),
    2 = with QK-norm (``configs/microzoi_v2.json``, experimental). Training-recipe changes that leave the
    architecture alone do not change the version."""
    return 2 if config.get("qk_norm", False) else 1


class Microzoi(nn.Module):
    def __init__(
        self,
        seq_conv_width=15,
        tower_mults=[8, 9, 10, 11, 12, 14, 16, 16],
        groups=[4, 1, 4, 1, 4, 1, 4],
        width=[3, 3, 3, 3, 3, 3, 3],
        base_channels=64,
        n_heads=16,
        ff_mult=4,
        transf_layers=12,
        seq_length=2**19 + 2**18,
        last_channels_mult=32,
        output_channels_human=7611,
        output_channels_mouse=2608,
        attn_dropout=0.4,
        ff_dropout=None,
        conv_dropout=0.15,
        qk_norm=False,
        num_gn_checkpoints=0,
        checkpoint_first=True,
        return_type="default",
    ):
        super().__init__()
        # the full architecture (defaults filled in), written into every checkpoint by microzoi_checkpoint()
        args = locals()
        self.arch_config = {
            k: list(args[k]) if isinstance(args[k], (list, tuple)) else args[k] for k in MICROZOI_ARCH_KEYS
        }
        self.version = microzoi_version(self.arch_config)
        # Hard-coding those as Borzoi training data never changes
        self.nbins = seq_length // 256
        self.crop = (seq_length // 32 - 6144) // 8 // 2

        self.channels_1d = [base_channels * i for i in tower_mults]
        ch_1d = self.channels_1d
        working_channels = self.working_channels = ch_1d[-1]
        self.last_chanels = base_channels * last_channels_mult
        self.nheads = n_heads
        self.output_channels_human = output_channels_human
        self.output_channels_mouse = output_channels_mouse
        self.return_type = return_type
        self.checkpoint_first = checkpoint_first

        # utility layers
        self.dropout = nn.Dropout1d(p=conv_dropout)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Convolutional layers
        self.conv0 = nn.Conv1d(4, ch_1d[0], kernel_size=seq_conv_width, padding=seq_conv_width // 2)

        self.conv_blocks = nn.ModuleList()

        for ind, (c_st, c_end, gr, W) in enumerate(zip(self.channels_1d[:-1], self.channels_1d[1:], groups, width)):
            do_checkpoint = ind < num_gn_checkpoints
            self.conv_blocks.append(ConvolutionalBlock1d(c_st, c_end, W=W, groups=gr, checkpoint_gn=do_checkpoint))

        self.mha_tower = TransformerTower(
            n_layers=transf_layers,
            d_model=working_channels,
            n_bins=self.nbins,
            n_heads=n_heads,
            attn_drop_p=attn_dropout,
            ff_drop_p=attn_dropout if ff_dropout is None else ff_dropout,  # None: the old single rate on both
            ff_mult=ff_mult,
            qk_norm=qk_norm,
        )
        if self.return_type in ["default"]:
            self.output_conv_block = ConvolutionalBlock1d(working_channels, self.last_chanels, W=1, D=1, groups=1)
            self.output_conv_block2 = ConvolutionalBlock1d(self.last_chanels, self.last_chanels, W=3, D=1, groups=8)

            self.final_conv_human = nn.Conv1d(self.last_chanels, self.output_channels_human, kernel_size=1)
            self.final_conv_mouse = nn.Conv1d(self.last_chanels, self.output_channels_mouse, kernel_size=1)

    def _first(self, x):
        return self.maxpool(self.conv0(x))

    def trunk(self, x):
        """One-hot sequence [B, 4, L] -> transformer output [B, working_channels, L // 256]."""
        # The first convolution has the largest activation in the model; checkpointing it costs ~2% compute.
        # It MUST be non-reentrant: the one-hot input carries no gradient, and a reentrant checkpoint with no
        # grad-carrying input is skipped in the backward pass -- conv0 then never trains (it did not, in the
        # first generation of MicroZoi checkpoints).
        if self.checkpoint_first and self.training:
            x = checkpoint(self._first, x, use_reentrant=False)
        else:
            x = self._first(x)

        for conv_block in self.conv_blocks:
            x = self.maxpool(conv_block(x))

        return self.mha_tower(x)

    def compile_trunk(self, **kwargs):
        """``torch.compile`` the sequence trunk (conv tower + transformer) in place. The heads stay eager: their
        integer crop offset changes every step and would force recompiles. The state dict is unchanged."""
        self.trunk = torch.compile(self.trunk, **kwargs)
        return self

    def forward(self, x, genome="hg38", offset=0, crop_mha=0, crop_result=True):
        x = self.trunk(x)

        if self.return_type == "mha":
            return x[:, :, crop_mha : x.shape[2] - crop_mha]  # crop on device to make it faster

        # Crop after transformers - the rest is just local convolutions
        if crop_result:
            crop_st, crop_end = self.crop + offset, x.shape[2] - self.crop + offset
            x = x[:, :, crop_st:crop_end]

        x = self.output_conv_block(x)  # groups=1, w=1
        x = self.dropout(x)
        x = self.output_conv_block2(x)  # groups=8, w=3
        x = self.dropout(x)
        x = F.gelu(x)

        if genome == "hg38":
            x = self.final_conv_human(x)
        elif genome == "mm10":
            x = self.final_conv_mouse(x)
        else:
            raise ValueError("Target can be hg38 or mm10")
        x = F.softplus(x)
        return x


def microzoi_checkpoint(model, state_dict=None):
    """A self-describing MicroZoi checkpoint: ``{"format", "version", "config", "state_dict"}``, where ``config``
    is the full architecture (:data:`MICROZOI_ARCH_KEYS`) and ``version`` its generation. ``state_dict`` defaults
    to the model's own (pass e.g. an EMA's to save that instead). Read it back with :func:`load_microzoi`."""
    return {
        "format": MICROZOI_CHECKPOINT_FORMAT,
        "version": model.version,
        "config": dict(model.arch_config),
        "state_dict": model.state_dict() if state_dict is None else state_dict,
    }


def save_microzoi(model, path, state_dict=None):
    """Save :func:`microzoi_checkpoint` to ``path`` (a filename or a binary buffer)."""
    torch.save(microzoi_checkpoint(model, state_dict), path)


def _same(a, b):
    norm = lambda v: list(v) if isinstance(v, (list, tuple)) else v
    return norm(a) == norm(b)


def resolve_microzoi_checkpoint(obj, params=None, return_type="default"):
    """Check a loaded MicroZoi checkpoint against ``params`` and return ``(constructor_kwargs, state_dict)``
    ready for a strict ``load_state_dict``.

    ``obj`` is either a self-describing checkpoint (:func:`microzoi_checkpoint`) or a raw state dict (all
    first-generation checkpoints and the weights in caches written before checkpoints carried their config).
    ``params`` is the ``"model"`` section of a params json, or None.

    - Self-describing: the stored config is authoritative. Any architecture key in ``params`` that disagrees
      with it is an error; non-architecture keys (dropouts etc.) are kept from ``params``.
    - Raw state dict: the architecture comes from ``params`` (library defaults where missing). Whether the
      weights have QK-norm is read from the keys, and must agree with ``params``.

    Obsolete ``freqs_cis`` buffers are dropped, and the head weights too when ``return_type="mha"``.
    """
    params = dict(params or {})
    if not isinstance(obj, dict):
        raise TypeError(f"expected a MicroZoi checkpoint dict, got {type(obj).__name__}")
    if "format" in obj or "state_dict" in obj:
        if obj.get("format") != MICROZOI_CHECKPOINT_FORMAT:
            raise ValueError(f"not a MicroZoi checkpoint (format={obj.get('format')!r})")
        config, sd = dict(obj["config"]), obj["state_dict"]
        if missing := [k for k in MICROZOI_ARCH_KEYS if k not in config]:
            raise ValueError(f"MicroZoi checkpoint config lacks architecture keys {missing}")
        if obj["version"] != microzoi_version(config):
            raise ValueError(
                f"MicroZoi checkpoint says version {obj['version']} but its config is version "
                f"{microzoi_version(config)}"
            )
        bad = {k: (params[k], config[k]) for k in MICROZOI_ARCH_KEYS if k in params and not _same(params[k], config[k])}
        if bad:
            raise ValueError(
                "params disagree with the MicroZoi checkpoint's own architecture (params, checkpoint): "
                + ", ".join(f"{k}={a!r} vs {b!r}" for k, (a, b) in bad.items())
            )
        kwargs = params | config
    else:
        sd = obj
        has_qk = any(k.endswith((".q_norm.weight", ".k_norm.weight")) for k in sd)
        if has_qk != bool(params.get("qk_norm", False)):
            raise ValueError(
                f"MicroZoi weights are version {2 if has_qk else 1} ({'with' if has_qk else 'without'} QK-norm) "
                f"but params say qk_norm={params.get('qk_norm', False)}; pass the params the model was trained with"
            )
        kwargs = params
    sd = {k: v for k, v in sd.items() if not k.endswith("freqs_cis")}
    if return_type == "mha":
        sd = {k: v for k, v in sd.items() if not k.startswith(_HEAD_PREFIXES)}
    return kwargs | {"return_type": return_type}, sd


def load_microzoi(source, params=None, *, device="cpu", return_type="default"):
    """Load a MicroZoi from ``source`` (a path, a binary buffer, or an already-loaded checkpoint dict), checked
    against ``params`` by :func:`resolve_microzoi_checkpoint` and loaded strictly. Returns the model in eval
    mode on ``device``."""
    obj = source if isinstance(source, dict) else torch.load(source, map_location=device, weights_only=True)
    kwargs, sd = resolve_microzoi_checkpoint(obj, params, return_type)
    model = Microzoi(**kwargs).to(device)
    model.load_state_dict(sd, strict=True)
    return model.eval()


def borzoi_loss(output, target, total_weight=0.2):
    "Multinomial-Poisson loss like in Borzoi"
    epsilon = 1e-7

    seq_len = output.shape[2]

    s_targ = target.sum(dim=2) + 1e-4  # (batch_size, num_targets)
    s_pred = output.sum(dim=2) + 1e-4

    targ = target + epsilon  # (batch_size, num_targets, seq_len)
    pred = output + epsilon

    poisson_loss = (s_pred - s_targ * torch.log(s_pred + epsilon)).mean() / seq_len  # sum(L) -> mean

    p_pred = pred / s_pred.unsqueeze(2)  # probability of prediction
    multinomial_loss = (-targ * torch.log(p_pred)).mean()  # mean(batch, track)

    return multinomial_loss + total_weight * poisson_loss


def corr(target, output, return_individual=False):
    "Fast correlation between target and output"
    t_mean, p_mean = target - target.mean(dim=-1, keepdim=True), output - output.mean(dim=-1, keepdim=True)
    cr = (t_mean * p_mean).sum(dim=-1) / (1e-5 + (t_mean.pow(2).sum(dim=-1) * p_mean.pow(2).sum(dim=-1)).sqrt())

    cr_mean = cr.mean().item()
    if return_individual:
        cr = cr.detach().cpu().numpy()
        return cr_mean, cr
    return cr_mean


def sequence_table(*, base, genomes, seq_length, batch_size=1, val_fold=3, test_fold=4, mode="val"):
    """The training/validation sequences of ``mode`` for all ``genomes``, each genome truncated to a multiple of
    ``batch_size`` (a batch never mixes genomes). Unshuffled."""
    dfs = []
    omit_folds = [int(val_fold), int(test_fold)]
    for gen in genomes:
        seqdf, _ = get_seqs_targs(gen)
        # add index and genome, convert fold to int, add seq start and end
        seqdf = seqdf.with_columns(
            pl.int_range(pl.len()).over("fold").alias("index"),
            pl.col("fold").str.replace("fold", "").cast(int),
            ((pl.col("start") + pl.col("end")) // 2 - seq_length // 2).alias("seq_start"),
            ((pl.col("start") + pl.col("end")) // 2 + seq_length // 2).alias("seq_end"),
        )
        # add file path per fold and keys
        seqdf = seqdf.with_columns(
            (pl.lit(base + "/" + gen + "/fold") + pl.col("fold").cast(str) + pl.lit(".h5")).alias("file"),
            (pl.lit("sample") + pl.col("index").cast(str)).alias("key"),
        )

        if mode == "train":
            seqdf = seqdf.filter(~pl.col("fold").cast(int).is_in(omit_folds))
        elif mode == "val":
            seqdf = seqdf.filter(pl.col("fold").cast(int) == int(val_fold))
        elif mode == "test":
            seqdf = seqdf.filter(pl.col("fold").cast(int) == int(test_fold))
        elif mode != "all":
            raise ValueError("Mode should be train, val, test, or all")
        dfs.append(seqdf[: len(seqdf) // batch_size * batch_size])  # truncate to batch size
    return pl.concat(dfs)


def dataGenerator(
    *,
    base,
    genomes,
    seq_length,
    fasta_path,
    resolution,  # end storage parameters
    batch_size=1,  # start training/validation parameters
    val_fold=3,
    test_fold=4,
    max_shift=0,
    random_shift_bp=27,  # +- 27bp out of 256
    mode="val",
):
    """Ranks and world size aware data generator that would alternate hg38 and mm10 blocks"""

    fastas = {name: pysam.FastaFile(fasta_path.format(name)) for name in genomes}
    seqdf = sequence_table(
        base=base,
        genomes=genomes,
        seq_length=seq_length,
        batch_size=batch_size,
        val_fold=val_fold,
        test_fold=test_fold,
        mode=mode,
    )
    # shuffle in blocks of batch_size - lol polars is good!
    seqdf = seqdf.sort((pl.int_range(pl.len()) // batch_size).hash(np.random.randint(0, 1e9)))

    for batch in seqdf.iter_slices(batch_size):
        if mode == "train":
            shift_bins = np.random.randint(-max_shift, max_shift + 1)
            extra_shift_bp = np.random.randint(-random_shift_bp, random_shift_bp + 1)
            shift_bp = shift_bins * resolution + extra_shift_bp
        else:
            shift_bins = 0
            shift_bp = 0

        global_meta = {"shift_bins": shift_bins, "shift_bp": shift_bp, "genome": batch["genome"][0]}
        batch_arrs = [[], []]
        metadatas = []
        for row in batch.iter_rows(named=True):
            with h5py.File(row["file"], "r") as myfile:
                d = myfile[row["key"] + "/data"][:]
                batch_arrs[1].append(d)

                fa = fastas[row["genome"]]
                seq_hot = make_seq_1hot(fa, row["chrom"], row["seq_start"] - shift_bp, row["seq_end"] - shift_bp)
                batch_arrs[0].append(seq_hot)

                metadatas.append(row)
        yield (batch_arrs, global_meta, metadatas)
