#!/usr/bin/env python
"""
Repack Manta ``saved_model.pth`` files into the **self-describing** format
(``{"state_dict": ..., "config": {...}}`` -- what ``manta_hic.nn.manta.save_manta_checkpoint`` writes), sourcing
``genome`` / ``channel_names`` / resolution from the matching **banded input** (authoritative, by construction).

Old trained checkpoints are bare ``state_dict``s with no metadata; ``MantaInference`` now needs the config
(resolution is otherwise ambiguous at <=1024 bp, and the model is genome-specific). For each model directory
``<dataset>_<res>_<fold>/`` this looks up ``<inputs-dir>/<dataset>_<res>.bhic.h5`` and writes a config with
``resolution / n_bins / bins_pad / tower_height / output_channels / genome / channel_names``.

Usage
-----
    python scripts/repack_manta_checkpoints.py <models-root> --inputs-dir <banded-inputs> [--in-place] [--no-backup]

    # a whole tree of "<dataset>_<res>_<fold>/saved_model.pth":
    python scripts/repack_manta_checkpoints.py /net/.../2024_manta_trained_models \
        --inputs-dir /net/.../2026-manta-banded-inputs --in-place

Output location: default writes ``saved_model.packed.pth`` next to each input; ``--in-place`` overwrites the
original (first copying it to ``saved_model.pth.bak`` unless ``--no-backup``). Writes are atomic (temp file +
rename). Already-packed files get ``genome`` / ``channel_names`` backfilled from the input if missing, else are
skipped. A model whose output channel count disagrees with its banded input is skipped (dataset drift -- investigate).
"""

import argparse
import re
import shutil
from pathlib import Path

import torch

from manta_hic.io.banded import BandedHicFile

DIR_RE = re.compile(r"^(?P<dataset>.+)_(?P<res>\d+)_(?P<fold>all|folds?\d+)$")


def _input_meta(inputs_dir: Path, dataset: str, res: int):
    """genome / shortnames / n_channels / resolution from ``<inputs_dir>/<dataset>_<res>.bhic.h5``."""
    path = inputs_dir / f"{dataset}_{res}.bhic.h5"
    if not path.exists():
        raise FileNotFoundError(path)
    with BandedHicFile(path) as bf:
        return {
            "genome": bf.genome,
            "channel_names": list(bf.shortnames),
            "n_channels": int(bf.n_channels),
            "resolution": int(bf.resolution),
        }


def repack(src: Path, dst: Path, meta: dict, *, backup: bool) -> str:
    obj = torch.load(src, map_location="cpu", weights_only=True)
    packed = isinstance(obj, dict) and "state_dict" in obj and "config" in obj
    state = obj["state_dict"] if packed else obj
    output_channels = int(state["final_conv.weight"].shape[0])
    if output_channels != meta["n_channels"]:
        return f"SKIP {src}: model has {output_channels} channels but input has {meta['n_channels']} (dataset drift?)"

    resolution = meta["resolution"]
    if resolution & (resolution - 1):
        return f"SKIP {src}: input resolution {resolution} is not a power of two"

    if packed:  # already self-describing -- just backfill the fields the input owns, keep the rest
        config = dict(obj["config"])
        if "genome" in config and "channel_names" in config:
            return f"skip (already complete): {src}"
        config.setdefault("genome", meta["genome"])
        config.setdefault("channel_names", meta["channel_names"])
    else:
        config = {
            "resolution": resolution,
            "n_bins": 1024,
            "bins_pad": 128,
            "tower_height": resolution.bit_length() - 10,  # resolution == 2 ** (tower_height + 9)
            "output_channels": output_channels,
            "genome": meta["genome"],
            "channel_names": meta["channel_names"],
        }

    if backup and dst.exists():
        bak = dst.with_name(dst.name + ".bak")
        if not bak.exists():
            shutil.copy2(dst, bak)
    tmp = dst.with_name(dst.name + ".tmp")
    torch.save({"state_dict": state, "config": config}, tmp)
    tmp.replace(dst)
    return f"packed {src} -> genome={config['genome']}, res={config['resolution']}, channels={output_channels}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Repack Manta checkpoints into the self-describing format.")
    ap.add_argument("path", type=Path, help="a saved_model.pth, or a directory tree of <dataset>_<res>_<fold>/")
    ap.add_argument("--inputs-dir", type=Path, required=True, help="banded inputs dir (<dataset>_<res>.bhic.h5)")
    ap.add_argument("--in-place", action="store_true", help="overwrite the input instead of writing *.packed.pth")
    ap.add_argument("--no-backup", action="store_true", help="with --in-place, do NOT keep a *.bak of the original")
    args = ap.parse_args()

    files = sorted(args.path.rglob("saved_model.pth")) if args.path.is_dir() else [args.path]
    if not files:
        raise SystemExit(f"no saved_model.pth found under {args.path}")

    for src in files:
        m = DIR_RE.match(src.parent.name)
        if not m:
            print(f"  skip (dir name not <dataset>_<res>_<fold>): {src}")
            continue
        try:
            meta = _input_meta(args.inputs_dir, m["dataset"], int(m["res"]))
        except FileNotFoundError as e:
            print(f"  skip (no matching banded input {e}): {src}")
            continue
        dst = src if args.in_place else src.with_suffix(".packed.pth")
        print("  " + repack(src, dst, meta, backup=args.in_place and not args.no_backup))


if __name__ == "__main__":
    main()
