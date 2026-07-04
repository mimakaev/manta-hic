#!/usr/bin/env python
"""
Repack bare Manta ``saved_model.pth`` state dicts into the **self-describing** format that
``manta_hic.nn.manta.save_manta_checkpoint`` writes -- ``{"state_dict": ..., "config": {...}}`` with the
resolution baked in, so ``MantaInference`` no longer has to guess it from tensor shapes (which is ambiguous
below 1024 bp; see nn/inference.py). A no-op on already-packed files.

Usage
-----
    # one checkpoint, resolution given (or inferred from a "<name>_<res>_<fold>/" parent dir):
    python scripts/repack_manta_checkpoints.py MODEL.pth --resolution 2048
    python scripts/repack_manta_checkpoints.py 4dn-diff_2048_all/saved_model.pth   # 2048 from the dir name

    # a whole tree of "<dataset>_<res>_<fold>/saved_model.pth" (resolution from each parent dir):
    python scripts/repack_manta_checkpoints.py /path/to/2024_manta_trained_models/

    # source is read-only (e.g. an NFS mount)? mirror the packed tree to a writable location:
    python scripts/repack_manta_checkpoints.py /mnt/ro/.../2024_manta_trained_models/ --out-dir ~/repacked/

    # attach channel names from the matching banded target (single file):
    python scripts/repack_manta_checkpoints.py MODEL.pth --resolution 2048 --channel-names-from target.bhic.h5

Output location (pick one):
  * default        -- write ``<name>.packed.pth`` next to each input (needs a writable source dir).
  * ``--in-place`` -- overwrite the original in place (needs a writable source dir); the untouched original is
                      first copied to ``saved_model.pth.bak`` (disable with ``--no-backup``).
  * ``--out-dir D``-- write under D, mirroring the input's relative tree and keeping the filename
                      (``<D>/<dataset>_<res>_<fold>/saved_model.pth``). The only option that works when the
                      source is read-only. Never touches the source.
Only files literally named ``saved_model.pth`` are picked up in directory mode; intermediate
``model_<epoch>.pth`` checkpoints are ignored (pass them individually if you want them). Already-packed files
are skipped, so re-running is a safe no-op.
"""

import argparse
import re
import shutil
from pathlib import Path

import torch


def _resolution_from_dirname(path: Path) -> int | None:
    """Extract the resolution from a ``<dataset>_<res>_<fold>`` directory name (e.g. ``4dn-diff_2048_all``)."""
    m = re.search(r"_(\d+)_(all|folds?\d+)$", path.parent.name)
    return int(m.group(1)) if m else None


def repack(
    src: Path, dst: Path, resolution: int, *, n_bins: int, bins_pad: int, channel_names=None, backup=False
) -> None:
    obj = torch.load(src, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj and "config" in obj:
        print(f"  skip (already packed): {src}")
        return
    if resolution & (resolution - 1):
        raise ValueError(f"resolution {resolution} is not a power of two")
    state = obj
    config = {
        "resolution": int(resolution),
        "n_bins": int(n_bins),
        "bins_pad": int(bins_pad),
        "tower_height": resolution.bit_length() - 10,  # resolution == 2 ** (tower_height + 9)
        "output_channels": int(state["final_conv.weight"].shape[0]),
    }
    if channel_names is not None:
        config["channel_names"] = list(channel_names)
    # keep the untouched original alongside the packed file (only meaningful when overwriting in place).
    if backup and dst.exists():
        bak = dst.with_name(dst.name + ".bak")
        if not bak.exists():  # never clobber an existing backup (a prior run's pristine original)
            shutil.copy2(dst, bak)
    # atomic write: save to a sibling temp file, then rename over dst -- an interruption leaves either the
    # original or the finished file, never a truncated one (matters for --in-place over 497 models).
    tmp = dst.with_name(dst.name + ".tmp")
    torch.save({"state_dict": state, "config": config}, tmp)
    tmp.replace(dst)
    print(f"  packed {src} -> {dst}  (resolution={config['resolution']}, channels={config['output_channels']})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Repack bare Manta checkpoints into the self-describing format.")
    ap.add_argument("path", type=Path, help="a saved_model.pth, or a directory tree of them")
    ap.add_argument("--resolution", type=int, default=None, help="resolution in bp (else inferred from the dir name)")
    ap.add_argument("--channel-names-from", type=Path, default=None, help="banded .bhic.h5 to read channel names from")
    ap.add_argument("--n-bins", type=int, default=1024)
    ap.add_argument("--bins-pad", type=int, default=128)
    ap.add_argument("--in-place", action="store_true", help="overwrite the input instead of writing *.packed.pth")
    ap.add_argument("--out-dir", type=Path, default=None, help="mirror packed files here (works on read-only sources)")
    ap.add_argument("--no-backup", action="store_true", help="with --in-place, do NOT keep a *.bak of the original")
    args = ap.parse_args()

    if args.in_place and args.out_dir is not None:
        raise SystemExit("--in-place and --out-dir are mutually exclusive")

    names = None
    if args.channel_names_from is not None:
        from manta_hic.io.banded import BandedHicFile

        with BandedHicFile(args.channel_names_from) as bf:
            names = list(bf.shortnames)

    is_dir = args.path.is_dir()
    files = sorted(args.path.rglob("saved_model.pth")) if is_dir else [args.path]
    if not files:
        raise SystemExit(f"no saved_model.pth found under {args.path}")
    for src in files:
        res = args.resolution or _resolution_from_dirname(src)
        if res is None:
            print(f"  skip (no resolution; pass --resolution or use a <name>_<res>_<fold>/ dir): {src}")
            continue
        if args.out_dir is not None:
            rel = src.relative_to(args.path) if is_dir else Path(src.name)
            dst = args.out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
        elif args.in_place:
            dst = src
        else:
            dst = src.with_suffix(".packed.pth")
        backup = args.in_place and not args.no_backup
        repack(src, dst, res, n_bins=args.n_bins, bins_pad=args.bins_pad, channel_names=names, backup=backup)


if __name__ == "__main__":
    main()
