#!/usr/bin/env bash
#
# Train (or re-train) Manta heads from banded Hi-C inputs + a MicroZoi activation cache.
#
# Each job is "<input-basename> <genome> <fold>", where <fold> is either:
#   all   -> train on all data (no val/test split), output dir "<base>_all"
#   NM    -> two digits: val=foldN, test=foldM, output dir "<base>_foldsNM"  (e.g. 70 -> val fold7, test fold0)
# The input is <INPUT_DIR>/<base>.bhic.h5, the cache <CACHE_DIR>/microzoi_cache_<genome>_<foldcache>.h5
# (foldcache = "all" or the two digits), and the model is written to <OUTPUT_PARENT>/<outdir>.
# Resolution and channel count come from the input file; the checkpoint is self-describing.
#
# Usage:  ./train_manta.sh [device]           # device defaults to cuda:0
# Override paths via env, e.g.:  MANTA_CACHE_DIR=/ssd/caches ./train_manta.sh cuda:1

set -euo pipefail

# ---- paths (edit or override via env) ------------------------------------------------------------
INPUT_DIR="${MANTA_INPUT_DIR:-/net/levsha/scratch2/max/2026-manta-banded-inputs}"       # banded .bhic.h5 inputs
OUTPUT_PARENT="${MANTA_OUTPUT_DIR:-/net/levsha/scratch2/max/2024_manta_trained_models}"  # trained models go here
CACHE_DIR="${MANTA_CACHE_DIR:-/workspace/data_ssd}"    # MicroZoi caches -- MUST be on local SSD for speed
FASTA_DIR="${MANTA_FASTA_DIR:-/workspace/data_ssd}"    # <genome>.fa lives here
WORK_DIR="${MANTA_WORK_DIR:-/tmp}"                     # fast-local scratch the input is copied to during training
BATCH_SIZE="${MANTA_BATCH_SIZE:-2}"

DEVICE="${1:-cuda:0}"

# ---- jobs: the 6 folders that need (re)training --------------------------------------------------
JOBS=(
    "4dn-diff_8192          hg38  all"
    "bonev-merged_1024      mm10  70"
    "bonev-merged_2048      mm10  70"
    "bonev-merged_16384     mm10  70"
    "intact-celllines_4096  hg38  70"
    "intact-celllines_8192  hg38  70"
)

for job in "${JOBS[@]}"; do
    read -r base genome fold <<< "$job"

    input="${INPUT_DIR}/${base}.bhic.h5"
    fasta="${FASTA_DIR}/${genome}.fa"

    if [[ "$fold" == "all" ]]; then
        foldcache="all"
        outdir="${OUTPUT_PARENT}/${base}_all"
        fold_args=(--use-all-data --epoch-multiplier 2)   # all-data runs get 2x epochs (no val to early-stop on)
    else
        foldcache="$fold"
        outdir="${OUTPUT_PARENT}/${base}_folds${fold}"
        fold_args=(--val-fold "fold${fold:0:1}" --test-fold "fold${fold:1:1}")
    fi
    cache="${CACHE_DIR}/microzoi_cache_${genome}_${foldcache}.h5"

    echo "=== ${base}  (${genome}, fold=${fold}) -> ${outdir}"
    for f in "$input" "$cache" "$fasta"; do
        [[ -e "$f" ]] || { echo "  MISSING: $f -- skipping"; continue 2; }
    done

    manta_hic train manta \
        -i "$input" \
        -c "$cache" \
        -f "$fasta" \
        -o "$outdir" \
        -g "$genome" \
        -d "$DEVICE" \
        --work-dir "$WORK_DIR" \
        --batch-size "$BATCH_SIZE" \
        --overwrite \
        "${fold_args[@]}"
done
