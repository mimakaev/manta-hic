#!/usr/bin/env bash
# Fan-out worker for the 2026 Manta training campaign: one process = one genome on one GPU.
#
#   ./train_worker.sh hg38 0      # run from the directory that holds this machine's caches
#   ./train_worker.sh mm10 1
#
# Assumes manta_hic is installed (the tree with epoch normalization, final_channels auto-expand and
# --val-fold -1) and this machine's cache microzoi_cache_<genome>_<folds>.h5 sits in the current
# directory. <folds> (12/34/56/70/all) picks the split: XY -> --val-fold X --test-fold Y;
# 'all' -> no holdout (train on every fold, no validation).
#
# Banded inputs are rsynced once to $TMP_BANDED (network is slow); models + train.log go to
#   $MANTA_ROOT/trained_models/<genome>/folds_<folds>/<res>/
# A resolution whose output folder already exists is SKIPPED -- delete partials manually to retrain.
# Overridable env: MANTA_ROOT, BATCH_SIZE, TMP_BANDED (put it on a DISK-backed path: hg38 is ~150 GB,
# a tmpfs /tmp would eat that much RAM).

set -euo pipefail

GENOME=${1:?usage: train_worker.sh <hg38|mm10> <gpu-id>}
GPU=${2:?usage: train_worker.sh <hg38|mm10> <gpu-id>}
MANTA_ROOT=${MANTA_ROOT:-/net/levsha/scratch2/max/2026_manta}
BATCH_SIZE=${BATCH_SIZE:-4}                 # 8 OOMs on 24 GB at coarse resolutions
TMP_BANDED=${TMP_BANDED:-/tmp/manta_banded/$GENOME}
RES_ORDER=(2048 1024 4096 512 8192 256 16384)
export BLOSC_NTHREADS=${BLOSC_NTHREADS:-4}  # multi-threaded band-read decompression

# -- this machine's cache -> fold pair ----------------------------------------------------------- #
CACHES=(microzoi_cache_"$GENOME"_*.h5)
if [ ${#CACHES[@]} -ne 1 ] || [ ! -e "${CACHES[0]}" ]; then
    echo "need exactly one microzoi_cache_${GENOME}_*.h5 in $(pwd), found: ${CACHES[*]}" >&2
    exit 1
fi
CACHE=$(readlink -f "${CACHES[0]}")
FOLDS=$(basename "$CACHE" .h5)
FOLDS=${FOLDS##*_}
if [ "$FOLDS" = all ]; then VAL=-1 TEST=-1; else VAL=${FOLDS:0:1} TEST=${FOLDS:1:1}; fi
echo "[worker] genome=$GENOME gpu=$GPU folds=$FOLDS (val=$VAL test=$TEST) cache=$CACHE"
command -v manta_hic >/dev/null || { echo "manta_hic not on PATH" >&2; exit 1; }
[ -d "$MANTA_ROOT/banded_inputs/$GENOME" ] || { echo "no $MANTA_ROOT/banded_inputs/$GENOME" >&2; exit 1; }

# -- local copy of the banded inputs (idempotent) ------------------------------------------------ #
mkdir -p "$TMP_BANDED"
rsync -a --info=progress2 "$MANTA_ROOT/banded_inputs/$GENOME/" "$TMP_BANDED/"

# -- one co-training run per resolution, fine-tuned order ---------------------------------------- #
for RES in "${RES_ORDER[@]}"; do
    OUT=$MANTA_ROOT/trained_models/$GENOME/folds_$FOLDS/$RES
    FILES=("$TMP_BANDED"/*_"$RES".bhic.h5)
    if [ ! -e "${FILES[0]}" ]; then
        echo "[worker] $RES: no banded files for $GENOME, skipping"
        continue
    fi
    if [ -e "$OUT" ]; then
        echo "[worker] $RES: $OUT exists, skipping (delete it to retrain)"
        continue
    fi
    mkdir -p "$OUT"
    ARGS=()
    for f in "${FILES[@]}"; do
        b=$(basename "$f")
        ARGS+=(-m "${b%_"$RES".bhic.h5}=$f")
    done
    echo "[worker] $RES: ${#FILES[@]} datasets -> $OUT"
    manta_hic train manta "${ARGS[@]}" -c "$CACHE" -o "$OUT" -g "$GENOME" \
        --batch-size="$BATCH_SIZE" --val-fold="$VAL" --test-fold="$TEST" -d "cuda:$GPU" \
        2>&1 | tee "$OUT/train.log"
done
echo "[worker] $GENOME ALL DONE"
