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
# Per resolution, the banded slice is rsynced to $TMP_BANDED (network is slow), trained, then deleted;
# models + train.log go to
#   $MANTA_ROOT/trained_models/<genome>/folds_<folds>/<res>/
# A resolution whose output folder already exists is SKIPPED -- delete partials manually to retrain.
# Overridable env: MANTA_ROOT, BATCH_SIZE, TMP_BANDED (needs up to ~45 GB for hg38@256, ~15 GB mm10;
# put it on a DISK-backed path if /tmp is tmpfs), EPOCH_MULT.

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
# On the 'all' cache we deliberately overtrain (2x epochs): with every fold in the training set this
# basically "improves the resolution" of the predicted Hi-C. It should retain predictive capability --
# when a regular (held-out) model was overtrained 5x by mistake, its validation loss stabilized but
# did not decline.
if [ "$FOLDS" = all ]; then EPOCH_MULT=${EPOCH_MULT:-2}; else EPOCH_MULT=${EPOCH_MULT:-1}; fi
echo "[worker] genome=$GENOME gpu=$GPU folds=$FOLDS (val=$VAL test=$TEST epoch_mult=$EPOCH_MULT) cache=$CACHE"
command -v manta_hic >/dev/null || { echo "manta_hic not on PATH" >&2; exit 1; }
[ -d "$MANTA_ROOT/banded_inputs/$GENOME" ] || { echo "no $MANTA_ROOT/banded_inputs/$GENOME" >&2; exit 1; }

# -- one co-training run per resolution, fine-tuned order ---------------------------------------- #
# Only the current resolution's banded slice is copied to $TMP_BANDED (4-45 GB vs 150 GB for the whole
# genome) and it is deleted after a successful run; a failed run leaves it for the rerun (rsync skips
# already-copied files).
mkdir -p "$TMP_BANDED"
for RES in "${RES_ORDER[@]}"; do
    OUT=$MANTA_ROOT/trained_models/$GENOME/folds_$FOLDS/$RES
    SRC=("$MANTA_ROOT/banded_inputs/$GENOME/"*_"$RES".bhic.h5)
    if [ ! -e "${SRC[0]}" ]; then
        echo "[worker] $RES: no banded files for $GENOME, skipping"
        continue
    fi
    if [ -e "$OUT" ]; then
        echo "[worker] $RES: $OUT exists, skipping (delete it to retrain)"
        continue
    fi
    echo "[worker] $RES: copying ${#SRC[@]} banded files to $TMP_BANDED"
    rsync -a --info=progress2 "${SRC[@]}" "$TMP_BANDED/"
    FILES=("$TMP_BANDED"/*_"$RES".bhic.h5)
    mkdir -p "$OUT"
    ARGS=()
    for f in "${FILES[@]}"; do
        b=$(basename "$f")
        ARGS+=(-m "${b%_"$RES".bhic.h5}=$f")
    done
    echo "[worker] $RES: ${#FILES[@]} datasets -> $OUT"
    manta_hic train manta "${ARGS[@]}" -c "$CACHE" -o "$OUT" -g "$GENOME" \
        --batch-size="$BATCH_SIZE" --val-fold="$VAL" --test-fold="$TEST" \
        --epoch-multiplier="$EPOCH_MULT" -d "cuda:$GPU" \
        2>&1 | tee "$OUT/train.log"
    rm -f "${FILES[@]}"                     # reclaim /tmp; next resolution brings its own slice
done
echo "[worker] $GENOME ALL DONE"
