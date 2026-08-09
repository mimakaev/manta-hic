#!/bin/bash
# Small-vs-medium architecture comparison at 4096bp on 3 hg38 datasets (<=4 channels), co-trained.
# Datasets: krietenstein (2ch microC), hansen_microc_combined (3ch microC), combined-everything (1ch intact-HiC).
# cache hg38_70 <-> val-fold 7 / test-fold 0 (the trained-models convention). Compute-bound on the contended
# GPUs (~85% util), prefetch overlaps cache IO. Two runs, one per GPU, in parallel.
set -uo pipefail
source /workspace/.venv/bin/activate
IN=/mnt/ro/max/2026-manta-banded-inputs
EV=/mnt/rw/2024_intact_hic/banded_v2/combined-everything_4096.bhic.h5
CACHE=/workspace/data_ssd/microzoi_cache_hg38_70.h5
OUT=/workspace/arch_compare
EPOCHS=${EPOCHS:-20}
NBINS=${NBINS:-512}

run() {  # $1=preset $2=device
  manta_hic train manta \
    -m krietenstein=$IN/krietenstein_4096.bhic.h5 \
    -m hansen=$IN/hansen_microc_combined_4096.bhic.h5 \
    -m combined-everything=$EV \
    -c $CACHE -g hg38 --preset "$1" --n-bins "$NBINS" \
    --val-fold 7 --test-fold 0 --device "$2" \
    --n-epochs "$EPOCHS" --batch-size 8 \
    -o "$OUT/$1"
}
run "$1" "$2"
