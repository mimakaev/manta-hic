#!/usr/bin/env bash

INPUT_DIR="/home/magus/data/hic/2024_manta_saved_inputs/hg38"
OUTPUT_PARENT="/net/levsha/scratch2/max/2024_manta_trained_models"
# get device from the first argument or default to cuda:0
DEVICE=${1:-cuda:0}

for h5file in "$INPUT_DIR"/*.h5; do
    base="$(basename "$h5file" .h5)"
    outdir="${OUTPUT_PARENT}/${base}_folds34"

    if [[ -d "$outdir" ]]; then
        echo "Skipping $base since $outdir already exists."
        continue
    fi

    manta_hic train manta \
        -i "$h5file" \
        -c ../data_ssd/manta-hg38-cache.h5 \
        -d "$DEVICE" \
        -o "$outdir" \
        -f ../data_ssd/hg38.fa \
	--work-dir /tmp \
        --batch-size 3
done

