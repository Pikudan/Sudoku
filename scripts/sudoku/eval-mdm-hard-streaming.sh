#!/usr/bin/env bash
set -euo pipefail

checkpoint_dir=${1:?Usage: bash scripts/sudoku/eval-mdm-hard-streaming.sh CHECKPOINT_DIR [OUTPUT_DIR] [DECODING_STRATEGY]}
output_dir=${2:-"${checkpoint_dir}/sudoku_hard_margin_streaming"}
decoding_strategy=${3:-margin-linear}

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

python3 -u scripts/sudoku/eval-mdm-hard-streaming.py \
    --checkpoint_dir "$checkpoint_dir" \
    --output_dir "$output_dir" \
    --dataset sudoku_hard \
    --dataset_dir data \
    --model_name_or_path model_config_tiny \
    --cutoff_len 164 \
    --diffusion_steps 20 \
    --decoding_strategy "$decoding_strategy" \
    --topk_decoding \
    --per_device_eval_batch_size 1024 \
    --preprocessing_num_workers 8
