#!/bin/bash
# тестирование на всем hard
cd "$(dirname "$0")/.."
N=${1:-10000}
CSV=data_real/sudoku_hard_radcliffe.csv
OUT=results_final_new
PY=${PY:-python}
mkdir -p "$OUT"
run() { echo ">>> $1"; $PY infer.py "${@:2}" --csv "$CSV" --limit "$N" --out "$OUT/$1.json"; }

run stochastic                 --strategy stochastic
run adaptive                   --strategy adaptive --reveal_per_step 1
run searchdiff_beam8           --strategy searchdiff --beam_size 8
run guided                     --strategy guided --guidance_lr 0.5
run margin_ds81                --strategy margin --diffusion_steps 81
run remdm_324                  --strategy remdm --remdm_steps 324
run vbon_adaptive_n16          --strategy verifier-bon --base_strategy adaptive --n_samples 16
run vbon_searchdiff_beam8_n16  --strategy verifier-bon --base_strategy searchdiff --vbon_beam_size 8 --n_samples 16

echo "Done -> $OUT"
