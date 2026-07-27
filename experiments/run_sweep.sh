#!/bin/bash
# прогон по сетке на части примеров из hard
cd "$(dirname "$0")/.."
N=${1:-2000}
CSV=data_real/sudoku_hard_radcliffe.csv
OUT=results_sweep_new
PY=${PY:-python}
mkdir -p "$OUT"
run() { echo ">>> $1"; $PY infer.py "${@:2}" --csv "$CSV" --limit "$N" --out "$OUT/$1.json"; }

#лучшие настройки
run stochastic --strategy stochastic
run margin     --strategy margin --diffusion_steps 81
run adaptive   --strategy adaptive --reveal_per_step 1
run remdm      --strategy remdm --remdm_steps 324
run searchdiff --strategy searchdiff --beam_size 8
run guided     --strategy guided --guidance_lr 0.5

# перебор параметров
for ds in 20 64; do run margin_ds$ds    --strategy margin   --diffusion_steps $ds; done
for r  in 2 4;   do run adaptive_rps$r   --strategy adaptive --reveal_per_step $r;  done
for s  in 81 162;do run remdm_$s         --strategy remdm    --remdm_steps $s;      done
for lr in 1.0 2.0;do run guided_lr$lr    --strategy guided   --guidance_lr $lr;     done

# verifier-bon база x число сэмплов
for base in stochastic margin adaptive remdm searchdiff; do
  for n in 4 8 16 32; do
    run vbon_${base}_n$n --strategy verifier-bon --base_strategy $base --n_samples $n
  done
done

# лучший вариант (97.9% на N=10000)
run vbon_searchdiff_beam8_n16 --strategy verifier-bon --base_strategy searchdiff --vbon_beam_size 8 --n_samples 16

echo "Done -> $OUT"
