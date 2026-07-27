#!/bin/bash
# Воспроизведение бейзлайна на easy: оригинальный easy-тест репо (репо: 99.8%)
cd "$(dirname "$0")/.."
N=${1:-1000}
CSV=data_orig/sudoku_test_original.csv
OUT=results_easy_new
PY=${PY:-python}
mkdir -p "$OUT"
run() { echo ">>> $1"; $PY infer.py "${@:2}" --csv "$CSV" --limit "$N" --out "$OUT/$1.json"; }

run stochastic --strategy stochastic
run margin     --strategy margin

echo "Done -> $OUT"
