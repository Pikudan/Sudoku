#!/usr/bin/env python3
"""
Build easy/hard Sudoku eval splits from the public sudoku-extreme benchmark
(sapientinc/sudoku-extreme, HRM paper) into the repo's expected CSV schema
(quizzes,solutions with 0 = blank).

This reproduces the repo's *definition* of the hard split with real, citable,
auth-free public data. The source CSV has columns:
    source,question,answer,rating
where `question` uses '.' for blanks, `answer` is the 81-char solution, and
`rating` = number of backtracks the tdoku solver needs (0 = solvable by pure
constraint propagation = Shah et al. "no-backtracking / easy"; >=1 = requires
search = "hard"). This is exactly the easy/hard criterion the repo README uses
(Radcliffe minus Shah-easy no-backtracking subset).

Outputs:
    <out>/sudoku_test.csv  : rating == 0 (easy), sampled to --easy_n
    <out>/sudoku_hard.csv  : rating >= --hard_min_rating (hard), sampled to --hard_n
    <out>/dataset_info.json

Usage:
    python build_real_splits.py --src data_real/sudoku_extreme_test.csv \
        --out data_real --easy_n 1000 --hard_n 10000 --seed 42

ТОЧНАЯ команда, которой собран hard-сплит из README (все числа N=2000/10000 считаны на нём):
    python build_real_splits.py --src data_real/sudoku_extreme_test.csv --out data_real \
        --easy_n 1000 --hard_n 10000 --seed 42 \
        --hard_sources puzzles1_unbiased --hard_name sudoku_hard_radcliffe
(без --hard_sources получится смесь источников с другим распределением подсказок — числа не совпадут)
"""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


def dot_to_zero(q):
    return q.replace(".", "0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="data_real")
    ap.add_argument("--easy_n", type=int, default=1000)
    ap.add_argument("--hard_n", type=int, default=10000)
    ap.add_argument("--hard_name", default="sudoku_hard",
                     help="имя выходного hard-файла без .csv")
    ap.add_argument("--hard_min_rating", type=int, default=1,
                     help="min tdoku backtrack count to count as 'hard' (1 = any backtracking)")
    ap.add_argument("--hard_sources", default=None,
                     help="comma-separated source prefixes to restrict the hard split to "
                          "(e.g. 'puzzles0_kaggle,puzzles1_unbiased' to stay close to Radcliffe); "
                          "default = all sources")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    hard_source_filter = None
    if args.hard_sources:
        hard_source_filter = tuple(s.strip() for s in args.hard_sources.split(","))

    easy, hard = [], []
    rating_hist = Counter()
    source_hist = Counter()
    n = 0
    with open(args.src, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            n += 1
            q = dot_to_zero(row["question"])
            s = row["answer"]
            if len(q) != 81 or len(s) != 81:
                continue
            rating = int(row["rating"])
            src = row["source"]
            rating_hist[min(rating, 10)] += 1
            source_hist[src] += 1
            if rating == 0:
                easy.append((q, s))
            elif rating >= args.hard_min_rating:
                if hard_source_filter is None or src.startswith(hard_source_filter):
                    hard.append((q, s))

    print(f"[read] {n} rows; easy(rating0)={len(easy)} hard(rating>={args.hard_min_rating})={len(hard)}")
    print(f"[rating hist (capped at 10)] {dict(sorted(rating_hist.items()))}")
    print(f"[sources] {dict(source_hist)}")

    rng.shuffle(easy)
    rng.shuffle(hard)
    easy = easy[: args.easy_n]
    hard = hard[: args.hard_n]

    def write(path, rows):
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["quizzes", "solutions"])
            w.writerows(rows)

    write(out / "sudoku_test.csv", easy)
    write(out / f"{args.hard_name}.csv", hard)
    info = {
        name: {"file_name": f"{name}.csv",
               "columns": {"prompt": "quizzes", "query": "", "response": "solutions", "history": ""}}
        for name in ("sudoku_test", args.hard_name)
    }
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2))
    print(f"[write] {out}/sudoku_test.csv ({len(easy)}) {out}/{args.hard_name}.csv ({len(hard)})")


if __name__ == "__main__":
    main()
