#!/usr/bin/env python3
"""Можно ли заменить проверку валидности сигналом самой модели (rule-free отбор в Best-of-N)?

Сэмплим N траекторий (MC-dropout, как в verifier-bon) и выбираем одну ЧЕТЫРЬМЯ способами:
  случайный        — нулевой селектор, точность одного сэмпла
  self_nll         — средний -log p, который модель даёт своим же цифрам на готовой сетке
  verifier         — символьная проверка валидности + совпадения с подсказками (как в verifier-bon)
  оракул           — есть ли верный сэмпл среди N (верхняя граница любого селектора)
Правила судоку в self_nll не используются: это мера того, насколько модели «комфортна» её же сетка.

Запуск: python scripts/entropy_selector.py --csv data_real/sudoku_hard_radcliffe.csv --n 2000 --n_samples 16
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from infer import (load_model_and_tokenizer, build_batch, read_csv_rows, get_device,
                   decode_adaptive, self_nll, decode_batch_to_solution_strings,
                   is_valid_solution, matches_quiz)

DEFAULT_CKPT = str(ROOT / "weights" / "mdm_tiny")


def auroc(pos, neg):
    import numpy as np
    if not pos or not neg:
        return float("nan")
    s = np.array(pos + neg)
    lab = np.array([1] * len(pos) + [0] * len(neg))
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - cnt / 2 + 0.5)[inv]
    n1, n0 = int(lab.sum()), int((1 - lab).sum())
    return float((ranks[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", default=DEFAULT_CKPT)
    ap.add_argument("--csv", default=str(ROOT / "data_real" / "sudoku_hard_radcliffe.csv"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--n_samples", type=int, default=16)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = get_device()
    model, tok = load_model_and_tokenizer(args.checkpoint_dir, args.checkpoint_dir, device)
    rows = read_csv_rows(args.csv, limit=args.n)
    quizzes = [q for q, _ in rows]
    sols = [s for _, s in rows]
    NS = args.n_samples

    picked_nll, picked_ver, oracle_hit, rand_rate = [], [], [], []
    score_pool, correct_pool = [], []
    start = time.time()

    for s0 in range(0, len(rows), args.batch):
        qb, sb = quizzes[s0:s0 + args.batch], sols[s0:s0 + args.batch]
        Bc = len(qb)
        cand_ok = torch.zeros(NS, Bc, dtype=torch.bool)
        cand_valid = torch.zeros(NS, Bc, dtype=torch.bool)
        cand_score = torch.zeros(NS, Bc)

        for trial in range(NS):
            torch.manual_seed(1000 * trial + 7 + s0)
            x, sm = build_batch(tok, qb, sb)
            model.train()
            xt = decode_adaptive(model, tok, x, sm, 20, device, reveal_per_step=1)
            model.eval()
            cand_score[trial] = self_nll(model, tok, xt, sm, device).cpu()
            for i, cand in enumerate(decode_batch_to_solution_strings(tok, xt, sm.to(device))):
                cand_ok[trial, i] = (cand == sb[i])
                cand_valid[trial, i] = is_valid_solution(cand) and matches_quiz(qb[i], cand)

        for i in range(Bc):
            picked_nll.append(bool(cand_ok[int(cand_score[:, i].argmin()), i]))
            v = cand_valid[:, i].nonzero().flatten()
            picked_ver.append(bool(cand_ok[v[0], i]) if len(v) else False)
            oracle_hit.append(bool(cand_ok[:, i].any()))
            rand_rate.append(float(cand_ok[:, i].float().mean()))
            for t in range(NS):
                score_pool.append(float(cand_score[t, i]))
                correct_pool.append(bool(cand_ok[t, i]))
        print(f"  ...{min(s0 + args.batch, len(rows))}/{len(rows)} ({time.time() - start:.0f}s)", flush=True)

    n = len(oracle_hit)
    res = {"n": n, "n_samples": NS,
           "random": sum(rand_rate) / n,
           "self_nll": sum(picked_nll) / n,
           "verifier": sum(picked_ver) / n,
           "oracle": sum(oracle_hit) / n,
           "auroc_self_nll": auroc([-s for s, c in zip(score_pool, correct_pool) if c],
                                   [-s for s, c in zip(score_pool, correct_pool) if not c])}
    print(f"\nN={n}, сэмплов={NS}")
    print(f"  случайный сэмпл (без отбора)   {res['random']*100:6.2f}%")
    print(f"  rule-free (self_nll)           {res['self_nll']*100:6.2f}%")
    print(f"  проверка валидности (verifier) {res['verifier']*100:6.2f}%")
    print(f"  оракул (есть верный среди N)   {res['oracle']*100:6.2f}%")
    print(f"  AUROC self_nll (верная сетка vs неверная) = {res['auroc_self_nll']:.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
