#!/usr/bin/env python3
"""
Train the constraint-violation value net (weights/value_net) used by --strategy guided.
It takes an 81-cell board as soft digit distributions and predicts whether the grid is
valid (BCE), trained on true solutions and corrupted copies. guided decoding then follows
its gradient to push the denoiser's logits toward constraint satisfaction.

Usage:
  python train_value_net.py --train_csv data/sudoku_train.csv --steps 2000 --out_dir runs/value_net
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from infer import get_device, read_csv_rows, ValueNet

N = 9


def box_index(r, c):
    return (r // 3) * 3 + (c // 3)


def count_violations(grid81):
    def unit_dups(vals):
        from collections import Counter
        cnt = Counter(v for v in vals if 1 <= v <= 9)
        return sum(c * (c - 1) // 2 for c in cnt.values())
    total = 0
    g = grid81
    for r in range(9):
        total += unit_dups([g[r * 9 + c] for c in range(9)])
    for c in range(9):
        total += unit_dups([g[r * 9 + c] for r in range(9)])
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            total += unit_dups([g[(br + i) * 9 + (bc + j)] for i in range(3) for j in range(3)])
    return total


def solution_to_onehot(sol_str):
    idx = torch.tensor([int(ch) - 1 for ch in sol_str], dtype=torch.long)
    return F.one_hot(idx, num_classes=9).float()


def corrupt(sol_str, rng, max_flips=20):
    g = [int(ch) for ch in sol_str]
    mode = rng.random()
    if mode < 0.45:
        k = rng.randint(1, max_flips)
        cells = rng.sample(range(81), k)
        for i in cells:
            wrong = rng.randint(1, 9)
            while wrong == g[i]:
                wrong = rng.randint(1, 9)
            g[i] = wrong
        oh = F.one_hot(torch.tensor([v - 1 for v in g]), 9).float()
        return oh, count_violations(g)
    elif mode < 0.7:
        oh = F.one_hot(torch.tensor([v - 1 for v in g]), 9).float()
        return oh, 0
    else:
        oh = F.one_hot(torch.tensor([v - 1 for v in g]), 9).float()
        k = rng.randint(5, 40)
        cells = rng.sample(range(81), k)
        for i in cells:
            oh[i] = torch.full((9,), 1.0 / 9)
        return oh, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--max_train_rows", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"[device] {device}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(args.train_csv, limit=args.max_train_rows)
    sols = [s for _, s in rows]
    print(f"[data] {len(sols)} solutions", flush=True)

    net = ValueNet(d=args.d).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[value-net] {n_params/1e3:.1f}K params", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)

    start = time.time()
    running = 0.0
    log = []
    for step in range(1, args.steps + 1):
        batch_oh, batch_lbl = [], []
        for _ in range(args.batch_size):
            s = sols[rng.randrange(len(sols))]
            oh, y = corrupt(s, rng)
            batch_oh.append(oh)
            batch_lbl.append(1.0 if y > 0 else 0.0)
        x = torch.stack(batch_oh).to(device)
        lbl = torch.tensor(batch_lbl, device=device)
        logit = net(x)
        loss = F.binary_cross_entropy_with_logits(logit, lbl)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        running += loss.item()
        if step % 100 == 0 or step == 1:
            elapsed = time.time() - start
            avg = running / (100 if step > 1 else 1)
            running = 0.0
            with torch.no_grad():
                s = sols[0]
                p_valid = torch.sigmoid(net(solution_to_onehot(s).unsqueeze(0).to(device))).item()
                bad_oh, bad_y = corrupt(s, random.Random(step), max_flips=20)
                p_bad = torch.sigmoid(net(bad_oh.unsqueeze(0).to(device))).item()
            print(f"step={step}/{args.steps} bce={avg:.4f} P(invalid|valid_grid)={p_valid:.3f} "
                  f"P(invalid|corrupt,{bad_y}viol)={p_bad:.3f} elapsed={elapsed:.1f}s", flush=True)
            log.append({"step": step, "loss": avg, "p_invalid_valid": p_valid, "p_invalid_corrupt": p_bad})

    with torch.no_grad():
        vs, bs = [], []
        for i in range(200):
            s = sols[i % len(sols)]
            vs.append(torch.sigmoid(net(solution_to_onehot(s).unsqueeze(0).to(device))).item())
            bad_oh, _ = corrupt(s, random.Random(10000 + i), max_flips=15)
            bs.append(torch.sigmoid(net(bad_oh.unsqueeze(0).to(device))).item())
    sep = sum(b > v for v, b in zip(vs, bs)) / len(vs)
    print(f"[gate] mean P(invalid): valid={np.mean(vs):.3f} corrupt={np.mean(bs):.3f} "
          f"separated={sep:.2%} (want high)", flush=True)

    torch.save(net.state_dict(), out_dir / "value_net.bin")
    with (out_dir / "value_config.json").open("w") as f:
        json.dump({"d": args.d, "args": vars(args)}, f, indent=2)
    with (out_dir / "train_log.json").open("w") as f:
        json.dump({"log": log, "total_time_sec": time.time() - start}, f, indent=2)
    print(f"[done] saved value net to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
