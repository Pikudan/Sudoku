#!/usr/bin/env python3
"""Где ошибается базовый декодер и почему single-pass упирается в ~42-45%.

Инструментированный greedy margin-декод (1 клетка за шаг) записывает для каждого коммита
ранг истинной цифры, уверенность и был ли контекст ещё «чистым» (без предыдущих ошибок).
Дальше — потолки классов методов через оракулы:
  fix_k : починить первые k неверных коммита и продолжить жадно
  topK  : всегда выбирать истину, если она в top-K (потолок выбора значения)
  order : всегда брать клетку, где argmax верен (потолок методов порядка)

Запуск: python scripts/base_analysis.py --csv data_real/sudoku_hard_radcliffe.csv --n 2000
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from infer import (load_model_and_tokenizer, build_batch, logits_shifted,
                   _answer_mask_and_eos, read_csv_rows, get_device)

DEFAULT_CKPT = str(ROOT / "weights" / "mdm_tiny")


def entropy(logp):
    return -(logp.exp() * logp).sum(-1)


@torch.no_grad()
def instrumented(model, tok, x, sm, device, digit_ids):
    V = tok.vocab_size
    x = x.to(device); sm = sm.to(device)
    truth = x.clone()
    B = x.size(0)
    answer_mask, _, xt = _answer_mask_and_eos(x, sm, tok, device)
    remaining = answer_mask.clone()
    rowsel = torch.arange(B, device=device)
    t0 = torch.zeros(B, dtype=torch.long, device=device)
    dig = torch.tensor(digit_ids, device=device)
    clean = torch.ones(B, dtype=torch.bool, device=device)
    recs = []
    for step in range(81):
        if not remaining.any():
            break
        logits = logits_shifted(model, xt, t0, torch.ones_like(x))
        logits[:, :, V:] = -1000.0
        logp = torch.log_softmax(logits, dim=-1)
        p = logp.exp()
        t2 = p.topk(2, dim=-1).values
        margin = (t2[..., 0] - t2[..., 1]).masked_fill(~remaining, -1e4)
        cell = margin.argmax(dim=1)
        cl = logp[rowsel, cell]
        chosen = cl.argmax(-1)
        tr = truth[rowsel, cell]
        corr = chosen == tr
        dl = cl[:, dig]
        tr_idx = (tr.unsqueeze(1) == dig.unsqueeze(0)).float().argmax(1)
        rank = (dl > dl[rowsel, tr_idx].unsqueeze(1)).sum(1) + 1
        any_ok = ((logp.argmax(-1) == truth) & remaining).any(1)
        alive = remaining.any(1)
        for b in range(B):
            if alive[b]:
                recs.append(dict(step=step, rank=int(rank[b]), corr=bool(corr[b]),
                                 clean=bool(clean[b]), any_ok=bool(any_ok[b]), puz=b))
        clean = clean & corr
        xt[rowsel, cell] = chosen
        remaining[rowsel, cell] = False
    grid_ok = (xt[answer_mask] == truth[answer_mask]).view(B, -1).all(1)
    return recs, grid_ok, answer_mask.sum(1)


@torch.no_grad()
def oracle(model, tok, x, sm, device, digit_ids, mode, k=1, K=2):
    V = tok.vocab_size
    x = x.to(device); sm = sm.to(device)
    truth = x.clone()
    B = x.size(0)
    answer_mask, _, xt = _answer_mask_and_eos(x, sm, tok, device)
    remaining = answer_mask.clone()
    rowsel = torch.arange(B, device=device)
    t0 = torch.zeros(B, dtype=torch.long, device=device)
    dig = torch.tensor(digit_ids, device=device)
    fixes_left = torch.full((B,), k, device=device)
    for step in range(81):
        if not remaining.any():
            break
        logits = logits_shifted(model, xt, t0, torch.ones_like(x))
        logits[:, :, V:] = -1000.0
        logp = torch.log_softmax(logits, dim=-1)
        p = logp.exp()
        t2 = p.topk(2, dim=-1).values
        margin = (t2[..., 0] - t2[..., 1]).masked_fill(~remaining, -1e4)
        if mode == "order":
            ok = (logp.argmax(-1) == truth) & remaining
            m2 = margin.masked_fill(~ok, -1e4)
            cell = torch.where(ok.any(1), m2.argmax(1), margin.argmax(1))
        else:
            cell = margin.argmax(dim=1)
        cl = logp[rowsel, cell]
        chosen = cl.argmax(-1)
        tr = truth[rowsel, cell]
        if mode == "fix_k":
            use = (chosen != tr) & remaining.any(1) & (fixes_left > 0)
            chosen = torch.where(use, tr, chosen)
            fixes_left = fixes_left - use.long()
        elif mode == "topK":
            dl = cl[:, dig]
            tr_idx = (tr.unsqueeze(1) == dig.unsqueeze(0)).float().argmax(1)
            rank = (dl > dl[rowsel, tr_idx].unsqueeze(1)).sum(1) + 1
            chosen = torch.where(rank <= K, tr, chosen)
        xt[rowsel, cell] = chosen
        remaining[rowsel, cell] = False
    return (xt[answer_mask] == truth[answer_mask]).view(B, -1).all(1).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", default=DEFAULT_CKPT)
    ap.add_argument("--csv", default=str(ROOT / "data_real" / "sudoku_hard_radcliffe.csv"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = get_device()
    torch.manual_seed(0)
    model, tok = load_model_and_tokenizer(args.checkpoint_dir, args.checkpoint_dir, device)
    digit_ids = [tok.encode(str(d))[0] for d in range(1, 10)]
    rows = read_csv_rows(args.csv, limit=args.n)

    recs, grid_ok, n_masked = [], [], []
    start = time.time()
    for s0 in range(0, len(rows), args.batch):
        sl = slice(s0, min(s0 + args.batch, len(rows)))
        x, sm = build_batch(tok, [q for q, _ in rows[sl]], [s for _, s in rows[sl]])
        r, g, nm = instrumented(model, tok, x, sm, device, digit_ids)
        for rec in r:
            rec["puz"] += s0
        recs += r; grid_ok.append(g); n_masked.append(nm)
    grid_ok = torch.cat(grid_ok); n_masked = torch.cat(n_masked)
    base_acc = float(grid_ok.float().mean())
    n_cells = int(n_masked.float().median())
    print(f"паззлов={len(rows)}  точность бейзлайна={base_acc:.4f}  ({time.time()-start:.0f}s)", flush=True)

    clean = [r for r in recs if r["clean"]]
    n_root = sum(1 for r in clean if not r["corr"])
    p_root = n_root / max(1, len(clean))
    print(f"\nкорневых ошибок {n_root} на {len(clean)} чистых коммитов -> p={p_root:.5f}")
    print(f"предсказание (1-p)^{n_cells} = {(1-p_root)**n_cells:.4f}   (факт {base_acc:.4f})")
    forced = sum(1 for r in clean if not r["any_ok"])
    print(f"шагов без единой верной клетки (чистый контекст): {forced} из {len(clean)}")

    def hist(sub, label):
        c = collections.Counter(r["rank"] for r in sub); n = max(1, len(sub))
        cols = "  ".join(f"r{i}:{100*c.get(i,0)/n:5.2f}%" for i in (1, 2, 3))
        print(f"  {label:26s} n={n:6d}  {cols}  r>3:{100*sum(v for k,v in c.items() if k>3)/n:5.2f}%")

    print("\nранг истинной цифры на момент коммита:")
    hist(recs, "все коммиты")
    hist([r for r in recs if r["corr"] and r["clean"]], "без промахов")
    hist([r for r in recs if not r["corr"] and r["clean"]], "первый промах")
    hist([r for r in recs if not r["corr"] and not r["clean"]], "после первого промаха")

    per_puz = collections.Counter()
    root_puz = collections.Counter()
    for r in recs:
        if not r["corr"]:
            per_puz[r["puz"]] += 1
            if r["clean"]:
                root_puz[r["puz"]] += 1
    fails = [i for i in range(len(rows)) if not grid_ok[i]]
    errs = [per_puz.get(i, 0) for i in fails]
    roots = [root_puz.get(i, 0) for i in fails]
    print(f"\nпровалов={len(fails)}; неверных клеток на провал: среднее {sum(errs)/max(1,len(errs)):.1f}")
    print(f"корневых ошибок на провал: среднее {sum(roots)/max(1,len(roots)):.2f}, "
          f"ровно одна в {100*sum(1 for r in roots if r==1)/max(1,len(roots)):.1f}% провалов")

    print("\nпотолки классов методов (оракулы):")
    res = {}
    for label, kw in (("починить 1-ю ошибку", dict(mode="fix_k", k=1)),
                      ("починить первые 2", dict(mode="fix_k", k=2)),
                      ("идеальный выбор в top-2", dict(mode="topK", K=2)),
                      ("идеальный выбор в top-3", dict(mode="topK", K=3)),
                      ("идеальный порядок", dict(mode="order",))):
        accs = []
        for s0 in range(0, len(rows), args.batch):
            sl = slice(s0, min(s0 + args.batch, len(rows)))
            x, sm = build_batch(tok, [q for q, _ in rows[sl]], [s for _, s in rows[sl]])
            accs.append(oracle(model, tok, x, sm, device, digit_ids, **kw))
        res[label] = sum(accs) / len(accs)
        print(f"  {label:26s} {res[label]:.4f}", flush=True)

    if args.out:
        json.dump({"base_acc": base_acc, "p_root": p_root, "n_cells": n_cells,
                   "oracles": res}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
