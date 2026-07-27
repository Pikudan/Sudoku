#!/usr/bin/env python3
"""
Sudoku inference

Decoding strategies (--strategy):
  stochastic    baseline in the repo (gumbel-noise top-k schedule)
  margin        the repo's improvement: rank cells by top1-top2 probability gap
  adaptive      margin ranking but reveal ~1 cell per step (== margin with many diffusion steps)
  guided        constraint-value-network guided decoding (needs weights/value_net)
  remdm         remasking + inference-time step scaling
  searchdiff    constrained beam search over the decode
  verifier-bon  run a base strategy N times and keep the completion that is a valid
                Sudoku consistent with the clues (a valid+consistent grid is the unique answer)

Examples:
  python infer.py --strategy margin       --csv data/sudoku_hard.csv --limit 2000
  python infer.py --strategy adaptive     --csv data/sudoku_hard.csv --reveal_per_step 1
  python infer.py --strategy verifier-bon --base_strategy searchdiff --n_samples 8 --csv data/sudoku_hard.csv
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from llmtuner.tuner.core.custom_tokenizer import CustomTokenizer
from llmtuner.tuner.mdm.model import DiffusionModel
from transformers import GPT2Config, GPT2LMHeadModel

DEFAULT_CKPT = str(ROOT / "weights" / "mdm_tiny")
DEFAULT_VALUE = str(ROOT / "weights" / "value_net")


class ValueNet(nn.Module):

    def __init__(self, d=96):
        super().__init__()
        self.register_buffer("incidence", _build_incidence("cpu"))
        self.head = nn.Sequential(nn.Linear(27 * 9, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, soft_digits):
        unit_counts = torch.einsum("uc,bcd->bud", self.incidence, soft_digits)
        excess = torch.relu(unit_counts - 1.0)
        return self.head(excess.flatten(1)).squeeze(-1)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_tokenizer(checkpoint_dir, model_config_dir, device):
    tokenizer = CustomTokenizer.from_pretrained(checkpoint_dir)
    config = GPT2Config.from_pretrained(model_config_dir)
    base_model = GPT2LMHeadModel(config)
    model = DiffusionModel(base_model, config, diffusion_args=None)
    state = torch.load(Path(checkpoint_dir) / "pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer


def read_csv_rows(path, limit=None):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if limit is not None and len(rows) >= limit:   # check before append, else --limit 0 yields 1 row
                break
            q, s = row["quizzes"], row["solutions"]
            assert len(q) == 81 and len(s) == 81, \
                f"{path} row {len(rows)+1}: expected 81 chars each, got quiz={len(q)} solution={len(s)}"
            rows.append((q, s))
    return rows


def build_batch(tokenizer, quizzes, solutions, cutoff_len=164):
    input_ids_list, src_mask_list = [], []
    for q, s in zip(quizzes, solutions):
        src_ids = tokenizer.encode(q)
        tgt_ids = tokenizer.encode(s)
        tgt_ids = tgt_ids[: cutoff_len - 2]
        src_ids = src_ids[-(cutoff_len - 2 - len(tgt_ids)) :]
        ids = src_ids + [tokenizer.sep_token_id] + tgt_ids + [tokenizer.eos_token_id]
        src_mask = [1] * (len(src_ids) + 1) + [0] * (len(ids) - len(src_ids) - 1)
        if len(ids) < cutoff_len:
            pad = cutoff_len - len(ids)
            ids = ids + [tokenizer.pad_token_id] * pad
            src_mask = src_mask + [0] * pad
        else:
            ids = ids[:cutoff_len]
            src_mask = src_mask[:cutoff_len]
        input_ids_list.append(ids)
        src_mask_list.append(src_mask)
    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    src_mask = torch.tensor(src_mask_list, dtype=torch.bool)
    return input_ids, src_mask


def logits_shifted(model, xt, t_tensor, attention_mask):
    logits = model(xt, t_tensor, attention_mask=attention_mask)
    logits = torch.cat([logits[:, 0:1], logits[:, :-1]], dim=1)
    return logits


def _predict(model, xt, t_tensor, attention_mask, vocab_size):
    logits = logits_shifted(model, xt, t_tensor, attention_mask)
    logits[:, :, vocab_size:] = -1000
    logp = torch.log_softmax(logits, dim=-1)
    top2 = torch.softmax(logits, dim=-1).topk(k=2, dim=-1).values
    return logp, top2[:, :, 0] - top2[:, :, 1], logits.argmax(-1)


def _answer_mask_and_eos(x, src_mask, tokenizer, device):
    maskable = ~src_mask
    B, L = x.size(0), x.size(1)
    idx_row = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    eos_idx = (maskable * idx_row).max(dim=1).values
    answer_mask = maskable.clone()
    answer_mask[torch.arange(B, device=device), eos_idx] = False
    xt = x.masked_fill(maskable, tokenizer.mask_token_id)
    xt[torch.arange(B, device=device), eos_idx] = tokenizer.eos_token_id
    return answer_mask, eos_idx, xt


def _build_incidence(device):
    M = torch.zeros(27, 81, device=device)
    for r in range(9):
        for c in range(9):
            cell = r * 9 + c
            M[r, cell] = 1.0
            M[9 + c, cell] = 1.0
            M[18 + (r // 3) * 3 + (c // 3), cell] = 1.0
    return M


def _violations(board_digits, incidence):
    R = board_digits.size(0)
    onehot = torch.zeros(R, 81, 9, device=board_digits.device)
    filled = board_digits > 0
    dig_idx = (board_digits.clamp(min=1) - 1)
    onehot.scatter_(2, dig_idx.unsqueeze(-1), filled.float().unsqueeze(-1))
    counts = torch.einsum("uc,rcd->rud", incidence, onehot)
    dup = (counts * (counts - 1) / 2.0)
    return dup.sum(dim=(1, 2))


@torch.no_grad()
def decode_stochastic_or_margin(model, tokenizer, x, src_mask, diffusion_steps, mode, device, topk_noise=0.5, schedule="linear"):
    x = x.to(device)
    src_mask = src_mask.to(device)
    attention_mask = torch.ones_like(x)
    batch_size = x.size(0)
    maskable_mask = ~src_mask
    init_maskable_mask = maskable_mask.clone()

    xt = x.masked_fill(maskable_mask, tokenizer.mask_token_id)
    for t in range(diffusion_steps - 1, -1, -1):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        logits = logits_shifted(model, xt, t_tensor, attention_mask)
        scores = torch.log_softmax(logits, dim=-1)
        scores[:, :, tokenizer.vocab_size :] = -1000
        x0_scores, x0 = scores.max(-1)
        if mode == "margin":
            probs = torch.softmax(scores, dim=-1)
            top2 = probs.topk(k=2, dim=-1).values
            x0_scores = top2[:, :, 0] - top2[:, :, 1]

        x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])

        if t > 0:
            rate = t / diffusion_steps if schedule == "linear" else np.cos((diffusion_steps - t) / diffusion_steps * np.pi * 0.5)
            cutoff_len = (init_maskable_mask.sum(1, keepdim=True) * rate).long()
            scores_for_topk = x0_scores.masked_fill(~init_maskable_mask, 1000.0)
            if mode == "stochastic":
                gumbel = -torch.log(-torch.log(torch.rand_like(scores_for_topk) + 1e-8) + 1e-8)
                _scores = scores_for_topk + topk_noise * rate * gumbel
            else:
                _scores = scores_for_topk
            sorted_scores = _scores.sort(-1)[0]
            cutoff = sorted_scores.gather(-1, cutoff_len)
            lowest_k_mask = _scores < cutoff
            xt = x0.masked_scatter(lowest_k_mask, torch.full_like(x0[lowest_k_mask], tokenizer.mask_token_id))
        else:
            xt = x0
    return xt


@torch.no_grad()
def decode_adaptive(model, tokenizer, x, src_mask, diffusion_steps, device, reveal_per_step=None):
    x = x.to(device)
    src_mask = src_mask.to(device)
    attention_mask = torch.ones_like(x)
    maskable_mask = ~src_mask
    remaining = maskable_mask.clone()
    xt = x.masked_fill(maskable_mask, tokenizer.mask_token_id)

    if reveal_per_step is None:
        n_masked_max = int(maskable_mask.sum(1).max().item())
        reveal_per_step = max(1, (n_masked_max + diffusion_steps - 1) // diffusion_steps)

    init_count = maskable_mask.sum(1).clamp(min=1).float()
    while remaining.any():
        frac_remaining = remaining.sum(1).float() / init_count
        t_tensor = (frac_remaining * (diffusion_steps - 1)).round().long().clamp(0, diffusion_steps - 1)
        _, margin, x0 = _predict(model, xt, t_tensor, attention_mask, tokenizer.vocab_size)

        margin_for_topk = margin.masked_fill(~remaining, -1000.0)
        k = min(reveal_per_step, int(remaining.sum(1).max().item()))
        if k <= 0:
            break
        cutoff = margin_for_topk.topk(k=k, dim=-1).values[:, -1:]
        reveal_mask = (margin_for_topk >= cutoff) & remaining

        xt = xt.masked_scatter(reveal_mask, x0[reveal_mask])
        remaining = remaining & ~reveal_mask
    return xt


def decode_guided(model, tokenizer, x, src_mask, diffusion_steps, device, value_net,
                  reveal_per_step=1, guidance_steps=3, guidance_lr=1.0, kl_lambda=1.0):
    x = x.to(device)
    src_mask = src_mask.to(device)
    attention_mask = torch.ones_like(x)
    B = x.size(0)

    answer_mask, _, xt = _answer_mask_and_eos(x, src_mask, tokenizer, device)

    remaining = answer_mask.clone()
    init_count = answer_mask.sum(1).clamp(min=1).float()

    V = tokenizer.vocab_size
    digit_ids = torch.tensor([tokenizer.encode(str(d))[0] for d in range(1, 10)], device=device)

    while remaining.any():
        frac_remaining = remaining.sum(1).float() / init_count
        t_tensor = (frac_remaining * (diffusion_steps - 1)).round().long().clamp(0, diffusion_steps - 1)
        with torch.no_grad():
            base_logits = logits_shifted(model, xt, t_tensor, attention_mask)
            base_logits[:, :, V:] = -1000

        digit_logits = base_logits[:, :, digit_ids]
        board_logits = digit_logits[answer_mask].view(B, 81, 9).clone().detach().requires_grad_(True)
        p0 = torch.softmax(board_logits, dim=-1).detach()

        cur = board_logits
        for _ in range(guidance_steps):
            probs = torch.softmax(cur, dim=-1)
            v = value_net(probs).sum()
            kl = (p0 * (torch.log(p0 + 1e-9) - torch.log_softmax(cur, dim=-1))).sum()
            grad = torch.autograd.grad(v + kl_lambda * kl, cur, retain_graph=False)[0]
            cur = (cur - guidance_lr * grad).detach().requires_grad_(True)
        guided_board = cur.detach()

        gprobs = torch.softmax(guided_board, dim=-1)
        top2 = gprobs.topk(k=2, dim=-1).values
        margin_board = top2[:, :, 0] - top2[:, :, 1]
        val_board = gprobs.argmax(-1)

        margin_seq = torch.zeros_like(xt, dtype=torch.float)
        margin_seq[answer_mask] = margin_board.reshape(-1)
        chosen_digit_token = torch.zeros_like(xt)
        chosen_digit_token[answer_mask] = digit_ids[val_board.reshape(-1)]

        margin_for_topk = margin_seq.masked_fill(~remaining, -1e4)
        k = min(reveal_per_step, int(remaining.sum(1).max().item()))
        if k <= 0:
            break
        cutoff = margin_for_topk.topk(k=k, dim=-1).values[:, -1:]
        reveal_mask = (margin_for_topk >= cutoff) & remaining
        xt = torch.where(reveal_mask, chosen_digit_token, xt)
        remaining = remaining & ~reveal_mask
    return xt


def load_value_net(value_dir, device):
    cfg = json.load(open(Path(value_dir) / "value_config.json"))
    net = ValueNet(d=cfg["d"])
    net.load_state_dict(torch.load(Path(value_dir) / "value_net.bin", map_location="cpu"))
    net.to(device).eval()
    net.requires_grad_(False)
    return net


@torch.no_grad()
def decode_remdm(model, tokenizer, x, src_mask, diffusion_steps, device, num_steps=162):
    x = x.to(device)
    src_mask = src_mask.to(device)
    attention_mask = torch.ones_like(x)
    B = x.size(0)
    answer_mask, _, xt = _answer_mask_and_eos(x, src_mask, tokenizer, device)
    V = tokenizer.vocab_size
    n_ans = 81

    for step in range(num_steps):
        frac = 1.0 - (step + 1) / num_steps
        t_tensor = torch.full((B,), int(round(frac * (diffusion_steps - 1))), device=device, dtype=torch.long)
        _, margin, x0 = _predict(model, xt, t_tensor, attention_mask, V)

        marg_board = margin[answer_mask].view(B, n_ans)
        x0_board = x0[answer_mask].view(B, n_ans)
        k = int(round(frac * n_ans))
        if k <= 0:
            new_masked_board = torch.zeros(B, n_ans, dtype=torch.bool, device=device)
        else:
            # (torch.kthvalue is unimplemented on MPS, so use sort + index.)
            sorted_marg, _ = marg_board.sort(dim=1)
            thresh = sorted_marg[:, k - 1:k]
            new_masked_board = marg_board <= thresh

        new_tokens = torch.where(new_masked_board, torch.full_like(x0_board, tokenizer.mask_token_id), x0_board)
        xt = xt.clone()
        xt[answer_mask] = new_tokens.reshape(-1)
    return xt


def _board_digits(xt, answer_mask, digit_ids):
    cell = xt[answer_mask].view(xt.size(0), 81)
    dig = cell - int(digit_ids[0]) + 1
    return torch.where((cell >= digit_ids[0]) & (cell <= digit_ids[-1]), dig, torch.zeros_like(dig))


@torch.no_grad()
def decode_searchdiff(model, tokenizer, x, src_mask, diffusion_steps, device,
                      beam_size=4, cand=3, feas_lambda=3.0):
    x = x.to(device)
    src_mask = src_mask.to(device)
    B = x.size(0)
    incidence = _build_incidence(device)
    V = tokenizer.vocab_size
    digit_ids = torch.tensor([tokenizer.encode(str(d))[0] for d in range(1, 10)], device=device)

    # beam_size slots per puzzle, but only ONE starts alive: with identical copies every clone
    answer_mask, _, xt = _answer_mask_and_eos(x, src_mask, tokenizer, device)
    xt = xt.repeat_interleave(beam_size, dim=0)
    answer_mask = answer_mask.repeat_interleave(beam_size, dim=0)
    remaining = answer_mask.clone()
    R = xt.size(0)
    rows = torch.arange(R, device=device)
    score = torch.where(rows % beam_size == 0, torch.zeros(R, device=device),
                        torch.full((R,), -1e9, device=device))

    for _ in range(81):
        if not remaining.any():
            break
        logp, margin, _ = _predict(model, xt, torch.zeros(R, dtype=torch.long, device=device),
                                   torch.ones_like(xt), V)
        margin = margin.masked_fill(~remaining, -1e4)

        active = remaining.any(dim=1)
        cell = margin.argmax(dim=1)
        top = logp[rows, cell][:, digit_ids].topk(k=cand, dim=1)

        child_xt = xt.repeat_interleave(cand, dim=0).clone()
        child_rem = remaining.repeat_interleave(cand, dim=0).clone()
        child_score = score.repeat_interleave(cand, dim=0)
        child_cell = cell.repeat_interleave(cand, dim=0)
        child_tok = digit_ids[top.indices].reshape(-1)
        child_lp = top.values.reshape(-1)
        commit = active.repeat_interleave(cand, dim=0)
        crows = torch.arange(R * cand, device=device)

        child_xt[crows[commit], child_cell[commit]] = child_tok[commit]
        child_rem[crows[commit], child_cell[commit]] = False
        child_score = child_score + torch.where(commit, child_lp, torch.zeros_like(child_lp))

        viol = _violations(_board_digits(child_xt, answer_mask.repeat_interleave(cand, dim=0), digit_ids), incidence)
        keep = (child_score - feas_lambda * viol).view(B, beam_size * cand).topk(k=beam_size, dim=1).indices
        flat = (torch.arange(B, device=device).unsqueeze(1) * (beam_size * cand) + keep).reshape(-1)
        xt, remaining, score = child_xt[flat], child_rem[flat], child_score[flat]

    final = score - feas_lambda * _violations(_board_digits(xt, answer_mask, digit_ids), incidence)
    best = torch.arange(B, device=device) * beam_size + final.view(B, beam_size).argmax(dim=1)
    return xt[best]

def is_valid_solution(sol_str):
    if len(sol_str) != 81 or any(ch not in "123456789" for ch in sol_str):
        return False
    g = [[int(sol_str[r * 9 + c]) for c in range(9)] for r in range(9)]
    for r in range(9):
        if sorted(g[r]) != list(range(1, 10)):
            return False
    for c in range(9):
        if sorted(g[r][c] for r in range(9)) != list(range(1, 10)):
            return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            vals = [g[br + i][bc + j] for i in range(3) for j in range(3)]
            if sorted(vals) != list(range(1, 10)):
                return False
    return True


def matches_quiz(quiz_str, sol_str):
    return all(a == "0" or a == b for a, b in zip(quiz_str, sol_str))


@torch.no_grad()
def decode_verifier_bon(model, tokenizer, x, src_mask, diffusion_steps, device, base_strategy, n_samples, quizzes,
                        reveal_per_step=None, vbon_beam_size=4, value_net=None, select="verifier",
                        remdm_steps=162, cand=3, feas_lambda=3.0,
                        guidance_steps=3, guidance_lr=1.0, kl_lambda=1.0):
    batch_size = x.size(0)
    best_strs = [None] * batch_size
    found_valid = [False] * batch_size

    bases = [b.strip() for b in base_strategy.split(",") if b.strip()]
    known = {"stochastic", "margin", "adaptive", "searchdiff", "remdm", "guided"}
    unknown = [b for b in bases if b not in known]
    assert not unknown, f"unknown verifier-bon base: {unknown} (expected one of {sorted(known)})"
    rps = reveal_per_step if reveal_per_step is not None else 1
    use_mc_dropout = any(b != "stochastic" for b in bases)
    if use_mc_dropout:
        model.train()

    def decode_one(base):
        if base in ("stochastic", "margin"):
            return decode_stochastic_or_margin(model, tokenizer, x, src_mask, diffusion_steps, base, device)
        if base == "searchdiff":
            return decode_searchdiff(model, tokenizer, x, src_mask, diffusion_steps, device,
                                     beam_size=vbon_beam_size, cand=cand, feas_lambda=feas_lambda)
        if base == "remdm":
            return decode_remdm(model, tokenizer, x, src_mask, diffusion_steps, device, num_steps=remdm_steps)
        if base == "guided":
            assert value_net is not None, "guided base needs value_net"
            with torch.enable_grad():
                return decode_guided(model, tokenizer, x, src_mask, diffusion_steps, device, value_net,
                                     reveal_per_step=rps, guidance_steps=guidance_steps,
                                     guidance_lr=guidance_lr, kl_lambda=kl_lambda)
        return decode_adaptive(model, tokenizer, x, src_mask, diffusion_steps, device, reveal_per_step=rps)

    best_score = [float("inf")] * batch_size

    for trial in range(n_samples):
        torch.manual_seed(1000 * trial + 7)
        xt = decode_one(bases[trial % len(bases)])
        candidates = decode_batch_to_solution_strings(tokenizer, xt, src_mask)

        if select == "entropy":
            if use_mc_dropout:
                model.eval()
            score = self_nll(model, tokenizer, xt, src_mask, device)
            if use_mc_dropout:
                model.train()
            for i, candidate in enumerate(candidates):
                if float(score[i]) < best_score[i]:
                    best_score[i] = float(score[i])
                    best_strs[i] = candidate
            continue

        for i, candidate in enumerate(candidates):
            if found_valid[i]:
                continue
            valid = is_valid_solution(candidate) and matches_quiz(quizzes[i], candidate)
            if valid:
                best_strs[i] = candidate
                found_valid[i] = True
            elif best_strs[i] is None:
                best_strs[i] = candidate

    if use_mc_dropout:
        model.eval()
    return best_strs


@torch.no_grad()
def self_nll(model, tokenizer, xt, src_mask, device):
    V = tokenizer.vocab_size
    xt = xt.to(device)
    answer_mask, _, _ = _answer_mask_and_eos(xt, src_mask.to(device), tokenizer, device)
    t_tensor = torch.zeros(xt.size(0), dtype=torch.long, device=device)
    logp, _, _ = _predict(model, xt, t_tensor, torch.ones_like(xt), V)
    own = logp.gather(-1, xt.unsqueeze(-1).clamp(max=logp.size(-1) - 1)).squeeze(-1)
    am = answer_mask.float()
    return (-own * am).sum(1) / am.sum(1).clamp(min=1)


def decode_batch_to_solution_strings(tokenizer, xt, src_mask):
    pred = xt.masked_fill(src_mask.to(xt.device), tokenizer.pad_token_id)
    decoded = tokenizer.batch_decode(pred.detach().cpu().numpy().tolist(), skip_special_tokens=True)
    out = []
    for dec in decoded:
        s = dec.replace(" ", "")
        out.append(s[-81:] if len(s) >= 81 else s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", default=DEFAULT_CKPT, help="dir with pytorch_model.bin (default: bundled weights)")
    ap.add_argument("--model_config_dir", default=DEFAULT_CKPT, help="dir with config.json + tokenizer_config.json")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--strategy", choices=["stochastic", "margin", "adaptive", "verifier-bon", "guided", "remdm", "searchdiff"], default="margin")
    ap.add_argument("--remdm_steps", type=int, default=162, help="ReMDM inference-time-scaling steps")
    ap.add_argument("--beam_size", type=int, default=4, help="SearchDiff beam width")
    ap.add_argument("--cand", type=int, default=3, help="SearchDiff candidate digits per cell")
    ap.add_argument("--feas_lambda", type=float, default=3.0, help="SearchDiff constraint-violation penalty")
    ap.add_argument("--value_dir", default=DEFAULT_VALUE, help="dir with value_net.bin for --strategy guided")
    ap.add_argument("--guidance_steps", type=int, default=3)
    ap.add_argument("--guidance_lr", type=float, default=1.0)
    ap.add_argument("--kl_lambda", type=float, default=1.0)
    ap.add_argument("--base_strategy", default="margin",
                     help="base(s) for verifier-bon; single (adaptive/searchdiff/remdm/margin/stochastic/"
                          "guided) OR comma-separated ensemble, e.g. 'searchdiff,adaptive,remdm'")
    ap.add_argument("--vbon_beam_size", type=int, default=4, help="beam for a searchdiff base inside verifier-bon")
    ap.add_argument("--n_samples", type=int, default=8, help="samples for verifier-bon")
    ap.add_argument("--vbon_select", choices=["verifier", "entropy"], default="verifier",
                     help="how to pick a sample: verifier = validity check (uses Sudoku rules); "
                          "entropy = rule-free, by the model's own confidence in its grid")
    ap.add_argument("--diffusion_steps", type=int, default=20)
    ap.add_argument("--reveal_per_step", type=int, default=None,
                     help="cells committed per adaptive step; 1 = full greedy re-planning (paper-style), "
                          "default ties total forward passes to --diffusion_steps for a fair cost comparison")
    ap.add_argument("--cutoff_len", type=int, default=164)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"[device] {device}", flush=True)

    model, tokenizer = load_model_and_tokenizer(args.checkpoint_dir, args.model_config_dir, device)

    value_net = None
    if args.strategy == "guided" or (args.strategy == "verifier-bon" and "guided" in args.base_strategy):
        assert args.value_dir, "--value_dir required when guided is used"
        value_net = load_value_net(args.value_dir, device)
    rows = read_csv_rows(args.csv, limit=args.limit)
    print(f"[data] {len(rows)} puzzles from {args.csv}", flush=True)

    correct = 0
    seen = 0
    start = time.time()
    records = []

    for bstart in range(0, len(rows), args.batch_size):
        batch = rows[bstart : bstart + args.batch_size]
        quizzes = [q for q, _ in batch]
        solutions = [s for _, s in batch]
        x, src_mask = build_batch(tokenizer, quizzes, solutions, args.cutoff_len)

        if args.strategy == "verifier-bon":
            pred_strs = decode_verifier_bon(model, tokenizer, x, src_mask, args.diffusion_steps, device, args.base_strategy, args.n_samples, quizzes, reveal_per_step=args.reveal_per_step, vbon_beam_size=args.vbon_beam_size, value_net=value_net, select=args.vbon_select,
                                            remdm_steps=args.remdm_steps, cand=args.cand,
                                            feas_lambda=args.feas_lambda, guidance_steps=args.guidance_steps,
                                            guidance_lr=args.guidance_lr, kl_lambda=args.kl_lambda)
        else:
            if args.strategy == "adaptive":
                xt = decode_adaptive(model, tokenizer, x, src_mask, args.diffusion_steps, device, reveal_per_step=args.reveal_per_step)
            elif args.strategy == "guided":
                rps = args.reveal_per_step if args.reveal_per_step is not None else 1
                xt = decode_guided(model, tokenizer, x, src_mask, args.diffusion_steps, device, value_net,
                                    reveal_per_step=rps, guidance_steps=args.guidance_steps,
                                    guidance_lr=args.guidance_lr, kl_lambda=args.kl_lambda)
            elif args.strategy == "remdm":
                xt = decode_remdm(model, tokenizer, x, src_mask, args.diffusion_steps, device, num_steps=args.remdm_steps)
            elif args.strategy == "searchdiff":
                xt = decode_searchdiff(model, tokenizer, x, src_mask, args.diffusion_steps, device,
                                        beam_size=args.beam_size, cand=args.cand, feas_lambda=args.feas_lambda)
            else:
                xt = decode_stochastic_or_margin(model, tokenizer, x, src_mask, args.diffusion_steps, args.strategy, device)
            pred_strs = decode_batch_to_solution_strings(tokenizer, xt, src_mask)

        for q, s, pred in zip(quizzes, solutions, pred_strs):
            ok = (pred == s)
            correct += int(ok)
            seen += 1
            records.append({"quiz": q, "solution": s, "predict": pred, "correct": ok})

        elapsed = time.time() - start
        print(f"RUN_PROGRESS seen={seen} correct={correct} acc={correct/seen:.4f} elapsed={elapsed:.1f}s", flush=True)

    runtime = time.time() - start
    metrics = {
        "checkpoint_dir": args.checkpoint_dir,
        "csv": args.csv,
        "strategy": args.strategy,
        "base_strategy": args.base_strategy if args.strategy == "verifier-bon" else None,
        "n_samples": args.n_samples if args.strategy == "verifier-bon" else None,
        "diffusion_steps": args.diffusion_steps,
        "samples": seen,
        "correct": correct,
        "accuracy": correct / seen if seen else 0.0,
        "runtime_sec": runtime,
    }
    print(json.dumps(metrics, indent=2), flush=True)
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w") as f:
            json.dump({"metrics": metrics, "examples": records[:50]}, f, indent=2)


if __name__ == "__main__":
    main()
