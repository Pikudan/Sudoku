#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "results_sweep" / "n2000"
MAIN = ROOT / "results_sweep" / "final"


def load(d):
    out = {}
    for f in sorted(d.glob("*.json")):
        m = json.load(open(f))["metrics"]
        out[f.stem] = m["accuracy"] * 100
    return out


grid = load(GRID)
main = load(MAIN)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

bases = ["stochastic", "margin", "adaptive", "remdm", "searchdiff"]
ns = [4, 8, 16, 32]
for b in bases:
    xs, ys = [], []
    for n in ns:
        key = f"vbon_{b}_n{n}"
        if key in grid:
            xs.append(n)
            ys.append(grid[key])
    if xs:
        ax1.plot(xs, ys, marker="o", label=f"base={b}")
ax1.set_xscale("log", base=2)
ax1.set_xticks(ns)
ax1.set_xticklabels(ns)
ax1.set_xlabel("verifier-BoN samples (N)")
ax1.set_ylabel("hard-split accuracy (%)")
ax1.set_title("verifier-BoN scales with samples & base\n(GPU, N=2000 puzzles)")
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)
ax1.set_ylim(0, 100)

order = sorted(main.items(), key=lambda kv: kv[1])
names = [k for k, _ in order]
vals = [v for _, v in order]
ax2.barh(names, vals)
for i, v in enumerate(vals):
    ax2.text(v + 1, i, f"{v:.1f}", va="center", fontsize=8)
ax2.set_xlabel("hard-split accuracy (%)")
ax2.set_title("All methods, best settings, N=10000\nbaseline 4.9% -> best 97.9%")
ax2.set_xlim(0, 100)

fig.tight_layout()
out = ROOT / "results_sweep" / "scaling_curves.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved {out}")
