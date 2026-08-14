"""Build two PNGs from outputs/portfolio_analysis.json:
  - portfolio_scatter.png: SAT_PAR2 vs UNSAT_PAR2, point size+color = overall PAR2
  - portfolio_bars.png:    grouped bars per solver — SAT (blue), UNSAT (orange), Overall (gray),
                           solvers sorted ascending by overall PAR2.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/scratch/meru/LLM-SAT")
DATA = json.load(open(REPO / "outputs/portfolio_analysis.json"))
CANDS = DATA["candidates"]

# Sort ascending by overall PAR2 (best first)
CANDS = sorted(CANDS, key=lambda c: c["overall_par2"])
LABELS = [c["label"] for c in CANDS]
SAT = np.array([c["sat_par2"] for c in CANDS])
UNSAT = np.array([c["unsat_par2"] for c in CANDS])
OVERALL = np.array([c["overall_par2"] for c in CANDS])

# ---------- Option A: scatter ----------
fig, ax = plt.subplots(figsize=(10, 8))

# marker size reflects overall PAR2 (worse PAR2 = bigger marker would be confusing,
# so invert: best overall = biggest marker). Range marker areas roughly 80..400.
norm_overall = (OVERALL - OVERALL.min()) / (OVERALL.max() - OVERALL.min() + 1e-9)
sizes = 400 - norm_overall * 320  # best -> 400, worst -> 80

scatter = ax.scatter(
    SAT, UNSAT,
    s=sizes, c=OVERALL, cmap="viridis_r",  # reversed so darker = better
    edgecolor="black", linewidth=0.8, alpha=0.85, zorder=3,
)

# Diagonal SAT==UNSAT reference
lo = float(min(SAT.min(), UNSAT.min())) - 30
hi = float(max(SAT.max(), UNSAT.max())) + 30
ax.plot([lo, hi], [lo, hi], color="grey", linestyle="--", linewidth=1, alpha=0.6, zorder=1)
ax.text(hi - 20, hi - 5, "SAT = UNSAT", color="grey", fontsize=8,
        ha="right", va="bottom", rotation=45, alpha=0.8)

# Annotate each point. Per-label overrides for points that would otherwise
# get clipped by the colorbar / chart edge.
LABEL_LEFT_SIDE = {"compo_R1_B3"}  # rightmost — annotate on the left

for i, lbl in enumerate(LABELS):
    if lbl in LABEL_LEFT_SIDE:
        ax.annotate(
            lbl, (SAT[i], UNSAT[i]),
            xytext=(-7, 4), textcoords="offset points",
            fontsize=8, zorder=4, ha="right",
        )
    else:
        ax.annotate(
            lbl, (SAT[i], UNSAT[i]),
            xytext=(7, 4), textcoords="offset points",
            fontsize=8, zorder=4,
        )

cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Overall PAR2 (lower = better)")

ax.set_xlabel("SAT PAR2 (lower = better)")
ax.set_ylabel("UNSAT PAR2 (lower = better)")
ax.set_title("AE_kissat2025_MAB candidates — SAT vs UNSAT PAR2\n"
             "(marker size also encodes overall PAR2 — bigger = better overall)")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.grid(alpha=0.3, zorder=0)

plt.tight_layout()
out_a = REPO / "outputs/portfolio_scatter.png"
plt.savefig(out_a, dpi=150, bbox_inches="tight")
plt.close()
print(f"Wrote {out_a}")

# ---------- Option B: grouped bars ----------
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(LABELS))
width = 0.27

ax.bar(x - width, SAT,     width, label="SAT PAR2",     color="#1f77b4", edgecolor="black", linewidth=0.4)
ax.bar(x,         UNSAT,   width, label="UNSAT PAR2",   color="#ff7f0e", edgecolor="black", linewidth=0.4)
ax.bar(x + width, OVERALL, width, label="Overall PAR2", color="#7f7f7f", edgecolor="black", linewidth=0.4)

ax.set_xticks(x)
ax.set_xticklabels(LABELS, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("PAR2 (lower = better)")
ax.set_title("AE_kissat2025_MAB candidates — PAR2 by category, sorted by overall PAR2 (best → worst)")
# Pad y-axis upward so the legend sits above all bars/labels and doesn't hide
# the leftmost (B1) overall-PAR2 value label.
ax.set_ylim(0, max(OVERALL) * 1.18)
ax.legend(loc="upper right", framealpha=1.0)
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.set_axisbelow(True)

# Light value labels on top of each Overall bar (readability check)
for i, v in enumerate(OVERALL):
    ax.text(i + width, v + 20, f"{v:.0f}", ha="center", va="bottom", fontsize=7, color="#444")

plt.tight_layout()
out_b = REPO / "outputs/portfolio_bars.png"
plt.savefig(out_b, dpi=150, bbox_inches="tight")
plt.close()
print(f"Wrote {out_b}")
