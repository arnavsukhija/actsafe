"""Supervisor-update figure: one clean message — cost rises with action repeat.

Single panel. kappa=0.1 (the clean, with-margin setting). Median cost over 5 seeds + IQR band,
the d=25 budget line, and the safe/unsafe crossing. Reads agg.json (produced by
control_frequency_safety.py). That's it — no reward panel, no per-seed annotations.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
agg = json.load(open(os.path.join(HERE, "agg.json")))
ARS = [1, 2, 4, 8, 16]
KA = 0.1

cost_seeds = [agg[f"{KA}|{ar}"]["cost_seeds"] for ar in ARS]
med = np.array([np.median(c) for c in cost_seeds])
q1 = np.array([np.percentile(c, 25) for c in cost_seeds])
q3 = np.array([np.percentile(c, 75) for c in cost_seeds])
x = np.arange(len(ARS))

fig, ax = plt.subplots(figsize=(7.2, 4.8))
BLUE = "#2e86ab"
ax.fill_between(x, q1, q3, color=BLUE, alpha=0.18, lw=0)
ax.plot(x, med, "-o", color=BLUE, lw=3, ms=9, zorder=3)

ax.axhline(25, ls="--", color="black", lw=1.6)
ax.text(0.03, 25.8, "safety budget", fontsize=10, transform=ax.get_yaxis_transform())
ax.axhspan(25, ax.get_ylim()[1], color="#d1495b", alpha=0.06, zorder=0)

ax.set_xticks(x); ax.set_xticklabels(ARS)
ax.set_xlabel("action repeat   (lower control frequency  →)", fontsize=11)
ax.set_ylabel("realized episode cost", fontsize=11)
ax.set_title("Lower control frequency → higher safety cost\n(Safety-Gym PointGoal, ActSafe, 5 seeds)",
             fontsize=12)
ax.set_ylim(8, 32)
ax.grid(alpha=0.25)
fig.tight_layout()
out = os.path.join(HERE, "cost_vs_frequency.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
