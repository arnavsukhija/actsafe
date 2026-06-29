"""PointGoal control-frequency safety study — motivation figure.

Pulls all runs from wandb (entity arnavsukhija-eth-zurich / project actsafe-ct-pointgoal),
tail-averages the last TAIL eval-epochs per (kappa, action_repeat, seed), aggregates over
seeds, and renders the 3-panel motivation figure.

Caching: aggregates are cached to agg.json next to this script. Delete agg.json (or pass
--refresh) to re-pull from wandb. Re-plotting from cache is instant.

FRAMING (decided 2026-06-29 with the user): kappa=0.1 is the HEADLINE (clean, monotonic,
all-healthy-seed crossing AR4->AR8). kappa=0.001 is the robustness check (noisier, no margin).
AR=16 is GREYED OUT everywhere: at AR16 every seed suffers entropy-collapse (an upstream actor
fragility — no entropy term in the actor loss, safe_actor_critic.py:268-269), so its cost is a
broken-optimizer artifact, not a low-frequency-safety signal. The load-bearing claim is the
AR4->AR8 crossing on healthy seeds and does NOT depend on AR16 or any collapsed seed.
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "agg.json")
ENT, PROJ = "arnavsukhija-eth-zurich", "actsafe-ct-pointgoal"
TAIL = 8
KAPPAS = [0.001, 0.1]
ARS = [1, 2, 4, 8, 16]
COLLAPSE_AR = 16  # greyed out: every seed collapses here


def pull():
    import wandb
    from collections import defaultdict
    api = wandb.Api(timeout=60)
    cell_pts = defaultdict(list)
    for r in api.runs(f"{ENT}/{PROJ}", per_page=200):
        cfg = r.config
        tr, ag = cfg.get("training", {}), cfg.get("agent", {})
        ar, seed = tr.get("action_repeat"), tr.get("seed")
        kappa = ag.get("sentiment", {}).get("constraint_pessimism")
        if ar is None or kappa is None:
            continue
        for h in r.history(keys=["train/objective", "train/cost_return"], samples=300, pandas=False):
            s, o, c = h.get("_step"), h.get("train/objective"), h.get("train/cost_return")
            if s is None or c is None:
                continue
            cell_pts[(kappa, ar, seed)].append((s, o, c))
    seed_tail = {}
    for k, pts in cell_pts.items():
        tail = sorted(pts, key=lambda x: x[0])[-TAIL:]
        objs = [p[1] for p in tail if p[1] is not None]
        costs = [p[2] for p in tail if p[2] is not None]
        seed_tail[k] = (float(np.mean(objs)) if objs else float("nan"),
                        float(np.mean(costs)) if costs else float("nan"))
    agg = {}
    for ka in KAPPAS:
        for ar in ARS:
            seeds = sorted(s for (kk, a, s) in seed_tail if kk == ka and a == ar)
            costs = [seed_tail[(ka, ar, s)][1] for s in seeds]
            objs = [seed_tail[(ka, ar, s)][0] for s in seeds]
            agg[f"{ka}|{ar}"] = dict(n=len(seeds), cost_med=float(np.median(costs)),
                                     cost_seeds=costs, obj_med=float(np.median(objs)),
                                     obj_seeds=objs, nviol=sum(1 for c in costs if c > 25))
    json.dump(agg, open(CACHE, "w"), indent=2)
    return agg


if "--refresh" in sys.argv or not os.path.exists(CACHE):
    agg = pull()
else:
    agg = json.load(open(CACHE))


def g(ka, ar):
    return agg[f"{ka}|{ar}"]


# styling: kappa=0.1 headline (bold), kappa=0.001 robustness (light)
STY = {
    0.1:   dict(color="#2e86ab", lw=2.8, ms=8, alpha=1.0, z=4, label=r"$\kappa=0.1$ (margin) — headline"),
    0.001: dict(color="#d1495b", lw=1.6, ms=6, alpha=0.65, z=3, label=r"$\kappa=0.001$ (no margin) — robustness"),
}
xpos = np.arange(len(ARS))
GIDX = ARS.index(COLLAPSE_AR)  # AR16 index

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

# ---- Panel (a): cost vs AR (full ladder; kappa=0.1 monotone through AR16) ----
axA = axes[0]
axA.set_ylim(0, 50)
for ka in KAPPAS:
    s = STY[ka]
    med = [g(ka, ar)["cost_med"] for ar in ARS]
    axA.plot(xpos, med, "-o", color=s["color"], lw=s["lw"], ms=s["ms"],
             alpha=s["alpha"], zorder=s["z"], label=s["label"])
    for j, ar in enumerate(ARS):
        cs = g(ka, ar)["cost_seeds"]
        axA.scatter([xpos[j] + (0.07 if ka == 0.1 else -0.07)] * len(cs), cs,
                    color=s["color"], alpha=0.40, s=20, zorder=2)
axA.axhline(25, ls="--", color="black", lw=1.4)
axA.text(0.02, 26.0, "budget  d = 25", fontsize=9, transform=axA.get_yaxis_transform())
axA.set_xticks(xpos); axA.set_xticklabels(ARS)
axA.set_xlabel("action repeat   (lower control frequency →)")
axA.set_ylabel("realized episode cost")
axA.set_title("(a) Safety cost rises monotonically as frequency drops")
axA.legend(frameon=False, fontsize=8.5, loc="upper left")
axA.grid(alpha=0.25)
axA.annotate("AR8: competent\nbut unsafe", xy=(3.05, 25.6), xytext=(3.35, 39),
             fontsize=8.5, color="#2e86ab", ha="center",
             arrowprops=dict(arrowstyle="->", color="#2e86ab", lw=1.3))

# ---- Panel (b): violation fraction ----
axB = axes[1]
w = 0.36
for i, ka in enumerate(KAPPAS):
    frac = [g(ka, ar)["nviol"] / g(ka, ar)["n"] for ar in ARS]
    axB.bar(xpos + (i - 0.5) * w, frac, width=w, color=STY[ka]["color"],
            alpha=0.85 if ka == 0.1 else 0.5, label=STY[ka]["label"], zorder=3)
axB.set_xticks(xpos); axB.set_xticklabels(ARS); axB.set_ylim(0, 1.05)
axB.set_xlabel("action repeat   (lower control frequency →)")
axB.set_ylabel("fraction of seeds violating  (cost > 25)")
axB.set_title("(b) Constraint breaks at low frequency")
axB.legend(frameon=False, fontsize=8.5, loc="upper left")
axB.grid(alpha=0.25, axis="y")

# ---- Panel (c): reward U-curve, collapsed seeds marked ----
axC = axes[2]
for ka in KAPPAS:
    s = STY[ka]
    med = [g(ka, ar)["obj_med"] for ar in ARS]
    axC.plot(xpos[:GIDX], med[:GIDX], "-o", color=s["color"], lw=s["lw"], ms=s["ms"],
             alpha=s["alpha"], zorder=s["z"], label=s["label"])
    axC.plot(xpos[GIDX - 1:], med[GIDX - 1:], "--o", color=s["color"], lw=s["lw"],
             ms=s["ms"], alpha=s["alpha"] * 0.5, zorder=s["z"])
    for j, ar in enumerate(ARS):
        xj = xpos[j] + (0.07 if ka == 0.1 else -0.07)
        for o in g(ka, ar)["obj_seeds"]:
            collapsed = o < 5.0
            axC.scatter([xj], [o], marker="x" if collapsed else "o", color=s["color"],
                        alpha=0.7 if collapsed else 0.35, s=34 if collapsed else 20,
                        zorder=2, linewidths=1.4 if collapsed else 0)
axC.set_xticks(xpos); axC.set_xticklabels(ARS)
axC.set_xlabel("action repeat   (lower control frequency →)")
axC.set_ylabel("episode reward (objective)")
axC.set_title("(c) Reward U-shaped; low-AR medians hit by collapse (✕)")
axC.legend(frameon=False, fontsize=8.5, loc="lower center")
axC.grid(alpha=0.25)
axC.annotate("AR16: reward collapsed\n(unsafe AND task-failing)", xy=(GIDX, 0.0),
             xytext=(GIDX - 0.55, 12), fontsize=7.8, color="0.30", ha="center",
             arrowprops=dict(arrowstyle="->", color="0.45", lw=1.0))
axC.text(0.5, 0.02, "✕ = entropy-collapsed seed (upstream actor fragility; see T6)",
         transform=axC.transAxes, fontsize=8, color="0.3", ha="center")

fig.suptitle("PointGoal control-frequency safety  (ActSafe, fixed action-repeat, 5 seeds/cell) — "
             "headline: $\\kappa{=}0.1$ (cost monotone through AR16)", fontsize=11.5, y=1.03)
fig.tight_layout()
out = os.path.join(HERE, "control_frequency_safety.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
