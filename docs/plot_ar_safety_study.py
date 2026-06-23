#!/usr/bin/env python
"""Plot the PointGoal safety-frequency study: cost_return and reward vs action_repeat.

Pulls the `safe_goal_ar_study` sweep from W&B, groups by action_repeat, and produces
the GO/NO-GO motivation plot (mean +/- std over seeds) plus a printed table.

Run on a machine with W&B auth (the cluster / your laptop), not the sandbox:

    ./venv/bin/python docs/plot_ar_safety_study.py \
        --entity arnavsukhija-eth-zurich --project actsafe-ct-pointgoal --budget 25

The hypothesis: cost_return should RISE with action_repeat and cross `budget` at high
AR (a long open-loop hold is a blind window). A flat curve falsifies the CT thesis.
"""
import argparse
from collections import defaultdict

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--entity", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--budget", type=float, default=25.0, help="episode cost budget")
    p.add_argument("--cost-key", default="train/cost_return")
    p.add_argument("--reward-key", default="train/objective")
    p.add_argument("--out", default="ar_safety_study.png")
    p.add_argument(
        "--min-step",
        type=float,
        default=0.0,
        help="only count a run as complete if its last logged step >= this "
        "(set to e.g. 4.5e6 to drop truncated runs)",
    )
    args = p.parse_args()

    import wandb

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}")

    # action_repeat -> {"cost": [...], "reward": [...]}
    by_ar: dict[int, dict[str, list]] = defaultdict(lambda: {"cost": [], "reward": []})
    skipped = []
    for run in runs:
        cfg = run.config
        ar = cfg.get("training", {}).get("action_repeat", cfg.get("action_repeat"))
        if ar is None:
            continue
        # final values: prefer summary, fall back to history tail
        cost = run.summary.get(args.cost_key)
        reward = run.summary.get(args.reward_key)
        last_step = run.summary.get("_step", 0) or 0
        if cost is None or reward is None:
            hist = run.history(keys=[args.cost_key, args.reward_key], pandas=True)
            if hist is not None and len(hist):
                cost = hist[args.cost_key].dropna().iloc[-1]
                reward = hist[args.reward_key].dropna().iloc[-1]
        if cost is None or reward is None:
            skipped.append((run.name, "no metrics"))
            continue
        if args.min_step and last_step < args.min_step:
            skipped.append((run.name, f"truncated step={last_step:g}"))
            continue
        by_ar[int(ar)]["cost"].append(float(cost))
        by_ar[int(ar)]["reward"].append(float(reward))

    if not by_ar:
        raise SystemExit("No runs with action_repeat + metrics found. Check keys/project.")

    ars = sorted(by_ar)
    print(f"\n{'AR':>4} {'n':>3} {'cost_return (mean±std)':>26} "
          f"{'reward (mean±std)':>22}  violates_budget?")
    print("-" * 70)
    cost_m, cost_s, rew_m, rew_s = [], [], [], []
    for ar in ars:
        c = np.array(by_ar[ar]["cost"])
        r = np.array(by_ar[ar]["reward"])
        cost_m.append(c.mean()); cost_s.append(c.std())
        rew_m.append(r.mean()); rew_s.append(r.std())
        flag = "  <-- VIOLATES" if c.mean() > args.budget else ""
        print(f"{ar:>4} {len(c):>3} {c.mean():>12.2f} ± {c.std():>8.2f}   "
              f"{r.mean():>10.2f} ± {r.std():>7.2f}{flag}")
    for name, why in skipped:
        print(f"  (skipped {name}: {why})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(ars))
    ax1.errorbar(x, cost_m, yerr=cost_s, marker="o", color="tab:red",
                 label="cost_return", capsize=3)
    ax1.axhline(args.budget, ls="--", color="tab:red", alpha=0.6,
                label=f"budget = {args.budget:g}")
    ax1.set_xlabel("action_repeat  (lower control frequency →)")
    ax1.set_ylabel("episode cost_return", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_xticks(x); ax1.set_xticklabels(ars)

    ax2 = ax1.twinx()
    ax2.errorbar(x, rew_m, yerr=rew_s, marker="s", color="tab:blue",
                 label="reward (objective)", capsize=3)
    ax2.set_ylabel("episode reward", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="best", fontsize=8)
    plt.title("PointGoal: safety cost & reward vs control frequency")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
