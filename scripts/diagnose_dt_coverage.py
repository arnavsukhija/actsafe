"""Diagnose dt-coverage of the world-model training data in a TASE checkpoint.

Loads a run's ``state.pkl`` (the pickled Trainer state, on the cluster) and reports:

1. Coverage (Step 1): per hazard-proximity bucket, the distribution of executed
   dt_ratio values in the replay buffer. Proximity proxy = decisions until the
   nearest cost>0 transition within the same episode (0 = the hold itself touched
   a hazard). A collapsed (near-zero std) dt distribution in the near-hazard
   buckets while far buckets are diverse confirms the chicken-and-egg coverage
   gap; uniform-dt warm-up (agent.continuous_time.dt_exploration=uniform) is the
   fix, and re-running this script afterwards verifies it.

2. Two-curve model diagnostic (Step 3): at K near-hazard states (posterior belief
   inferred by filtering the episode prefix), sweep dt over [1, max_repeat]
   with the originally executed force, and report the predicted cost mean and the
   cost ensemble disagreement (std over the dynamics-ensemble arrival states) as
   functions of dt. A dt-flat mean curve after coverage is fixed would indict the
   model, not the data.

Usage (cluster, inside the poetry env):
    python scripts/diagnose_dt_coverage.py /path/to/run_dir/state.pkl
    python scripts/diagnose_dt_coverage.py state.pkl --two-curve 8
"""

import argparse
from collections import OrderedDict

import cloudpickle
import numpy as np

from actsafe.rl.ct_time import dt_ratio_from_pseudo, pseudo_from_dt_ratio


def load_agent(state_path: str):
    with open(state_path, "rb") as f:
        state = cloudpickle.load(f)
    return state["agent"], state.get("step")


def ct_params(ac) -> tuple[float, float]:
    """Repeat-unit hold bounds (k_min, k_max) from the actor-critic.

    Supports both the current layout (k_min/k_max repeat units) and legacy
    checkpoints (tmin/tmax/base_dt physical seconds).
    """
    k_min = getattr(ac, "k_min", None)
    k_max = getattr(ac, "k_max", None)
    if k_min is not None and k_max is not None:
        return float(k_min), float(k_max)
    return float(ac.tmin / ac.base_dt), float(ac.tmax / ac.base_dt)


PROXIMITY_BUCKETS = OrderedDict(
    [
        ("in-hazard (cost>0 hold)", (0, 0)),
        ("1 decision away", (1, 1)),
        ("2-3 decisions away", (2, 3)),
        ("4-8 decisions away", (4, 8)),
        (">8 / never (far)", (9, np.inf)),
    ]
)


def coverage_report(agent) -> None:
    buf = agent.replay_buffer
    k_min, k_max = ct_params(agent.actor_critic)
    max_dt = int(round(k_max))
    n = buf._valid_episodes
    print(f"\n=== Step 1: dt coverage by hazard proximity ({n} episodes) ===")
    print(f"k_min={k_min} k_max={k_max} (max dt_ratio {max_dt})")

    all_dt, all_prox = [], []
    for ep in range(n):
        length = int(buf.lengths[ep])
        if length == 0:
            continue
        cost = buf.cost[ep, :length]
        pseudo = buf.action[ep, :length, -1]
        dt = dt_ratio_from_pseudo(pseudo, k_min, k_max)
        # Decisions to the nearest costful hold (before or after) in the episode.
        hazard_idx = np.where(cost > 0)[0]
        if len(hazard_idx) == 0:
            prox = np.full(length, np.inf)
        else:
            idx = np.arange(length)
            prox = np.abs(idx[:, None] - hazard_idx[None, :]).min(1)
        all_dt.append(dt)
        all_prox.append(prox)
    dt = np.concatenate(all_dt)
    prox = np.concatenate(all_prox)

    edges = [1, 2, 3, 5, 9, max_dt + 1]  # dt_ratio histogram bins
    header = "  ".join(f"[{edges[i]},{edges[i+1]})" for i in range(len(edges) - 1))
    print(f"\n{'bucket':<28} {'count':>8} {'mean':>6} {'std':>6}   share per dt bin: {header}")
    for name, (lo, hi) in PROXIMITY_BUCKETS.items():
        mask = (prox >= lo) & (prox <= hi)
        d = dt[mask]
        if len(d) == 0:
            print(f"{name:<28} {0:>8}")
            continue
        hist, _ = np.histogram(d, bins=edges)
        shares = "  ".join(f"{h / len(d):>8.3f}" for h in hist)
        print(f"{name:<28} {len(d):>8} {d.mean():>6.2f} {d.std():>6.2f}   {shares}")

    # Same-state-multiple-dt check (coarse): among in-hazard-adjacent states, how
    # many distinct dt values occur at all.
    near = dt[prox <= 1]
    print(
        f"\nnear-hazard (<=1 decision) distinct dt values: "
        f"{sorted(np.unique(near).astype(int).tolist())[:20]}"
    )


def two_curve_report(agent, num_states: int, seed: int = 0) -> None:
    import jax
    import jax.numpy as jnp

    from actsafe.actsafe.actsafe import _prepare_features
    from actsafe.rl.trajectory import TrajectoryData

    buf = agent.replay_buffer
    model = agent.model
    k_min, k_max = ct_params(agent.actor_critic)
    max_dt = int(round(k_max))
    rs = np.random.RandomState(seed)
    key = jax.random.PRNGKey(seed)

    # Collect (episode, t) indices of costful holds with enough filter prefix.
    candidates = []
    for ep in range(buf._valid_episodes):
        length = int(buf.lengths[ep])
        idx = np.where(buf.cost[ep, :length] > 0)[0]
        candidates.extend((ep, t) for t in idx if t >= 5)
    if not candidates:
        print("\nNo costful holds in buffer; cannot run two-curve diagnostic.")
        return
    picks = [candidates[i] for i in rs.choice(len(candidates), size=min(num_states, len(candidates)), replace=False)]

    print(f"\n=== Step 3: two-curve dt->cost diagnostic ({len(picks)} near-hazard states) ===")
    dt_grid = np.unique(np.round(np.linspace(1, max_dt, min(max_dt, 16)))).astype(int)
    mean_curves, std_curves = [], []
    for ep, t in picks:
        length = int(buf.lengths[ep])
        # Filter the episode prefix [0, t) to get the belief BEFORE hold t.
        batch = TrajectoryData(
            buf.observation[None, ep, :t],
            buf.observation[None, ep, 1 : t + 1],
            buf.action[None, ep, :t],
            buf.reward[None, ep, :t],
            buf.cost[None, ep, :t],
        )
        features, actions = _prepare_features(batch)
        key, k1, k2 = jax.random.split(key, 3)
        result = model(
            jax.tree_map(lambda x: x[0], features), actions[0], k1
        )
        state = jax.tree_map(lambda x: x[-1], result.state)
        force = buf.action[ep, t, :-1]
        means, stds = [], []
        for dtv in dt_grid:
            pseudo = pseudo_from_dt_ratio(float(dtv), k_min, k_max)
            action = jnp.asarray(np.concatenate([force, [pseudo]], 0), dtype=jnp.float32)
            ensemble_states, _ = model.cell.predict(state, action, k2)
            flat = ensemble_states.flatten()  # [E, state_dim]
            if model.continuous_time:
                tiled = jnp.broadcast_to(action[None], (flat.shape[0], action.shape[0]))
                dec_in = jnp.concatenate([flat, tiled.astype(flat.dtype)], -1)
            else:
                dec_in = flat
            out = jax.vmap(model.reward_cost_decoder)(dec_in)
            cost = np.asarray(out[..., -1])  # [E]
            means.append(cost.mean())
            stds.append(cost.std())
        mean_curves.append(means)
        std_curves.append(stds)

    mean_curve = np.mean(mean_curves, 0)
    std_curve = np.mean(std_curves, 0)
    print(f"{'dt_ratio':>8} {'pred cost mean':>15} {'ensemble std':>13}")
    for i, dtv in enumerate(dt_grid):
        print(f"{dtv:>8} {mean_curve[i]:>15.4f} {std_curve[i]:>13.4f}")
    slope = np.polyfit(dt_grid, mean_curve, 1)[0]
    print(f"\nlinear dt->cost slope of mean curve: {slope:.5f} per dt step")
    print("(positive & non-trivial => model has learned dt-sensitive cost; ~0 => dt-flat)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state_pkl", help="path to a run's state.pkl")
    parser.add_argument("--two-curve", type=int, default=0, metavar="K",
                        help="also run the two-curve model diagnostic at K near-hazard states")
    args = parser.parse_args()

    agent, step = load_agent(args.state_pkl)
    print(f"Loaded agent at step {step}; buffer episodes: {agent.replay_buffer._valid_episodes}")
    coverage_report(agent)
    if args.two_curve > 0:
        two_curve_report(agent, args.two_curve)


if __name__ == "__main__":
    main()
