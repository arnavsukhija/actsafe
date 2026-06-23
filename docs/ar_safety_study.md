# Safety–Frequency Study on PointGoal (the GO/NO-GO gate)

**One-sentence hypothesis.** On a hazard-rich task, lowering the control frequency
(higher `action_repeat`) increases safety-constraint violations, because a longer
open-loop hold is a blind window in which a developing violation cannot be sensed or
corrected.

This is the foundational experiment for the whole continuous-time direction. It uses
**discrete ActSafe only** (verified faithful to upstream `yardenas/actsafe`) — no
continuous-time machinery, so none of the CT cost-accounting subtleties are in play.
Every outcome is informative (see "Reading the result").

Config: [`actsafe/configs/experiment/safe_goal_ar_study.yaml`](../actsafe/configs/experiment/safe_goal_ar_study.yaml).
Task: `go_to_goal` (PointGoal, dense reward, 9 hazards + 10 vases). Dense reward is
deliberate — it avoids the do-nothing/exploration-starvation confound (a stationary
agent is trivially safe at every frequency, which would fake a flat curve).

## Step 0 — reproduction sanity (do this first, 1 run)

Confirm discrete ActSafe actually learns and stays safe at the base frequency before
spending the full sweep. AR=1, one seed:

```bash
python train_actsafe.py \
  +experiment=safe_goal_ar_study +hardware=4090_rtx hydra/launcher=slurm \
  +wandb.project=actsafe-ct-pointgoal \
  training.action_repeat=1 training.seed=0
```

Pass criterion: `train/objective` climbs clearly positive (agent reaches goals) and
`train/cost_return` is in a sane range (the agent is actually encountering hazards,
i.e. cost is NOT pinned at 0 — if it is, the agent is doing nothing and the sweep is
meaningless). If this looks right, launch the sweep.

## Step 1 — the sweep (12 runs: AR × seed)

```bash
python train_actsafe.py -m \
  +experiment=safe_goal_ar_study +hardware=4090_rtx hydra/launcher=slurm \
  +wandb.project=actsafe-ct-pointgoal \
  training.action_repeat=1,2,4,8 training.seed=0,1,2
```

Only `action_repeat` and `seed` vary — everything else is fixed by the config so the
study isolates frequency. `exploration_steps` and `time_limit` are in **base steps**,
so the exploration phase and the physical episode length are identical at every
`action_repeat` (the comparison is fair by construction — see the config header and
`episodic_async_env.py:193-209`).

## Reading the result

Pull the final-epoch metrics with the plotting script:

```bash
./venv/bin/python docs/plot_ar_safety_study.py \
  --entity arnavsukhija-eth-zurich --project actsafe-ct-pointgoal --budget 25
```

It produces `ar_safety_study.png` (cost_return and reward vs action_repeat, mean±std
over seeds, budget line at 25) and prints a per-AR table.

| Outcome | Meaning | Next step |
|---|---|---|
| **`cost_return` rises with AR, crosses the budget at high AR** (the U-shape) | The thesis is alive: fixed low frequency cannot uphold the guarantee. | This *is* the motivation plot. Proceed to CT (adaptive `dt` finds the bottom of the curve). |
| **Reward collapses at high AR while cost stays low** | The agent buys safety at high AR only by becoming over-conservative (does nothing useful). | Also supports the thesis (fixed AR forces a bad tradeoff); pair with the cost plot. |
| **Flat** — cost stays under budget and reward stable at every AR, incl. AR=8 | Frequency does not stress safety even on hazard-rich PointGoal. | The CT thesis has no empirical support; this is the honest signal to bring to your supervisor and re-scope. |

Bring the plot + this table to the supervisor meeting. A clear result in *any*
direction (including flat) is a real result and a sharpened question — which is
exactly what unblocks the conversation.

## Notes
- Requeue: `slurm.yaml` `max_num_timeout: 100` + `#SBATCH --requeue` + the Trainer's
  pickle resume let a run span multiple 4h (`timeout_min: 240`) windows to reach the
  full ~5M base steps. Do not set `max_num_timeout: 0` — it truncates every run.
- Metric keys: `train/objective`, `train/cost_return`, `agent/lbsgd/safe`.
- Harder follow-up once the signal is established: rerun on `go_to_goal_scarce`
  (sparse, paper-faithful) to show it is not an artifact of dense reward.
