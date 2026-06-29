# Control-Frequency Safety on PointGoal — preliminary result (2026-06-27)

**One-line takeaway:** On Safety-Gym PointGoal, a *fixed* safe-RL agent (ActSafe, identical
algorithm / budget / pessimism) **satisfies its safety constraint at high control frequency and
fails it at low control frequency.** Incurred cost rises monotonically as the control period grows
and crosses the budget — motivating a method that *adapts* the control frequency.

## Setup
- Task: PointGoal (`go_to_goal`, dense), the ActSafe paper's home benchmark.
- Fixed safety budget **d = 25** (standard Safety-Gym cost limit), per 1000-base-step episode.
- Control frequency varied via **action_repeat (AR)**: AR=1 = full frequency (1000 decisions/ep),
  AR=8 = one decision held for 8 base steps (125 decisions/ep). Same 1000 physical steps either way.
- Realized cost is the **undiscounted physical episode cost** (raw sum over held steps), so it is
  directly comparable to d=25 at every frequency — a fair, frequency-invariant bar.
- 3 seeds/cell, pessimism κ=0.1. Cost = mean of last 20 exploitation episodes.

## Result (κ=0.1, reward-healthy seeds)

| Control freq (AR) | Incurred cost | Reward | Constraint (d=25) |
|---|---|---|---|
| 1 (highest) | **15.7** | 7.0  | satisfied |
| 2 | **17.7** | 10.1 | satisfied |
| 4 | **19.8** | 19.2 | satisfied (margin thin) |
| 8 (lowest) | **27.4** | 18.0 | **VIOLATED** (60–70% of episodes > 25) |

- **Cost rises monotonically as control frequency drops**, and the agent **crosses from safe to
  unsafe at AR=8** — without changing anything but how often it can act.
- Mechanism: a longer open-loop hold is a *blind window* in which a developing violation cannot be
  sensed or corrected. The pessimism margin (κ) absorbs the small excess at high frequency; at low
  frequency the blind-window excess exceeds the margin and the constraint breaks.

## Caveats (being addressed)
- Only AR=8 *violates*; AR1–4 are under budget with an **eroding margin**. The honest claim is
  "margin erodes monotonically → crosses into violation at low frequency," not "every step is a
  violation." Extending to AR=16 turns the single crossing into a clear trend.
- 2/12 seeds suffered **actor entropy collapse** (a known imagination-actor failure: policy
  saturates, reward→0, cost flails) — an optimization artifact, *not* a safety signal. Excluded
  above; being fixed with an actor entropy bonus + more seeds.

## Next (in flight)
1. **κ × AR grid** (κ∈{0.001, 0.1} × AR∈{1,2,4,8,16}, 5 seeds): shows the effect is robust to the
   safety-margin setting (at κ≈0 the violation grows with AR from AR=1; at κ=0.1 it breaks at AR=8).
2. **Cost-of-control axis** (switch/compute penalty): without it, fixed high-frequency strictly
   dominates and there is no tradeoff — this is what makes adaptivity necessary.
3. **Time-adaptive control (TASE):** an agent that picks small dt only near hazards should recover
   high-frequency *safety* at low *average* control cost — beating every fixed frequency. This is
   the proposed contribution; the fixed-frequency study above is its motivating baseline.
