# Architectural Review — Flow-Based ActSafe-CT (TASE) from SMDP First Principles

_Reviewed 2026-07-11 against `wip/tase-pointgoal` (working tree). Supersedes the 2026-07-07
review. Files audited: `actsafe.py`, `ct_time.py`, `world_model.py`, `rssm.py`,
`make_actor_critic.py`, `safe_actor_critic.py`, `replay_buffer.py`, `sentiment.py`, `lbsgd.py`
(+ `actor_critic.py`, `exploration.py`, `opax.py`, `opax_bridge.py`, `wrappers.py` where the
audit required them). Symptoms under investigation: (1) the safety critic believes the agent is
safe while realized cost violates the budget; (2) OPAX explores at dt=1._

**Verified against wandb (`actsafe-ct-pointgoal`, this review):**

- `e6cttmuk` (switch_cost=0.01, max_repeat=16, steps 1.3M–2.3M): `agent/safety_critic/constraint`
  > 0 (critic says safe) at **100%** of logged steps, mean slack 0.47–0.65 of B=2.5, while
  realized `train/cost_return` = 33–38 vs budget 25 at **100%** of steps. `wee7kzun`:
  safe-while-violating at 57% of steps. Symptom 1 confirmed hard.
- `czo8evls`: `train/ct/mean_dt_ratio` ≈ 11.7 for steps 0–200k, then dt=1 for 200k–500k. The
  break is exactly `offline_steps = 200000`. The "healthy spread" phase was the **offline
  `UniformExploration` policy**, which samples pseudo-time in [0,1) → k ∈ [8.5,16] (predicted
  mean ≈ 12.2 ≈ observed 11.7). **OPAX chose dt=1 from the first step it controlled** — it
  never "converged" there. This corrects the handoff narrative ("collapses after 250k").

---

## 1. The SMDP / zero-order-hold foundation is conceptually valid — proof

### 1.1 The Bellman decomposition is exact

The agent solves a semi-MDP whose options are zero-order holds: at decision epoch $j$ in state
$x$, choose $(u, k)$ with $k \in \{k_{\min},\dots,k_{\max}\}$ base steps. The wrapper
(`SwitchCostWrapper.step`, `wrappers.py:181-252`) returns the **within-hold discounted**
aggregates

$$\bar r(x,u,k) = \sum_{i=0}^{k-1} \gamma^i\, r(x_i, u), \qquad
  \bar c(x,u,k) = \sum_{i=0}^{k-1} \gamma_c^i\, c(x_i, u),$$

and the agent applies the **variable continuation discount** $\gamma^k$
(`safe_actor_critic.evaluate_actor`, `safe_actor_critic.py:227-249`). The SMDP Bellman equation

$$Q(x, u, k) = \bar r(x,u,k) + \gamma^{k}\, \mathbb{E}\big[V(x_{t+k})\big]$$

telescopes to the base MDP's per-step discounted return for **any** hold schedule
$(k_0, k_1, \dots)$:

$$\sum_j \gamma^{\sum_{l<j} k_l}\,\bar r_j
  \;=\; \sum_{\text{base steps } i} \gamma^i\, r_i .$$

This chunk-invariance is exact (including horizon-clipped final holds) and is regression-tested
in `tests/test_budget_invariance.py`. The identical identity for the cost channel makes the
safety budget $B = d / (T(1-\gamma_c))$ (`make_actor_critic.compute_episode_safety_budget`)
dt-schedule-independent — holding longer cannot buy allowance. **The discounted CMDP
formulation, with variable discounting as the forcing mechanism, is the standard and
theoretically sound construction. Nothing at this level is broken.**

### 1.2 Time augmentation with analytic propagation is valid (doubt #1)

Under episode truncation at $T$ base steps, the process is non-stationary; augmenting the state
to $(s, t)$ restores the Markov property (full-MDP time). Since $t$'s dynamics are **known and
deterministic** ($t' = t + k$), propagating it analytically instead of through the encoder is
exact dynamic programming on the augmented MDP — asking the RSSM to *learn* the clock could
only add estimation error. The implementation is consistent everywhere it must be:

- **Acting**: the clock channel is read back as a scalar and appended to the flat latent
  (`actsafe.py:42-43`).
- **Training**: posterior states get the arrival-time fraction from the same (batch, step)
  positions (`actsafe.py:318-321`).
- **Imagination**: time advances by the executed hold, `time' = min(time + k/T, 1)`
  (`world_model.py:286-289`), with $k$ from the same `ct_time` quantizer the wrapper executes.

**On the units question** ($t' = t + k/T$ vs $t' = t + k$): these are the same recurrence. The
elapsed time *is* a discrete count of executed base steps; the code stores it normalized by the
horizon for two implementation reasons only: (a) the clock must survive the replay buffer's
uint8 pixel storage, hence the $255 \cdot t/T$ channel encoding (`wrappers.py:170-179`); (b) the
policy/critic networks get a bounded $[0,1]$ feature instead of a $[0,1000]$ one. The discrete
count is recoverable as $t \cdot T$ at any point. No continuous-time semantics are smuggled in.

Requirement for validity: the base dynamics, reward, and hazards must be time-homogeneous
(true for PointGoal), because only the policy/critics — not the RSSM prior or the reward/cost
decoder — condition on $t$.

### 1.3 Gradients for the hold length DO flow (doubt #2) — three paths, audited

Imagination (`world_model.sample`) rolls the reparametrized actor through the RSSM. The actor's
output $(u, p)$ — $p$ the pseudo-time — is differentiable w.r.t. actor parameters (distrax
tanh-Normal, reparametrized sampling). From there, three gradient paths reach the dt head:

1. **Dynamics path**: the RSSM prior consumes the raw action *including the continuous $p$*
   (`rssm.py:75-85` via `cell.predict`), so the imagined arrival latent — and everything decoded
   from it downstream — is differentiable in $p$. Note the model is trained on quantized
   executions but queried on continuous $p$; the network's smoothing of the ground-truth
   staircase is what provides usable gradient here.
2. **Decoder path**: the action-conditioned reward/cost decoder sees $(z_{\text{arrival}}, u, p)$
   (`world_model.py:319-329`), giving a direct learned $\partial \hat c / \partial p$.
3. **Discount path**: $\gamma^{k}$ with $k = \mathrm{STE}(p)$
   (`safe_actor_critic.py:234-238`). The STE (`ct_time.ste_dt_ratio`) is correctly constructed:
   forward = the executed integer hold (nearest-int, ties up, clipped — bit-identical to
   `SwitchCostWrapper`'s quantization), backward = the affine map's constant slope
   $(k_{\max}-k_{\min})/2$. The internal `stop_gradient` is the estimator mechanism itself
   (the quantizer has zero derivative a.e.), not a gradient block. $\gamma^k$ is then smoothly
   differentiable in $k$.

LBSGD (`lbsgd.py:92-126`) computes the Jacobian of $[\,\text{loss} - \eta\log(\text{constraint}),
\,-\text{constraint}\,]$ over **all** actor parameters — the dt-head columns are present in both
the objective gradient $g$ and the constraint gradient $\nabla f_1$. **Gradient existence is not
the problem.** What the gradients *say* is (§2–§3).

### 1.4 Tanh on the dt head — enforced, accounted for, two real risks

The actor squashes the **full** action vector, dt dim included, through
`Transformed(Normal, Tanh())` (`actor_critic.py:56-57`), so policy pseudo-time lies in
$(-1,1)$. The nearest-integer mapping makes $k_{\max}$ reachable regardless (any
$p \ge 1 - 1/(k_{\max}-k_{\min})$, e.g. $p \ge 0.933$ at $k_{\max}=16$, executes $k_{\max}$).
Two genuine tanh risks to monitor:

- **Saturation kills the dt gradient where it matters most.** $\partial a/\partial(\text{pre-tanh})
  = 1 - a^2 \to 0$ at the extremes. If the switch cost parks the policy near $k_{\max}$, the dt
  head stops receiving gradient — including the constraint gradient that should pull it back
  near hazards. Watch `frac_dt_max` together with a (to-be-added) `dt_head_grad_norm`;
  the `dt_init_stddev` knob exists as mitigation.
- **Initialization bimodality.** At init, $\mu \approx 0$, $\sigma = 5$; $N(0,5)$ through tanh
  puts most mass at $\pm 1$, so the *initial* policy dt distribution is approximately bimodal
  $\{k_{\min}, k_{\max}\}$ — not spread. This shapes the earliest policy-generated data before
  any learning has happened.

### 1.5 The flow-based trade-off, stated honestly

Predicting the *end of the flow* — arrival latent plus whole-hold aggregates
$\bar r(s,u,k), \bar c(s,u,k)$ — is SMDP-consistent and is the method's identity. Its price
relative to a per-base-step model is fundamental: **a per-step model generalizes across $k$ by
composition; a flow model must be shown data at every $k$ region it will be queried on.** There
is no mechanism by which $\bar c(s,u,16)$ is inferred from $\bar c(s,u,1)$ data — the map is a
free function of $k$. This is not a defect; it is a data requirement, and it is the theoretical
reason the coverage diagnosis in §3 is load-bearing.

### 1.6 Verified-sound checklist (audited this review — stop re-litigating)

- **λ-return indexing** (`evaluate_actor` + `compute_lambda_values`): the cost of hold $j$ is
  decoded at arrival$_j$ and bootstrapped with $V(\text{arrival}_{j+1})$. At its fixed point the
  critic is a one-hold-delayed value function ($V(\text{arrival}_j) = $ cost-to-go from
  start$_j$), and the λ-values used in the constraint are correct cost-to-go's from the rollout
  start. Self-consistent; inherited from upstream; **not** the bug. (Residual: the critic input
  carries $t_{j+1}$ for a target anchored at $t_j$ — a one-hold time smear, ≤ 16/1000.)
- **Nearest-integer mapping** (`ct_time.py`): single source of truth for wrapper execution, STE
  forward, and coverage diagnostics; $k_{\max}$ reachable. Correct.
- **Buffer plumbing** (`replay_buffer.py`): `cost` (within-hold discounted), `cost_realized`
  (raw), `exposure` ($k$) stored per decision; variable-length episodes handled via `lengths`.
  Correct. Caveat retained from the previous review: `_ensure_aux_arrays`
  (`replay_buffer.py:67-76`) **silently backfills** resumed pre-exposure pickles with
  `exposure≡1` — per-k diagnostics on such a run are fiction; use fresh runs (see Part B, guard).
- **LBSGD mechanics** (`lbsgd.py`): direction selection and fallback are faithful to the
  reference. Two notes: (a) $\eta$ decays by `eta_rate` in **both** the happy and fallback
  branches, so the log-barrier weight monotonically → 0 over a long run; (b) with the constraint
  reading positive 100% of the time (observed), the fallback never triggers and the safety
  machinery is effectively inert. **LBSGD is a downstream victim of the constraint value it is
  fed, not a cause.**
- **Pessimism (`sentiment.py`)**: the default `latent` UCB adds a *unitless* epistemic ratio
  scaled by α=0.1 to a constraint of scale 2.5. Against any systematic model error it is inert
  (the discrete audit measured a gap ≈ 28× the ensemble std). Keep κ as an honest epistemic
  margin; do not tune it to paper over model error.

---

## 2. The k-slope mechanism (doubt #3: "the only constraint gradient for dt is the discounting")

The true SMDP action-value of extending a hold decomposes as

$$Q_c(s,u,k) = \bar c(s,u,k) + \gamma_c^{\,k}\, V_c(s'_k), \qquad
\frac{\partial \bar c}{\partial k} = \gamma_c^{\,k}\,\rho_k \ \ge 0, \qquad
\frac{\partial (\gamma_c^{\,k} V_c)}{\partial k} = \gamma_c^{\,k} \ln(\gamma_c)\, V_c \ < 0,$$

where $\rho_k$ is the hazard rate at the hold's leading edge. Net:

$$\frac{\partial Q_c}{\partial k} \;=\; \gamma_c^{\,k}\Big(\rho_k - (1-\gamma_c)\,V_c\Big)
\quad (\text{using } -\ln\gamma_c \approx 1-\gamma_c).$$

**Interpretation — and the answer to "are longer holds perceived safer?":** the true gradient
has *no universal direction*. Extending a hold is genuinely safer (in discounted terms) when the
local hazard rate is below the discount-normalized average future rate, and genuinely costlier
near hazards where $\rho_k$ is large. The negative discount gradient is correct physics; the
variable discount is the right forcing mechanism and stays.

**The asymmetry in the code:** the negative term is *exact* — $\gamma_c^k$ is computed
analytically through the STE. The positive term is the *learned* decoder slope
$\partial \hat c/\partial p$, a free MLP direction with no structural tie to $\rho_k$. Wherever
that learned slope is flat — and under §3's diagnosis it is flat at under-covered $k$ simply
because **no data exists there** — the perceived gradient degenerates to
$\partial \hat Q_c/\partial k < 0$ everywhere: *holding longer always looks safer*. LBSGD then
rationally treats "increase $k$" as a constraint-satisfying direction. The agent hacks the
critic, not the budget.

Two diagnostics separate mechanism from cause (both in Part B):

- **Mechanism**: perceived-vs-realized k-slope — finite-difference $\hat c(s,u,k)$ and
  $\hat Q_c$ across $k$ at fixed $(s,u)$, compared with the buffer's realized per-k cost slope.
- **Cause** (no-data vs estimator): per-k calibration gaps **jointly with per-k sample counts**,
  read *after* the coverage fix. A clean gap on an empty bucket proves nothing
  (`cost_calibration` returns 0.0 for empty buckets, `world_model.py:531`).

---

## 3. Central diagnosis: phase-wise dt-coverage mismatch

The failure is a pipeline property, not a single module bug. The training run passes through
three data regimes, none of which covers the region the final policy uses:

| Phase (base steps) | Controller | dt executed | State coverage |
|---|---|---|---|
| 0 – 200k (`offline_steps`) | `UniformExploration` | **k ∈ [8.5, 16] only** — samples pseudo-time in **[0,1)** (`exploration.py:102`), the upper half of the affine map; forces also only [0,1) | random-walk states |
| 200k – 500k (`exploration_steps`) | OPAX | **k = 1 only** (rational optimum, §4) | curiosity-driven |
| 500k+ | task policy | k pushed up by `switch_cost` — observed mean_dt ≈ 2 (`e6cttmuk`) to ≈ 8–13 (`wee7kzun`, `czo8evls`) | goal-directed, near hazards |

The buffer's dt coverage is therefore **bimodal** — upper-half k at random states, k=1 at
curiosity states — and the task-phase operating points (k ≈ 2–13, at goal-directed states near
hazards) fall squarely in the gaps. By §1.5, the flow model has no mechanism to fill those gaps
by generalization: $\hat f(s,u,k)$ and $\hat c(s,u,k)$ at the queried $(s,u,k)$ are
extrapolations. The safety critic is trained *purely on imagination* over those extrapolations
(`update_safe_actor_critic`), and the constraint is evaluated on the same imagination — so its
"safe" reading inherits whatever the extrapolated flow says.

**Why the failure is specifically optimistic:** the cost channel's targets are ≈ 90% exact
zeros. An MLP queried off-distribution on such a channel defaults toward its output prior — ≈ 0,
i.e. "no cost" — and nothing constrains predictions to be non-negative, so hedging on the zero
mass can even go slightly negative and *add* slack to the constraint. Optimism is the default
failure direction of an under-covered cost head, no estimator-bias assumption needed.

**Status: hypothesis with a decisive test.** This diagnosis is endorsed by the evidence above
but is falsifiable, and the review deliberately front-loads its test: fix coverage (Part A/B),
then read the per-k calibration gaps *at buckets with real counts*. If gaps close, coverage was
the cause and plain Gaussian/MSE suffices (the theoretically expected outcome). If material
optimism persists at well-covered k, the estimator-level question reopens (§6, escalation
clause) — that is what refutation would look like. Secondary contributors that persist either
way and are covered by the diagnostics battery: imagination compounding error (the calibration
probe is teacher-forced one-step, `actsafe.py:347-356`, and never exercises multi-step drift),
and bootstrap self-reference (at small dt, a 15-decision rollout spans ~15–45 base steps; most
of each λ-value is the safety critic's own bootstrap, so optimism is slow to self-correct).

---

## 4. OPAX at dt=1: rational optimum of the stated objective, not degeneration

For a stationary hold length $k$, the exploration return is approximately

$$V_{\text{explore}}(k) \;\approx\; \frac{b(k)}{1-\gamma^{k}} \;\approx\; \frac{b(k)}{k\,(1-\gamma)},$$

where $b(k)$ is the per-decision bonus. dt=1 loses only if $b(k)$ grows at least *linearly*
in $k$. It cannot: the bonus is log-squashed, $b = \tfrac12\log(1 + \text{epistemic}/\text{aleatoric})$
(`opax.py:49-61`), and its denominator (the prior's aleatoric variance) also grows with hold
length. Additionally, the switch cost — the one term that rewards long holds in the task phase —
**never enters the exploration objective**: `OpaxBridge.sample` replaces the reward wholesale
with the bonus (`opax_bridge.py:28-44`). So there is *currently no objective for OPAX to explore
higher k*. The wandb data confirms the theory exactly: dt=1 from the first OPAX-controlled step,
not a gradual collapse.

Consequence: whichever hypothesis explains the critic's optimism, OPAX-phase data validates the
flow model only at k=1 — the task phase then plans on an unvalidated model. The near-term remedy
is to decouple model-data coverage from the exploration policy (Part A/B: uniform dt
resampling); reforming the objective itself is a research question, parked as a study arm:

**Study arm — dt-aware OPAX (design sketch, post-Wave-1):**
1. *Economics alignment*: charge `switch_cost` inside the exploration objective (add the
   per-decision penalty to the bonus reward in `OpaxBridge`), so exploration faces the same
   decision price as the task phase and long holds amortize it.
2. *k-sweep disagreement*: at each imagined state, evaluate the ensemble prior's disagreement
   across a grid of k values and reward the k with the largest disagreement (optionally per unit
   time). This targets the flow map's k-axis directly — explore where the *model* is uncertain
   in k, not where decisions are cheap.
3. *Count-based (state-region, k) novelty*: an explicit coverage bonus over dt bins,
   complementing ensemble disagreement (which is unreliable off-distribution).
4. For the record: the deleted `opax_dt_normalization` flag (bonus ÷ k) pushes the *wrong* way
   for coverage — $b(k)/k$ makes dt=1 *more* attractive, not less — but its plumbing pattern
   (dt-conditioned bonus shaping in `OpaxBridge`) is the natural insertion point for (1)–(3).

---

## 5. Part A — The Purge

Revert or delete the following; each item either embodies an unconfirmed hypothesis or is
non-vanilla machinery whose motivating failure mode is now explained by §3–§4.

1. **Delete the expectile cost head entirely** (never run in any observed experiment; one-sided
   error weighting risks introducing its own misalignment; superseded by the coverage-first
   theory under which plain MSE should suffice):
   - `world_model.py`: the `cost_head`/`cost_expectile_tau` fields, their `__init__` args and
     asserts, and the expectile branch in `variational_step` (`world_model.py:393-405`) — back
     to the single Gaussian log-prob for the full reward+cost output.
   - `configs/agent/actsafe.yaml`: remove `model.cost_head`, `model.cost_expectile_tau`.
   - `configs/experiment/safe_goal_tase.yaml`: remove the `cost_head: expectile` /
     `cost_expectile_tau` overrides.
   - `actsafe.py`: no change needed (it passes `**config.agent.model` through).
2. **Delete `opax_dt_normalization`** (non-vanilla, default-off, wrong-signed for the coverage
   problem, §4.4): the flag and dt-division branch in `opax.modify_reward` (`opax.py:25-37`),
   the `dt_normalization` field of `OpaxBridge`, its wiring in `exploration.py:68,83`, and the
   config key in `configs/agent/actsafe.yaml`.
3. **Un-revert `dt_exploration`: `policy` → `uniform`** in `safe_goal_tase.yaml`. The mechanism
   already exists and is correct: `ActSafe.__call__` (`actsafe.py:198-207`) resamples **only the
   dt head** of the *executed* action uniformly over the **full** pseudo-time range [−1,1]
   during the exploration window, keeps the policy's motor dims, and keeps `prev_action`
   consistent for the RSSM filter. Imagination and all losses untouched. §4 shows policy-mode
   coverage will never materialize; this is the accepted decoupling.
4. **Fix `UniformExploration` (dt dim only)** (`exploration.py:99-105`): sample the last action
   dim in [−1,1] instead of [0,1), so the offline phase covers all of $[k_{\min},k_{\max}]$
   rather than only $[8.5, 16]$. Motor dims deliberately stay at upstream's [0,1) for
   comparability with every existing baseline run (upstream oddity: offline forces are
   positive-only; flagged here, not changed).
5. **Explicitly NOT purged** (audited sound in §1): the variable discount $\gamma^k$ with the
   floor/nearest STE; the fully differentiable discount (no stop-gradient on dt); the
   nearest-integer mapping; the action-conditioned **aggregate** decoder (flow-consistent — no
   per-step factorization, no hazard-rate head); full-MDP time with analytic $k/T$ propagation;
   the discounted CMDP formulation and the $B = d/(T(1-\gamma_c))$ budget; the
   `cost_realized`/`exposure` buffer fields (retained as *diagnostic* targets); the per-k
   `cost_calibration` metrics; `tests/test_budget_invariance.py`; the
   `constraint_pessimism_source` flag (ablation only — κ·UCB is documented inert against
   systematic model error and must not be tuned as a fix).

## 6. Part B — The Fix (coverage-first)

Ordered; step 1 is the fix under the central diagnosis, step 2 is what proves or refutes it.

1. **Coverage restoration** = Part A items 3 + 4. Success criteria, read per epoch during the
   exploration window: `train/ct/buffer/frac_dt_q1..q4` roughly balanced,
   `near_hazard_distinct_dt` high (same-state multi-dt contrast near hazards is precisely what
   the flow cost head needs), `frac_dt_max > 0`.
2. **Diagnostics battery** (all cheap, per-epoch, logged alongside existing metrics):
   - **Per-k calibration gaps + per-k sample counts** (extend `cost_calibration` with raw
     counts, not just `frac_*`): the pass/fail gate is *gaps ≈ 0 at buckets with non-trivial
     counts*, `gap_hazard_holds` especially (zero-inflation hides optimism in bucket means).
   - **Per-k world-model error**: bucket the reconstruction error and KL of `variational_step`
     by `exposure` — tests "the flow model is unvalidated at large k" directly, on the dynamics
     and not just the cost channel. This is the diagnostic that discriminates §3 from a pure
     cost-head story.
   - **Imagined-vs-realized episode cost**: roll `model.sample` from matched buffer start
     states, compare $\sum_j \gamma_c^{t_j}\hat c_j$ against the stored discounted episode cost
     — the only probe that exercises multi-step imagination compounding (the current
     `report()` calibration is teacher-forced one-step and cannot see it).
   - **Perceived-vs-realized k-slope** (§2 mechanism check): finite-difference
     $\hat c(s,u,\cdot)$ and $\hat Q_c(s,u,\cdot)$ over the k-grid on a batch of imagined
     states; compare sign/magnitude against the buffer's realized per-k cost slope.
   - **`dt_head_grad_norm`** (per max_repeat cap): monitors both tanh saturation (§1.4) and the
     STE Jacobian's $(k_{\max}-k_{\min})/2$ scaling before any cap-invariance claim.
3. **Escalation clause** (not a plan item): if, *after* coverage is demonstrably fixed (balanced
   counts), material optimism persists at well-covered k — then and only then revisit
   estimator-level changes. The expected outcome under this review's theory is that plain
   Gaussian/MSE calibrates fine once the data exists.
4. **Guards / notes**:
   - Hard-fail (refuse, don't silently backfill) when a pre-exposure pickle is resumed with CT
     enabled — `_ensure_aux_arrays`'s `exposure≡1` backfill makes every per-k diagnostic above
     read clean for a data-corruption reason. Fresh runs only for anything reported.
   - LBSGD η decays every step in both branches → the barrier vanishes over long runs; once the
     constraint becomes honest this may need attention (e.g. η floor or per-phase reset). Watch
     item, no change now.
   - Imagination has no termination at $t=1$ (`times` saturate via `jnp.minimum`); imagined
     post-horizon transitions never occur in reality. Bounded impact under $\gamma_c<1$; note
     only.
5. **Study arm** (design only, post-Wave-1): dt-aware OPAX per §4 — switch cost in the
   exploration objective, k-sweep disagreement, count-based (state, k) novelty.

## 7. Verification

- **Tests**: `pytest -v` stays green (`test_budget_invariance.py`, `test_ct_time.py`,
  `test_world_model_action_conditioning.py`); expectile-specific tests removed with the head;
  new unit test asserting `UniformExploration`'s dt dim spans [−1,1] and the dt-exploration
  resample covers all of $[k_{\min}, k_{\max}]$ under the nearest-int map.
- **Run protocol**: fresh run (never resumed from pre-exposure pickles). Read order:
  (1) buffer coverage (`frac_dt_q*`, `near_hazard_distinct_dt`) during exploration — if this
  fails, nothing downstream is interpretable; (2) per-k model error + calibration gaps *with
  counts*; (3) imagined-vs-realized episode cost; (4) constraint sign vs realized
  `cost_return` — the 100%-safe-while-violating signature of `e6cttmuk` must break;
  (5) task-phase `cost_return ≤ 25` and `dt_near_far_ratio < 1` (decides faster near hazards)
  for the adaptiveness claim.

---

### Summary of answers to the three doubts

1. **Time augmentation / encoder bypass** — valid; exact DP on the augmented MDP;
  $t+k/T$ is $t+k$ in normalized units; consistency verified at all three touchpoints.
2. **Do k-gradients flow, is $\gamma^k$/STE sound?** — yes, three paths, STE correct, LBSGD
   receives real dt columns. The issue is not flow but *fidelity*: the discount side of the
   constraint gradient is exact while the cost side is learned, and it is learned from a buffer
   with essentially no data at the k the task phase uses.
3. **Constraint gradients for dt** — the discount term is correct physics with no universal
   direction (net $\gamma_c^k(\rho_k - (1-\gamma_c)V_c)$); the failure mode is the learned
   $\partial\hat c/\partial k$ flattening at under-covered k, making "longer = safer" the
   perceived gradient everywhere. The fix is data (coverage), verified by the k-slope and per-k
   calibration diagnostics — not a change to the formulation.
