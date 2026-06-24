---
name: project-paper-direction
description: Paper-worthiness assessment and framing strategy for ActSafe-CT (as of 2026-06-17)
metadata:
  node_type: memory
  type: project
  originSessionId: current
---

# ActSafe-CT Paper Direction

## User's concern (2026-06-17)
User is worried that "combine TaCoS + ActSafe" is not enough for ICLR/ICML.
Also: query budget / switch cost auto-tuning idea was ruled out because user has
another paper in that direction (model-free, hardware validated).

## Honest Assessment

**Weak framing (avoid):** "We extend ActSafe to continuous time."  
**Stronger framing:** "Safe exploration with learned world models has a frequency
dimension — the agent must learn not just WHERE to explore safely but HOW OFTEN.
Fixed-frequency control has a U-shaped failure curve: too slow = unsafe (missed
corrections), too fast = no reward gain. CT adaptive finds the bottom of the U."

## What IS novel
1. No prior work does model-based safe RL with learned world model in CT.
   TaCoS = CT theory. ActSafe = model-based safe RL. Combination is new.
2. The **safety-frequency insight**: optimal dt is state-dependent, set by the
   safety constraint. Short dt near boundary, long dt when safe. If this shows up
   empirically in the dt distribution conditioned on safety slack, that's compelling.
3. The empirical trade-off curve: fixed-frequency Pareto-dominated by CT adaptive.

## What makes it stronger
- Harder environment than CartPole (continuous control, Safety-Gym car/point tasks)
- Log dt distribution conditioned on proximity to safety boundary → show state-dependent timing
- Even a sketch theory: dt*(s) ∝ safety_slack(s) or similar

## Venue calibration
- ICLR/ICML main: needs CT experiments working well + harder env + safety-frequency theory
- NeurIPS: same
- ICLR workshop (Safe RL / Real World RL): CT works on CartPole + clear failure mode visualization
- CoRL: hardware-relevant env

## Critical dependency
All CT results before 2026-06-17 were collected with Bug A (missing stop_gradient)
making CT training structurally impossible. No conclusions can be drawn from those
runs. The fixed version has not yet been run. **Wait for re-run results before any
paper decision.**

**Why:** The stop_gradient bug caused the CT actor to always increase dt to hide
imagined costs. This is a fundamental training failure, not a tuning issue.
