# Memory Index

- [Bugs Fixed in CT Implementation](project_bugs_fixed.md) — 6 bugs total; critical: stop_gradient missing on discount (primary CT failure, fixed 2026-06-17), round→floor in opax.py and epoch_summary.py
- [Euler Cluster Infrastructure](project_cluster_infra.md) — slurm.yaml state, mujoco<3.9.0 fix, standard training command
- [CT Architecture Design](project_ct_architecture.md) — SwitchCostWrapper floor semantics, STE invariants, LBSGD constraint flow, key config values
- [User Profile](user_profile.md) — ETH Euler, JAX/RL expertise, ls_krausea account
- [Interaction Style Feedback](feedback_style.md) — full code audits when debugging, explain math not just settings
- [Paper Direction](project_paper_direction.md) — novelty concerns, strongest framing, venue calibration; all CT results pre-2026-06-17 are invalid (Bug A)
- [Strategy 2026-06-23](project_strategy_2026-06-23.md) — DECIDED: pivot to Safety-Gym PointGoal, all-in ICLR 2027; upstream added; discrete path verified faithful; LOAD-BEARING bug: safety cost discounted within macro-window (wrappers.py:191-192)
