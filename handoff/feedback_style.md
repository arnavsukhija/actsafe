---
name: feedback-style
description: "User's preferred interaction style and debugging approach"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 9686672c-25cf-4aa9-82a2-e74562b76208
---

# Interaction Style Preferences

When something is broken and the user asks to investigate, do a full audit of ALL related code the user touched — not just the immediate symptom. The user explicitly said: "analyse every bit of code I gave you since the start, everything I implemented and see if I introduced bugs, i am pretty sure in all the issues I did introduce bugs."

**Why:** The user knows their own implementation may have multiple compounding bugs. Surface-level fixes that address one symptom while missing deeper root causes waste time and cause confusion (e.g., safety_discount=1.0 "fix" that made things worse).

**How to apply:** When debugging training failures or unexpected behavior, systematically trace through all user-modified files (wrappers.py, safe_actor_critic.py, make_actor_critic.py, lbsgd.py, etc.) rather than stopping at the first plausible explanation.

---

Don't suggest reverting to defaults as a cop-out — find the actual bug and explain why the correct implementation should work. The user wants to understand the math, not just "try this setting."

**Why:** User has strong ML background and wants to reason about correctness, not cargo-cult config values.
