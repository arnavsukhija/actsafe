from typing import Protocol
import jax

from actsafe.actsafe.rssm import ShiftScale
from actsafe.opax import normalized_epistemic_uncertainty


class Sentiment(Protocol):
    def __call__(self, values: jax.Array, state_distribution: ShiftScale) -> jax.Array:
        ...

def make_sentiment(alpha, source: str = "latent") -> Sentiment:
    if alpha is None or alpha == 0.0:
        return bayes
    elif alpha > 0.0:
        if source == "cost_spread":
            return ValueSpreadUCB(alpha)
        elif source == "latent":
            return UpperConfidenceBound(alpha)
        else:
            raise ValueError(f"Invalid sentiment source: {source}")
    else:
        raise ValueError(f"Invalid alpha: {alpha}")

def identity(values: jax.Array, state_distribution: ShiftScale) -> jax.Array:
    return values


def bayes(values: jax.Array, state_distribution: ShiftScale) -> jax.Array:
    return values.mean(1)


class UpperConfidenceBound(Sentiment):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, values: jax.Array, state_distribution: ShiftScale) -> jax.Array:
        bonus = normalized_epistemic_uncertainty(state_distribution, 1)
        return values.mean(1) + self.alpha * bonus


class ValueSpreadUCB(Sentiment):
    """UCB from the decoded value ensemble itself: mean + alpha * std over the
    ensemble axis. Unlike UpperConfidenceBound (latent epistemic-uncertainty
    ratio), the bonus lives in the units of the value being estimated, so as a
    constraint pessimism it directly widens the cost estimate where the cost
    heads disagree. Ablation option (sentiment.constraint_pessimism_source:
    cost_spread); default stays the latent UCB.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, values: jax.Array, state_distribution: ShiftScale) -> jax.Array:
        return values.mean(1) + self.alpha * values.std(1)
