from typing import NamedTuple, TypedDict
import jax
import jax.nn as jnn
import jax.numpy as jnp
import equinox as eqx
import distrax as dtx
from optax import OptState

from actsafe.common.learner import Learner
from actsafe.common.mixed_precision import apply_mixed_precision
from actsafe.actsafe.rssm import RSSM, Features, ShiftScale, State
from actsafe.rl import ct_time
from actsafe.rl.types import Prediction
from actsafe.actsafe.utils import marginalize_prediction
from actsafe.rl.types import Policy
from actsafe.rl.utils import nest_vmap

_EMBEDDING_SIZE = 1024


class Encoder(eqx.Module):
    cnn_layers: list[eqx.nn.Conv2d]

    def __init__(
        self,
        image_channels: int,
        *,
        key: jax.Array,
    ):
        kernels = [4, 4, 4, 4]
        depth = 32
        keys = jax.random.split(key, len(kernels))
        in_channels = image_channels
        self.cnn_layers = []
        for i, (key, kernel) in enumerate(zip(keys, kernels)):
            out_channels = 2**i * depth
            self.cnn_layers.append(
                eqx.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                    stride=2,
                    key=key,
                )
            )
            in_channels = out_channels

    def __call__(self, observation: jax.Array) -> jax.Array:
        x = observation
        for layer in self.cnn_layers[:-1]:
            x = jnn.elu(layer(x))
        x = self.cnn_layers[-1](x)
        x = x.ravel()
        return x


class ImageDecoder(eqx.Module):
    linear: eqx.nn.Linear
    cnn_layers: list[eqx.nn.ConvTranspose2d]
    output_shape: tuple[int, int, int] = eqx.static_field()

    def __init__(
        self,
        state_dim: int,
        output_shape: tuple[int, int, int],
        *,
        key: jax.Array,
    ):
        kernels = [5, 5, 6, 6]
        depth = 32
        linear_key, *keys = jax.random.split(key, len(kernels) + 1)
        in_channels = _EMBEDDING_SIZE
        self.linear = eqx.nn.Linear(state_dim, in_channels, key=linear_key)
        self.cnn_layers = []
        for i, (key, kernel) in enumerate(zip(keys, kernels)):
            out_channels = 2 ** (len(kernels) - i - 2) * depth
            if i != len(kernels) - 1:
                self.cnn_layers.append(
                    eqx.nn.ConvTranspose2d(
                        in_channels, out_channels, kernel, 2, key=key
                    )
                )
            else:
                self.cnn_layers.append(
                    eqx.nn.ConvTranspose2d(
                        in_channels, output_shape[0], kernel, 2, key=key
                    )
                )
            in_channels = out_channels
        self.output_shape = output_shape

    def __call__(self, flat_state: jax.Array) -> jax.Array:
        x = self.linear(flat_state)
        x = x.reshape(_EMBEDDING_SIZE, 1, 1)
        for layer in self.cnn_layers[:-1]:
            x = jnn.elu(layer(x))
        x = self.cnn_layers[-1](x)
        output = x.reshape(self.output_shape)
        return output


class InferenceResult(NamedTuple):
    state: State
    image: jax.Array
    reward_cost: jax.Array
    posteriors: ShiftScale
    priors: ShiftScale


class WorldModel(eqx.Module):
    cell: RSSM
    encoder: Encoder
    image_decoder: ImageDecoder
    reward_cost_decoder: eqx.nn.MLP
    continuous_time: bool = eqx.field(static=True)
    # CT: hold bounds (repeat units) and episode horizon (base steps) for the
    # exact elapsed-time recurrence in imagination; unused when discrete.
    k_min: float = eqx.field(static=True)
    k_max: float = eqx.field(static=True)
    horizon_steps: float = eqx.field(static=True)

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        action_dim: int,
        deterministic_size: int,
        stochastic_size: int,
        hidden_size: int,
        ensemble_size: int,
        initialization_scale: float,
        num_rewards: int,
        continuous_time: bool = False,
        k_min: float = 1.0,
        k_max: float = 1.0,
        horizon_steps: float = 1.0,
        *,
        key,
    ):
        (
            cell_key,
            encoder_key,
            image_decoder_key,
            reward_cost_decoder_key,
        ) = jax.random.split(key, 4)
        # Full-MDP time handling (CT): the elapsed-time clock channel appended by
        # SwitchCostWrapper is a spatially-constant scalar, so it is kept OUT of
        # the CNNs (encoder input and image-decoder output see only the C real
        # pixel channels). The clock still reaches the agent: it is carried as an
        # exact scalar ALONGSIDE the latent (extracted from the channel on real
        # steps, advanced analytically by the executed hold length in
        # imagination) — time is bookkeeping, not something to learn.
        self.continuous_time = continuous_time
        self.k_min = float(k_min)
        self.k_max = float(k_max)
        self.horizon_steps = float(horizon_steps)
        image_channels = image_shape[0] - 1 if continuous_time else image_shape[0]
        actual_image_shape = (image_channels,) + image_shape[1:]
        self.cell = RSSM(
            deterministic_size,
            stochastic_size,
            hidden_size,
            _EMBEDDING_SIZE,
            action_dim,
            ensemble_size,
            initialization_scale,
            key=cell_key,
        )
        self.encoder = Encoder(image_channels=image_channels, key=encoder_key)
        state_dim = stochastic_size + deterministic_size
        self.image_decoder = ImageDecoder(state_dim, actual_image_shape, key=image_decoder_key)
        # num_rewards + 1 = cost + reward
        # width = 400, layers = 2
        # Continuous time (oTaCoS alignment): reward/cost of a hold are transition
        # quantities r̄(s, u, t) / c̄(s, u, t) — the within-hold accumulation is not
        # observable from the arrival frame alone (e.g. passing through a hazard
        # mid-hold), so the decoder is conditioned on the action (incl. the dt dim).
        # Composed with the duration-conditioned dynamics this gives each ensemble
        # member a c̄_i(s, u, t), providing imagination a direct learned dt→cost
        # gradient instead of routing it through the prior/posterior KL.
        decoder_in_dim = state_dim + (action_dim if continuous_time else 0)
        self.reward_cost_decoder = eqx.nn.MLP(
            decoder_in_dim,
            num_rewards + 1,
            400,
            3,
            key=reward_cost_decoder_key,
            activation=jnn.elu,
        )

    def __call__(
        self,
        features: Features,
        actions: jax.Array,
        key: jax.Array,
        init_state: State | None = None,
    ) -> InferenceResult:
        # Keep the CNN on real pixels: the elapsed-time clock channel (last
        # channel, added by SwitchCostWrapper) has no spatial structure. It is
        # NOT dropped from the agent — callers read it back as the scalar time
        # component of the agent state (see ActSafe.update_model / policy).
        obs = features.observation[:, :-1] if self.continuous_time else features.observation
        obs_embeddings = jax.vmap(self.encoder)(obs)

        def fn(carry, inputs):
            prev_state = carry
            embedding, prev_action, key = inputs
            state, posterior, prior = self.cell.filter(
                prev_state, embedding, prev_action, key
            )
            return state, (state, posterior, prior)

        keys = jax.random.split(key, obs_embeddings.shape[0])
        _, (states, posteriors, priors) = jax.lax.scan(
            fn,
            init_state if init_state is not None else self.cell.init,
            (obs_embeddings, actions, keys),
        )
        # Training pairing: states[t] is the posterior that consumed action[t]
        # (post-hold), so decoding reward/cost of hold t from (state[t], action[t])
        # matches the imagination-side pairing in sample() exactly.
        decoder_in = (
            jnp.concatenate([states.flatten(), actions.astype(states.flatten().dtype)], -1)
            if self.continuous_time
            else states.flatten()
        )
        reward_cost = jax.vmap(self.reward_cost_decoder)(decoder_in)
        image = jax.vmap(self.image_decoder)(states.flatten())
        return InferenceResult(states, image, reward_cost, posteriors, priors)

    def infer_state(
        self,
        state: State,
        observation: jax.Array,
        action: jax.Array,
        key: jax.Array,
    ) -> State:
        # Keep the CNN on real pixels (same as in __call__); the caller reads the
        # clock channel back as the scalar time component of the agent state.
        obs = observation[:-1] if self.continuous_time else observation
        obs_embeddings = self.encoder(obs)
        state, *_ = self.cell.filter(state, obs_embeddings, action, key)
        return state

    def sample(
        self,
        horizon: int,
        initial_state: State | jax.Array,
        key: jax.Array,
        policy: Policy,
        initial_time: jax.Array | None = None,
    ) -> tuple[Prediction, ShiftScale]:
        def f(carry, inputs):
            prev_state, prev_time = carry
            if callable(policy):
                key = inputs
                key, p_key = jax.random.split(key)
                flat = prev_state.flatten()
                if self.continuous_time:
                    flat = jnp.concatenate([flat, prev_time[None].astype(flat.dtype)])
                action = policy(jax.lax.stop_gradient(flat), p_key)
            else:
                action, key = inputs
            ensemble_states, prior = self.cell.predict(prev_state, action, key)
            key, prior_key = jax.random.split(key)
            id = jax.random.randint(prior_key, (), 0, self.cell.ensemble_size)
            state = jax.tree_map(lambda x: x[id], ensemble_states)
            if self.continuous_time:
                # Exact bookkeeping, matching SwitchCostWrapper: the clock
                # advances by the executed hold length as a fraction of the
                # episode horizon. Nothing is learned here.
                k = ct_time.dt_ratio_from_pseudo_jnp(
                    action[-1], self.k_min, self.k_max
                )
                time = jnp.minimum(prev_time + k / self.horizon_steps, 1.0)
            else:
                time = prev_time
            return (state, time), (action, state, time, ensemble_states, prior)

        if isinstance(policy, jax.Array):
            inputs: tuple[jax.Array, jax.Array] | jax.Array = (
                policy,
                jax.random.split(key, policy.shape[0]),
            )
            assert policy.shape[0] <= horizon
        elif callable(policy):
            inputs = jax.random.split(key, horizon)
        else:
            raise ValueError("policy must be callable or jax.Array")
        if isinstance(initial_state, jax.Array):
            if self.continuous_time:
                # Flat agent states carry the elapsed-time fraction as last dim.
                initial_time = initial_state[..., -1]
                initial_state = initial_state[..., :-1]
            initial_state = State.from_flat(initial_state, self.cell.stochastic_size)
        if initial_time is None:
            initial_time = jnp.asarray(0.0, jnp.float32)
        initial_time = jnp.asarray(initial_time, jnp.float32)
        _, (actions, trajectory, times, ensemble_trajectories, priors) = jax.lax.scan(
            f, (initial_state, initial_time), inputs
        )

        # vmap twice: once for the ensemble, and second time for the horizon
        ensemble_flat = ensemble_trajectories.flatten()
        if self.continuous_time:
            # Broadcast actions [T, A] over the ensemble axis [T, E, A] so each
            # member decodes its own c̄_i(s, u, t) from (arrival latent, action).
            tiled_actions = jnp.broadcast_to(
                actions[:, None].astype(ensemble_flat.dtype),
                ensemble_flat.shape[:2] + actions.shape[-1:],
            )
            decoder_in = jnp.concatenate([ensemble_flat, tiled_actions], -1)
        else:
            decoder_in = ensemble_flat
        out = nest_vmap(self.reward_cost_decoder, 2)(decoder_in)
        # Ensemble axis before time axis.
        out, priors = _ensemble_first((out, priors))
        reward, cost = out[..., :-1], out[..., -1]
        next_state = trajectory.flatten()
        if self.continuous_time:
            # Agent state = (latent, elapsed-time fraction): critics and the
            # actor see the arrival time of each imagined hold.
            next_state = jnp.concatenate(
                [next_state, times[:, None].astype(next_state.dtype)], -1
            )
        out = Prediction(actions, next_state, reward, cost)
        return out, priors


class TrainingResults(TypedDict):
    reconstruction_loss: jax.Array
    kl_loss: jax.Array
    states: State


@eqx.filter_jit
@apply_mixed_precision(
    target_input_names=["features", "actions"],
    target_module_names=["model"],
)
def variational_step(
    features: Features,
    actions: jax.Array,
    model: WorldModel,
    learner: Learner,
    opt_state: OptState,
    key: jax.Array,
    beta: float = 1.0,
    free_nats: float = 0.0,
    kl_mix: float = 0.8,
    with_reward: bool = True,
    inference_only: bool = False,
) -> tuple[tuple[WorldModel, OptState], tuple[jax.Array, TrainingResults]]:
    def loss_fn(model, static_part=None):
        if static_part is not None:
            model = eqx.combine(model, static_part)
        infer_fn = lambda features, actions: model(features, actions, key)
        inference_result: InferenceResult = eqx.filter_vmap(infer_fn)(features, actions)
        batch_ndim = 2
        logprobs = (
            lambda predictions, targets: dtx.Independent(
                dtx.Normal(targets, 1.0), targets.ndim - batch_ndim
            )
            .log_prob(predictions)
            .mean()
        )
        predictions = inference_result.reward_cost
        reward_pred, cost_pred = predictions[..., :-1], predictions[..., -1]
        if not with_reward:
            # Train only the cost channel; the reward term degenerates to a
            # constant (zeros vs zeros), kept so the logged loss stays
            # comparable across branches.
            reward_pred = jnp.zeros_like(features.reward)
            reward_target = jnp.zeros_like(features.reward)
        else:
            reward_target = features.reward
        reward_logprobs = logprobs(reward_pred, reward_target)
        cost_target = features.cost
        cost_logprobs = logprobs(cost_pred[..., None], cost_target[..., None])
        reward_cost_logprobs = reward_logprobs + cost_logprobs
        # The image decoder is only trained to reconstruct real image pixels;
        # the clock channel is exact bookkeeping and never learned.
        target_obs = features.observation[:, :, :-1] if model.continuous_time else features.observation
        image_logprobs = logprobs(inference_result.image, target_obs)
        reconstruction_loss = -reward_cost_logprobs - image_logprobs
        kl_loss = kl_divergence(
            inference_result.posteriors, inference_result.priors, free_nats, kl_mix
        )
        assert isinstance(reconstruction_loss, jax.Array)
        aux = TrainingResults(
            reconstruction_loss=reconstruction_loss,
            kl_loss=kl_loss,
            states=inference_result.state,
        )
        return reconstruction_loss + beta * kl_loss, aux

    (loss, rest), model_grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    if inference_only:
        return (model, opt_state), (loss, rest)
    new_model, new_opt_state = learner.grad_step(model, model_grads, opt_state)
    return (new_model, new_opt_state), (loss, rest)


def partition_dynamics_rewards(model: WorldModel) -> tuple[WorldModel, WorldModel]:
    filter_spec = jax.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(lambda tree: tree.reward_cost_decoder, filter_spec, True)
    diff_model, static_model = eqx.partition(model, filter_spec)
    return diff_model, static_model


# https://github.com/danijar/dreamerv2/blob/259e3faa0e01099533e29b0efafdf240adeda4b5/common/nets.py#L130
def kl_divergence(
    posterior: ShiftScale, prior: ShiftScale, free_nats: float, mix: float
) -> jax.Array:
    sg = lambda x: jax.tree_map(jax.lax.stop_gradient, x)
    mvn = lambda scale_shift: dtx.MultivariateNormalDiag(*scale_shift)
    lhs = mvn(posterior).kl_divergence(mvn(sg(prior))).mean()
    rhs = mvn(sg(posterior)).kl_divergence(mvn(prior)).mean()
    return (1.0 - mix) * jnp.maximum(lhs, free_nats) + mix * jnp.maximum(rhs, free_nats)


@eqx.filter_jit
def evaluate_model(
    model: WorldModel, features: Features, actions: jax.Array, key: jax.Array
) -> jax.Array:
    observations = features.observation
    length = min(observations.shape[1] + 1, 50)
    conditioning_length = length // 5
    key, subkey = jax.random.split(key)
    features = jax.tree_map(lambda x: x[0, :conditioning_length], features)
    inference_result = model(features, actions[0, :conditioning_length], subkey)
    state = jax.tree_map(lambda x: x[-1], inference_result.state)
    if model.continuous_time:
        # Elapsed-time fraction at the end of conditioning (preprocess maps the
        # 255*frac clock channel to frac - 0.5).
        initial_time = features.observation[-1, -1, 0, 0] + 0.5
    else:
        initial_time = None
    prediction, _ = model.sample(
        length - conditioning_length,
        state,
        key,
        actions[0, conditioning_length:],
        initial_time=initial_time,
    )
    prediction = marginalize_prediction(prediction)
    latent = (
        prediction.next_state[..., :-1]
        if model.continuous_time
        else prediction.next_state
    )
    y_hat = jax.vmap(model.image_decoder)(latent)
    y = observations[0, conditioning_length:]
    # Strip time channel from ground-truth for comparison with decoded images.
    if model.continuous_time:
        y = y[:, :-1]
    error = jnp.abs(y - y_hat) / 2.0 - 0.5
    normalize = lambda image: ((image + 0.5) * 255).astype(jnp.uint8)
    
    # y, y_hat, error are all (T, H, W, 3)
    # Tile them horizontally: (T, H, 3*W, 3)
    out = jnp.concatenate([normalize(y), normalize(y_hat), normalize(error)], axis=2)
    
    # WandB expects (T, C, H, W)
    return out.transpose(0, 3, 1, 2)


def _ensemble_first(x):
    return jax.tree_map(lambda x: x.swapaxes(0, 1), x)


@eqx.filter_jit
def cost_calibration(
    model: WorldModel,
    features: Features,
    actions: jax.Array,
    exposure: jax.Array,
    key: jax.Array,
) -> dict[str, jax.Array]:
    """Per-hold-length calibration + model error on a replay batch.

    gap_* = mean(predicted − target) per exposure bucket (executed base steps
    k): NEGATIVE = the model under-predicts cost there — the optimistic
    direction behind the corr(violation, calibration gap) = +0.91 finding.
    count_* are the raw bucket sizes: a clean gap on an empty bucket proves
    nothing, so the pass/fail gate is gaps ≈ 0 AT BUCKETS WITH NON-TRIVIAL
    COUNTS (coverage-first diagnosis, code_review.md §3/§6).

    recon_*/kl_* bucket the image-reconstruction MSE and the posterior‖prior
    KL by exposure — the probe that tests "the flow model is unvalidated at
    large k" on the DYNAMICS, not just the cost channel (discriminates the
    coverage diagnosis from a pure cost-head story).
    """
    infer_fn = lambda features, actions: model(features, actions, key)
    result = eqx.filter_vmap(infer_fn)(features, actions)
    pred = result.reward_cost[..., -1]
    target = features.cost
    gap = pred - target
    # Per-position world-model error (both [batch, time]).
    target_obs = (
        features.observation[:, :, :-1]
        if model.continuous_time
        else features.observation
    )
    recon_error = ((result.image - target_obs) ** 2).mean(axis=(-3, -2, -1))
    mvn = lambda scale_shift: dtx.MultivariateNormalDiag(*scale_shift)
    kl = mvn(result.posteriors).kl_divergence(mvn(result.priors))
    buckets = {
        "k_1": exposure <= 1.0,
        "k_2_4": (exposure >= 2.0) & (exposure <= 4.0),
        "k_5_8": (exposure >= 5.0) & (exposure <= 8.0),
        "k_9_plus": exposure >= 9.0,
    }
    prefix = "agent/cost_calibration/"
    model_prefix = "agent/model_per_k/"
    metrics: dict[str, jax.Array] = {}
    for name, mask in buckets.items():
        count = mask.sum()
        denominator = jnp.maximum(count, 1)
        bucket_mean = lambda x: jnp.where(count > 0, (x * mask).sum() / denominator, 0.0)
        metrics[f"{prefix}gap_{name}"] = bucket_mean(gap)
        metrics[f"{prefix}target_{name}"] = bucket_mean(target)
        metrics[f"{prefix}frac_{name}"] = mask.mean()
        metrics[f"{prefix}count_{name}"] = count
        metrics[f"{model_prefix}recon_{name}"] = bucket_mean(recon_error)
        metrics[f"{model_prefix}kl_{name}"] = bucket_mean(kl)
    metrics[f"{prefix}gap_overall"] = gap.mean()
    # Hazard-positive holds only: zero-inflation hides the optimism in the
    # overall mean (most holds cost exactly zero and are predicted ≈ zero).
    positive = target > 0
    positive_count = positive.sum()
    metrics[f"{prefix}gap_hazard_holds"] = jnp.where(
        positive_count > 0,
        (gap * positive).sum() / jnp.maximum(positive_count, 1),
        0.0,
    )
    metrics[f"{prefix}frac_hazard_holds"] = positive.mean()
    return metrics


@eqx.filter_jit
def imagined_vs_realized_cost(
    model: WorldModel,
    features: Features,
    actions: jax.Array,
    exposure: jax.Array,
    safety_discount: float,
    key: jax.Array,
) -> dict[str, jax.Array]:
    """Open-loop imagined vs realized discounted cost over a replay window.

    Conditions the posterior on the first fifth of each sequence, then rolls
    the model OPEN-LOOP with the stored actions (same holds, same elapsed-time
    schedule) and compares the discounted cost sums. This is the only probe
    that exercises multi-step imagination compounding — cost_calibration above
    is teacher-forced one-step and cannot see it. NEGATIVE gap = imagination
    is optimistic about the very trajectories the buffer realized.
    """
    horizon = actions.shape[1]
    context = max(horizon // 5, 1)
    infer_fn = lambda features, actions: model(features, actions, key)
    context_features = jax.tree_map(lambda x: x[:, :context], features)
    result = eqx.filter_vmap(infer_fn)(context_features, actions[:, :context])
    context_state = jax.tree_map(lambda x: x[:, -1], result.state)
    if model.continuous_time:
        # Elapsed-time fraction at the end of conditioning (preprocess maps the
        # 255*frac clock channel to frac - 0.5).
        initial_time = features.observation[:, context - 1, -1, 0, 0] + 0.5
    else:
        initial_time = jnp.zeros(actions.shape[0])

    def rollout(state, acts, time, key):
        prediction, _ = model.sample(
            horizon - context, state, key, acts, initial_time=time
        )
        return prediction.cost.mean(0)  # ensemble mean, [horizon - context]

    keys = jax.random.split(key, actions.shape[0])
    imagined = eqx.filter_vmap(rollout)(
        context_state, actions[:, context:], initial_time, keys
    )
    realized = features.cost[:, context:]
    window_exposure = exposure[:, context:]
    # Discount each hold by the base steps elapsed before it (both sides use
    # the same realized schedule, so the difference isolates the model).
    elapsed = jnp.cumsum(window_exposure, axis=1) - window_exposure
    weights = safety_discount**elapsed
    imagined_return = (weights * imagined).sum(1)
    realized_return = (weights * realized).sum(1)
    prefix = "agent/imagination/"
    return {
        f"{prefix}cost_return_imagined": imagined_return.mean(),
        f"{prefix}cost_return_realized": realized_return.mean(),
        f"{prefix}cost_return_gap": (imagined_return - realized_return).mean(),
    }


@eqx.filter_jit
def k_slope_diagnostics(
    model: WorldModel,
    safety_critic,
    features: Features,
    actions: jax.Array,
    exposure: jax.Array,
    safety_discount: float,
    key: jax.Array,
) -> dict[str, jax.Array]:
    """Perceived vs realized dt→cost slope (code_review.md §2 mechanism check).

    Perceived: finite difference between k_min and k_max of the decoded hold
    cost ĉ(s, u, k) and of Q̂_c(s, u, k) = ĉ(s, u, k) + γc^k V̂_c(s'_k, t'),
    averaged over a replay batch's posterior states with the stored motor
    actions (counterfactual: same state, same u, different hold). Realized:
    OLS slope of the stored per-hold cost against exposure. perceived_qc < 0
    while realized_cost > 0 is the degenerate "longer looks safer" gradient
    that LBSGD rationally exploits. Batch-mean only — near-hazard states are
    where the sign matters most, so read alongside near_hazard_* coverage.
    """
    infer_fn = lambda features, actions: model(features, actions, key)
    result = eqx.filter_vmap(infer_fn)(features, actions)
    flat = result.state.flatten()
    states = flat.reshape(-1, flat.shape[-1])
    acts = actions.reshape(-1, actions.shape[-1]).astype(states.dtype)
    times = (features.observation[:, :, -1, 0, 0] + 0.5).reshape(-1)
    keys = jax.random.split(key, states.shape[0])

    def perceived(k: float):
        pseudo = ct_time.pseudo_from_dt_ratio(k, model.k_min, model.k_max)

        def one(state_flat, action, time, key):
            action = action.at[-1].set(pseudo)
            state = State.from_flat(state_flat, model.cell.stochastic_size)
            arrival, _ = model.cell.predict(state, action, key)
            arrival_flat = arrival.flatten()  # [ensemble, state_dim]
            tiled = jnp.broadcast_to(
                action, arrival_flat.shape[:1] + action.shape
            )
            cost = jax.vmap(model.reward_cost_decoder)(
                jnp.concatenate([arrival_flat, tiled], -1)
            )[..., -1].mean()
            arrival_time = jnp.minimum(time + k / model.horizon_steps, 1.0)
            critic_in = jnp.concatenate(
                [
                    arrival_flat,
                    jnp.full((arrival_flat.shape[0], 1), arrival_time, states.dtype),
                ],
                -1,
            )
            value = jax.vmap(safety_critic)(critic_in).mean()
            return cost, cost + safety_discount**k * value

        return jax.vmap(one)(states, acts, times, keys)

    cost_lo, q_lo = perceived(model.k_min)
    cost_hi, q_hi = perceived(model.k_max)
    dk = max(model.k_max - model.k_min, 1.0)
    # Realized: batch OLS of stored hold cost on exposure (confounded by where
    # the policy holds long, but the SIGN contrast vs perceived is the signal).
    x = exposure.reshape(-1)
    y = features.cost.reshape(-1)
    x_centered = x - x.mean()
    variance = (x_centered**2).mean()
    realized_slope = jnp.where(
        variance > 0,
        (x_centered * (y - y.mean())).mean() / jnp.maximum(variance, 1e-8),
        0.0,
    )
    prefix = "agent/ct/kslope/"
    return {
        f"{prefix}perceived_cost": ((cost_hi - cost_lo) / dk).mean(),
        f"{prefix}perceived_qc": ((q_hi - q_lo) / dk).mean(),
        f"{prefix}realized_cost": realized_slope,
    }
