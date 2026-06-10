"""Shared inverse-training engine (Strategy + Factory).

Every inverse mode runs on one optimization loop (:func:`_run_inverse_training`)
parameterized by an :class:`InverseStrategy`. The engine owns the invariant
``log_nu`` optimization (optax + warm-up + clipping), timing, history buffers,
Tesseract call counting, and callback dispatch. Presentation lives in
:mod:`burgers_inverse.reporting` so neither the CLI nor the Streamlit app
reimplements the optimization.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from tesseract_core import Tesseract

from burgers_inverse.components import (
    _solver_field,
    get_initial_params,
    image_name_for_backend,
)
from burgers_inverse.configs import LOSS_WEIGHT_NAMES
from burgers_inverse.constants import MIN_OBS_TIME
from burgers_inverse.losses import (
    _loss_from_log_and_params,
    compute_loss_components,
    compute_pointwise_losses,
    initialize_brdr_state,
    summarize_brdr_weights,
    update_brdr_state,
    validate_brdr_loss_weights,
)
from burgers_inverse.observations import (
    generate_grid_observations,
    generate_observations,
)


class TesseractCallCounter:
    """Counts real Tesseract container apply/VJP/JVP dispatches."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.apply_calls = 0
        self.vjp_calls = 0
        self.jvp_calls = 0


@contextmanager
def count_tesseract_calls(counter):
    """Instrument actual container round-trips by wrapping the dispatch layer.

    `tesseract_jax` routes every container call through `Jaxeract.apply`,
    `.vector_jacobian_product`, and `.jacobian_vector_product`. Wrapping these
    yields measured call counts instead of hardcoded estimates.
    """
    from tesseract_jax.tesseract_compat import Jaxeract

    orig_apply = Jaxeract.apply
    orig_vjp = Jaxeract.vector_jacobian_product
    orig_jvp = Jaxeract.jacobian_vector_product

    def apply(self, *args, **kwargs):
        counter.apply_calls += 1
        return orig_apply(self, *args, **kwargs)

    def vjp(self, *args, **kwargs):
        counter.vjp_calls += 1
        return orig_vjp(self, *args, **kwargs)

    def jvp(self, *args, **kwargs):
        counter.jvp_calls += 1
        return orig_jvp(self, *args, **kwargs)

    Jaxeract.apply = apply
    Jaxeract.vector_jacobian_product = vjp
    Jaxeract.jacobian_vector_product = jvp
    try:
        yield counter
    finally:
        Jaxeract.apply = orig_apply
        Jaxeract.vector_jacobian_product = orig_vjp
        Jaxeract.jacobian_vector_product = orig_jvp


@dataclass
class EpochRecord:
    """Per-epoch training state passed to callbacks."""

    epoch: int
    n_epochs: int
    viscosity: float
    log_viscosity: float
    loss: float
    visc_grad_norm: float
    param_grad_norm: float
    epoch_time: float
    apply_calls: int
    vjp_calls: int
    effective_weights: dict
    viscosity_updated: bool
    param_count: int
    brdr_weights: dict | None = None
    loss_components: dict | None = None


class TrainingCallback:
    """No-op base callback. Frontends override the hooks they need."""

    def on_start(self, context):
        pass

    def on_epoch(self, record):
        pass

    def on_finish(self, result):
        pass


def build_training_inputs(config):
    """Sample solver-backed observations and collocation/IC/BC points.

    Shared by the CLI and the app so both consume identical, seed-reproducible
    inputs.
    """
    problem = config.problem
    data_config = config.data
    training = config.training
    domain = problem.domain

    key = jax.random.PRNGKey(data_config.seed)
    key_obs, key_col_x, key_col_t, key_ic, key_bc = jax.random.split(key, 5)

    x_obs, t_obs, u_obs = generate_observations(
        data_config.n_obs,
        problem.true_viscosity,
        domain,
        key_obs,
        noise_std=data_config.noise_std,
    )
    x_col = jax.random.uniform(
        key_col_x, (training.n_col,), minval=domain["x"][0], maxval=domain["x"][1]
    )
    t_col = jax.random.uniform(
        key_col_t, (training.n_col,), minval=MIN_OBS_TIME, maxval=domain["t"][1]
    )
    x_ic = jax.random.uniform(
        key_ic, (training.n_ic,), minval=domain["x"][0], maxval=domain["x"][1]
    )
    t_bc = jax.random.uniform(
        key_bc, (training.n_bc,), minval=MIN_OBS_TIME, maxval=domain["t"][1]
    )
    return x_obs, t_obs, u_obs, x_col, t_col, x_ic, t_bc


# ---------------------------------------------------------------------------
# Shared inverse-training engine (Strategy + Factory)
#
# All three inverse modes — PINN inversion, solver-adjoint inversion, and the
# hybrid discrepancy calibration — share one epoch loop: optimize ``log_nu``
# with optax against a mode-specific differentiable objective, gate it behind a
# warm-up, time each step, count Tesseract dispatches, accumulate history, and
# emit ``EpochRecord``s to a callback. The invariant loop lives in
# ``_run_inverse_training``; each mode supplies an ``InverseStrategy`` owning
# what differs — the data + Tesseract component(s), the value-and-grad objective
# and any auxiliary trainable parameters, the per-epoch telemetry, and the
# mode-specific result keys. ``make_inverse_strategy`` is the factory that picks
# one by mode.
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """Mode-specific values from one optimization step.

    The engine consumes ``log_v_grad`` to advance ``log_nu`` and combines the
    remaining values with complete-epoch Tesseract call telemetry.
    """

    loss: float
    log_v_grad: object
    param_grad_norm: float
    effective_weights: dict
    param_count: int
    brdr_weights: object | None = None
    aux: object = None


class InverseStrategy(ABC):
    """Mode-specific behavior plugged into :func:`_run_inverse_training`.

    The engine owns the invariant ``log_nu`` optimization (optax + warm-up +
    clipping), timing, history buffers, and callback dispatch. A concrete
    strategy owns everything that varies between modes: the observations and the
    Tesseract component(s), the differentiable objective and any auxiliary
    trainable parameters, the per-epoch loss components, and the mode-specific
    keys glued onto the final result dict.
    """

    #: Loss-component names this mode reports; drives the history buffers.
    loss_component_names: tuple[str, ...] = ("total",)

    @abstractmethod
    def open_components(self, stack: ExitStack) -> None:
        """Enter any *owned* Tesseract context managers on ``stack``."""

    @abstractmethod
    def start_context(self, warmup_epochs: int) -> dict:
        """Build the context dict handed to ``callback.on_start``."""

    @abstractmethod
    def run_step(self, log_viscosity, epoch: int) -> StepResult:
        """Run one backward pass, update auxiliary params, and return step values."""

    @abstractmethod
    def epoch_loss_components(self, viscosity, log_viscosity, step, record):
        """Loss components for this epoch's ``EpochRecord`` (``None`` to omit)."""

    @abstractmethod
    def finalize_result(self, base: dict) -> dict:
        """Augment the engine's common ``base`` result with mode-specific keys."""


class PINNStrategy(InverseStrategy):
    """PINN inversion: jointly optimize ``log_nu`` and the PINN parameters,
    differentiating one composed loss through the PINN Tesseract VJP."""

    loss_component_names = ("total", "data", "physics", "ic", "bc")

    def __init__(self, config, *, pinn=None):
        self.config = config
        self.backend = config.backend
        self.training = config.training
        self.data_config = config.data
        self.loss_weights = config.loss.as_dict()
        if self.training.adaptive_loss_weights:
            validate_brdr_loss_weights(self.loss_weights)

        self.inputs = build_training_inputs(config)
        self.image_name = image_name_for_backend(self.backend)
        self.owns_pinn = pinn is None
        self.pinn = pinn if pinn is not None else Tesseract.from_image(self.image_name)

        self.params_flat = get_initial_params(self.backend, seed=self.data_config.seed)
        self.param_optimizer = optax.adam(self.training.param_learning_rate)
        self.param_opt_state = self.param_optimizer.init(self.params_flat)
        self.loss_and_grads = jax.value_and_grad(
            _loss_from_log_and_params, argnums=(0, 1)
        )
        self.brdr_state = None
        self.loss_weight_history = {
            name: [self.loss_weights[name]] for name in LOSS_WEIGHT_NAMES
        }

    def open_components(self, stack):
        if self.owns_pinn:
            stack.enter_context(self.pinn)

    def start_context(self, warmup_epochs):
        x_obs, t_obs, u_obs = self.inputs[:3]
        return {
            "config": self.config,
            "backend": self.backend,
            "image_name": self.image_name,
            "warmup_epochs": warmup_epochs,
            "x_obs": x_obs,
            "t_obs": t_obs,
            "u_obs": u_obs,
        }

    def run_step(self, log_viscosity, epoch):
        x_obs, t_obs, u_obs, x_col, t_col, x_ic, t_bc = self.inputs
        training = self.training
        viscosity = jnp.exp(log_viscosity)

        brdr_weights = None
        if training.adaptive_loss_weights:
            pointwise_losses = compute_pointwise_losses(
                viscosity,
                self.params_flat,
                x_obs,
                t_obs,
                u_obs,
                x_col,
                t_col,
                x_ic,
                t_bc,
                self.pinn,
            )
            if self.brdr_state is None:
                self.brdr_state = initialize_brdr_state(pointwise_losses)
            self.brdr_state = update_brdr_state(
                self.brdr_state,
                pointwise_losses,
                beta_c=training.brdr_beta_c,
                beta_w=training.brdr_beta_w,
                eps=training.brdr_epsilon,
            )
            brdr_weights = self.brdr_state["weights"]

        # One reverse-mode sweep yields the loss and both gradients.
        loss, (log_v_grad, p_grad) = self.loss_and_grads(
            log_viscosity,
            self.params_flat,
            x_obs,
            t_obs,
            u_obs,
            x_col,
            t_col,
            x_ic,
            t_bc,
            self.pinn,
            brdr_weights,
            self.loss_weights,
        )

        param_updates, self.param_opt_state = self.param_optimizer.update(
            p_grad, self.param_opt_state
        )
        self.params_flat = optax.apply_updates(self.params_flat, param_updates)

        effective_weights = (
            dict(self.loss_weights)
            if brdr_weights is None
            else summarize_brdr_weights(brdr_weights)
        )
        for name in LOSS_WEIGHT_NAMES:
            self.loss_weight_history[name].append(effective_weights[name])

        return StepResult(
            loss=float(loss),
            log_v_grad=log_v_grad,
            param_grad_norm=float(jnp.linalg.norm(p_grad)),
            effective_weights=effective_weights,
            param_count=int(self.params_flat.size),
            brdr_weights=brdr_weights,
        )

    def epoch_loss_components(self, viscosity, log_viscosity, step, record):
        if not record:
            return None
        x_obs, t_obs, u_obs, x_col, t_col, x_ic, t_bc = self.inputs
        return {
            name: float(value)
            for name, value in compute_loss_components(
                viscosity,
                self.params_flat,
                x_obs,
                t_obs,
                u_obs,
                x_col,
                t_col,
                x_ic,
                t_bc,
                self.pinn,
                brdr_weights=step.brdr_weights,
                loss_weights=self.loss_weights,
            ).items()
        }

    def finalize_result(self, base):
        warmup_epochs = min(
            self.training.viscosity_warmup_epochs, max(0, self.training.n_epochs - 1)
        )
        return {
            "backend": self.backend,
            "tesseract_image": self.image_name,
            **base,
            "loss_weights": self.loss_weights,
            "loss_weight_history": self.loss_weight_history,
            "brdr_state": self.brdr_state,
            "adaptive_loss_weights": self.training.adaptive_loss_weights,
            "params_flat": self.params_flat,
            "warmup_epochs": warmup_epochs,
            "observations": tuple(self.inputs[:3]),
            "pinn": self.pinn,
        }


def solver_inverse_loss(log_viscosity, obs, solver):
    """Data-fit loss for solver-adjoint inversion: ``||solver(nu)[obs] - u_obs||^2``.

    Differentiating w.r.t. ``log_viscosity`` routes entirely through the solver
    Tesseract's VJP endpoint. No neural network is involved -- this is the
    PDE-constrained / adjoint inverse method, a baseline for the PINN method.
    """
    nu = jnp.exp(log_viscosity)
    u_field = _solver_field(solver, nu, obs.x_grid, obs.t_grid)
    u_pred = u_field[obs.t_idx, obs.x_idx]
    return jnp.mean((u_pred - obs.u_obs) ** 2)


class SolverAdjointStrategy(InverseStrategy):
    """Solver-adjoint inversion: optimize ``log_nu`` against the in-loop solver,
    differentiating the data-fit loss through the solver Tesseract VJP. No neural
    network — the PDE-constrained baseline for the PINN method."""

    loss_component_names = ("total", "data")

    def __init__(self, config, *, solver=None):
        self.config = config
        self.problem = config.problem
        self.data_config = config.data
        key = jax.random.PRNGKey(self.data_config.seed)
        self.obs = generate_grid_observations(
            self.data_config.n_obs,
            self.problem.true_viscosity,
            self.problem.domain,
            key,
            noise_std=self.data_config.noise_std,
        )
        self.owns_solver = solver is None
        self.solver = (
            solver if solver is not None else Tesseract.from_image("burgers_solver")
        )
        self.loss_and_grad = jax.value_and_grad(solver_inverse_loss, argnums=0)

    def open_components(self, stack):
        if self.owns_solver:
            stack.enter_context(self.solver)

    def start_context(self, warmup_epochs):
        return {
            "config": self.config,
            "backend": "solver",
            "image_name": "burgers_solver",
            "warmup_epochs": warmup_epochs,
            "observations": self.obs,
        }

    def run_step(self, log_viscosity, epoch):
        loss, log_v_grad = self.loss_and_grad(log_viscosity, self.obs, self.solver)
        return StepResult(
            loss=float(loss),
            log_v_grad=log_v_grad,
            param_grad_norm=0.0,
            effective_weights={},
            param_count=0,
        )

    def epoch_loss_components(self, viscosity, log_viscosity, step, record):
        return {"total": step.loss, "data": step.loss}

    def finalize_result(self, base):
        return {
            "mode": "solver-inverse",
            "backend": "solver",
            "tesseract_image": "burgers_solver",
            **base,
            "observations": self.obs,
            "solver": self.solver,
        }


def _run_inverse_training(config, strategy, *, callback=None, metrics_every=20):
    """Drive the invariant inverse-training loop for any :class:`InverseStrategy`.

    Optimizes ``log_nu`` against the strategy's objective: each epoch runs the
    strategy's step, applies the (warm-up-gated, optionally clipped) ``log_nu``
    update, records history, and emits an ``EpochRecord``. The strategy supplies
    the objective, telemetry, and the mode-specific result keys.
    """
    problem = config.problem
    data_config = config.data
    training = config.training
    callback = callback or TrainingCallback()
    counter = TesseractCallCounter()

    log_viscosity = jnp.log(jnp.asarray(problem.initial_viscosity))
    log_nu_bounds = None
    if training.clip_log_viscosity:
        log_nu_bounds = (
            jnp.log(jnp.asarray(training.nu_clip_min, dtype=jnp.float32)),
            jnp.log(jnp.asarray(training.nu_clip_max, dtype=jnp.float32)),
        )
    warmup_epochs = min(training.viscosity_warmup_epochs, max(0, training.n_epochs - 1))

    log_visc_optimizer = optax.adam(training.log_nu_learning_rate)
    log_visc_opt_state = log_visc_optimizer.init(log_viscosity)

    viscosity = jnp.exp(log_viscosity)
    times = []
    viscosity_history = [float(viscosity)]
    log_viscosity_history = [float(log_viscosity)]
    loss_history = {name: [] for name in strategy.loss_component_names}

    with ExitStack() as stack:
        strategy.open_components(stack)
        callback.on_start(strategy.start_context(warmup_epochs))

        for epoch in range(training.n_epochs):
            start_time = time.time()
            record = epoch % metrics_every == 0 or epoch == training.n_epochs - 1
            counter.reset()

            # Count the complete epoch: adaptive-weight preparation, gradient pass,
            # and any periodic loss-component evaluation.
            with count_tesseract_calls(counter):
                step = strategy.run_step(log_viscosity, epoch)

                viscosity_updated = epoch >= warmup_epochs
                if viscosity_updated:
                    updates, log_visc_opt_state = log_visc_optimizer.update(
                        step.log_v_grad, log_visc_opt_state
                    )
                    log_viscosity = optax.apply_updates(log_viscosity, updates)
                    if log_nu_bounds is not None:
                        log_viscosity = jnp.clip(
                            log_viscosity, log_nu_bounds[0], log_nu_bounds[1]
                        )
                viscosity = jnp.exp(log_viscosity)

                components = strategy.epoch_loss_components(
                    viscosity, log_viscosity, step, record
                )

            epoch_time = time.time() - start_time
            times.append(epoch_time)
            viscosity_history.append(float(viscosity))
            log_viscosity_history.append(float(log_viscosity))

            if record and components is not None:
                for name in loss_history:
                    loss_history[name].append(components[name])

            callback.on_epoch(
                EpochRecord(
                    epoch=epoch,
                    n_epochs=training.n_epochs,
                    viscosity=float(viscosity),
                    log_viscosity=float(log_viscosity),
                    loss=float(step.loss),
                    visc_grad_norm=float(jnp.abs(step.log_v_grad)),
                    param_grad_norm=step.param_grad_norm,
                    epoch_time=epoch_time,
                    apply_calls=counter.apply_calls,
                    vjp_calls=counter.vjp_calls,
                    effective_weights=step.effective_weights,
                    viscosity_updated=viscosity_updated,
                    param_count=step.param_count,
                    brdr_weights=step.brdr_weights,
                    loss_components=components,
                )
            )

    final_viscosity = float(viscosity)
    relative_error = (
        abs(final_viscosity - problem.true_viscosity) / problem.true_viscosity * 100
    )
    avg_time = sum(times) / len(times) * 1000 if times else 0.0

    base = {
        "final_viscosity": final_viscosity,
        "true_viscosity": problem.true_viscosity,
        "relative_error": relative_error,
        "avg_time_ms": avg_time,
        "viscosity_history": viscosity_history,
        "log_viscosity_history": log_viscosity_history,
        "loss_history": loss_history,
        "seed": data_config.seed,
        "config": config,
    }
    result = strategy.finalize_result(base)
    callback.on_finish(result)
    return result


def train_inverse(config, *, pinn=None, callback=None, metrics_every=20):
    """Run the inverse-viscosity optimization loop (PINN method).

    Thin wrapper over the shared engine with a :class:`PINNStrategy`: a single
    reverse-mode `value_and_grad` over both ``log_nu`` and the PINN parameters
    each step, routing gradients through the PINN Tesseract VJP. Presentation is
    delegated to ``callback``.

    Args:
        config: a validated ``RunConfig``.
        pinn: an already-open Tesseract. If ``None``, one is created from the
            backend image and managed for the duration of the call.
        callback: optional ``TrainingCallback`` for progress/visualization.
        metrics_every: cadence (in epochs) for computing full loss components.
    """
    strategy = PINNStrategy(config, pinn=pinn)
    return _run_inverse_training(
        config, strategy, callback=callback, metrics_every=metrics_every
    )


def train_solver_inverse(config, *, solver=None, callback=None, metrics_every=20):
    """Run solver-adjoint inversion: optimize ``log_nu`` against the in-loop solver.

    Thin wrapper over the shared engine with a :class:`SolverAdjointStrategy`.
    One reverse-mode pass per step differentiates the data-fit loss through the
    solver Tesseract VJP. Observations come from the same viscous-Burgers physics
    (clean, well-posed inverse), so the estimate recovers ``nu`` up to noise.
    """
    strategy = SolverAdjointStrategy(config, solver=solver)
    return _run_inverse_training(
        config, strategy, callback=callback, metrics_every=metrics_every
    )


def make_inverse_strategy(config, mode, *, pinn=None, solver=None):
    """Factory: build the :class:`InverseStrategy` for an inverse ``mode``.

    ``mode`` is one of ``"pinn"``, ``"solver-inverse"``, or ``"hybrid"``. Already
    -open Tesseracts may be injected via ``pinn``/``solver``; otherwise the
    strategy opens and manages its own.
    """
    if mode == "pinn":
        return PINNStrategy(config, pinn=pinn)
    if mode == "solver-inverse":
        return SolverAdjointStrategy(config, solver=solver)
    if mode == "hybrid":
        from burgers_inverse.experimental import HybridDiscrepancyStrategy

        return HybridDiscrepancyStrategy(config, solver=solver, pinn=pinn)
    raise ValueError(f"Unknown inverse mode: {mode!r}")
