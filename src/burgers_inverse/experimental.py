"""Experimental misspecification sidebar: KdV-Burgers truth + discrepancy hybrid.

A *documented negative result*, not a headline feature. The in-loop simulator
(the ``burgers_solver`` Tesseract) solves plain viscous Burgers; the truth here is
generated from the structurally richer KdV-Burgers equation
``u_t + u u_x = nu u_xx - beta u_xxx``. The dispersive ``-beta u_xxx`` term is
outside the simulator's reachable family for any ``nu``, so the discrepancy the
hybrid learns is irreducible. L2-minimizing a free-form discrepancy is confounded
with the calibration parameter and biases ``nu`` (Brynjarsdóttir & O'Hagan, 2014).
Kept to demonstrate the failure mode and motivate the posterior treatment.
"""

from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
import optax
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

from burgers_inverse.components import (
    _solver_field,
    get_initial_params,
    image_name_for_backend,
)
from burgers_inverse.configs import DEFAULT_NOISE_STD
from burgers_inverse.constants import MIN_OBS_TIME, SOLVER_NT, SOLVER_NX
from burgers_inverse.engine import (
    InverseStrategy,
    StepResult,
    _run_inverse_training,
)
from burgers_inverse.observations import GridObservations, _sample_grid_indices

_KDV_DT0 = 1e-3
_KDV_RTOL = 1e-6
_KDV_ATOL = 1e-6
_KDV_MAX_STEPS = 200_000


def _kdv_burgers_rhs(u, nu, beta, x_grid):
    """Spectral RHS for KdV-Burgers: -u u_x + nu u_xx - beta u_xxx (dealiased)."""
    nx = u.shape[0]
    dx = x_grid[1] - x_grid[0]
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(nx) / dx
    u_hat = jnp.fft.fft(u)
    u_x = jnp.fft.ifft(1j * k * u_hat).real
    u_xx = jnp.fft.ifft(-(k**2) * u_hat).real
    u_xxx = jnp.fft.ifft(-1j * (k**3) * u_hat).real  # (i k)^3 = -i k^3

    # 2/3-rule dealiasing of the nonlinear product, matching the solver Tesseract.
    mode_numbers = jnp.fft.fftfreq(nx) * nx
    keep = jnp.abs(mode_numbers) <= (nx // 3)
    nonlinear = jnp.fft.ifft(jnp.fft.fft(u * u_x) * keep).real

    return -nonlinear + nu * u_xx - beta * u_xxx


def solve_kdv_burgers(nu, beta, x_grid, t_grid, ic_amp=1.0, ic_phase=0.0):
    """High-fidelity KdV-Burgers truth oracle.

    Solves `u_t + u u_x = nu u_xx - beta u_xxx` spectrally with the same periodic
    sinusoidal IC and dealiasing as the in-loop `burgers_solver` Tesseract, plus the
    extra dispersive term the simulator omits. With ``beta == 0`` it reproduces the
    plain viscous-Burgers field (discrepancy collapses to zero).
    """
    nu = jnp.asarray(nu, dtype=jnp.float32)
    beta = jnp.asarray(beta, dtype=jnp.float32)
    x_grid = jnp.asarray(x_grid, dtype=jnp.float32)
    t_grid = jnp.asarray(t_grid, dtype=jnp.float32)
    u0 = jnp.asarray(ic_amp, dtype=jnp.float32) * jnp.sin(
        2.0 * jnp.pi * x_grid + jnp.asarray(ic_phase, dtype=jnp.float32)
    )

    def vector_field(_, u, args):
        nu_value, beta_value, grid = args
        return _kdv_burgers_rhs(u, nu_value, beta_value, grid)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Tsit5(),
        t0=t_grid[0],
        t1=t_grid[-1],
        dt0=_KDV_DT0,
        y0=u0,
        args=(nu, beta, x_grid),
        saveat=diffrax.SaveAt(ts=t_grid),
        stepsize_controller=diffrax.PIDController(rtol=_KDV_RTOL, atol=_KDV_ATOL),
        max_steps=_KDV_MAX_STEPS,
    )
    return sol.ys


def generate_kdv_observations(
    n_points, true_viscosity, dispersion_beta, domain, key, noise_std=DEFAULT_NOISE_STD
):
    """Sample sparse noisy observations from the KdV-Burgers truth at grid nodes.

    Mirrors :func:`burgers_inverse.observations.generate_observations`' grid, RNG
    split, time floor, and noise model so the hybrid mode's data is directly
    comparable to the other modes, but also returns the grid and the sampled
    `(t_idx, x_idx)` so the in-loop solver field can be gathered at the same nodes
    inside the differentiated objective.
    """
    nx, nt = SOLVER_NX, SOLVER_NT
    key_idx, key_noise = jax.random.split(key, 2)

    x_grid = jnp.linspace(
        domain["x"][0], domain["x"][1], nx, endpoint=False, dtype=jnp.float32
    )
    t_grid = jnp.linspace(domain["t"][0], domain["t"][1], nt, dtype=jnp.float32)

    u_field = solve_kdv_burgers(true_viscosity, dispersion_beta, x_grid, t_grid)

    x_idx, t_idx = _sample_grid_indices(key_idx, n_points, x_grid, t_grid)
    noise = jax.random.normal(key_noise, (n_points,)) * noise_std

    return GridObservations(
        x_obs=x_grid[x_idx],
        t_obs=t_grid[t_idx],
        u_obs=u_field[t_idx, x_idx] + noise,
        x_grid=x_grid,
        t_grid=t_grid,
        x_idx=x_idx,
        t_idx=t_idx,
    )


def hybrid_discrepancy_loss(
    log_viscosity,
    params_flat,
    obs,
    x_reg,
    t_reg,
    solver,
    pinn,
    w_data,
    w_reg,
    w_smooth,
):
    """Composed hybrid objective `u_model = solver(nu) + delta`.

    - Data term fits `solver(nu)[obs] + delta(obs)` to the sparse KdV observations.
    - L2 (and optional smoothness) regularization keeps `delta` small so `nu` stays
      identifiable.

    Reverse-mode differentiation composes two Tesseract VJPs in one pass:
    `dL/d(log nu)` through the JAX solver, `dL/d(params)` through the discrepancy
    network (JAX or PyTorch). Returns `(total, components)` for `has_aux`.
    """
    nu = jnp.exp(log_viscosity)

    u_field = _solver_field(solver, nu, obs.x_grid, obs.t_grid)
    u_solver_obs = u_field[obs.t_idx, obs.x_idx]
    delta_obs = apply_tesseract(
        pinn, {"x": obs.x_obs, "t": obs.t_obs, "params_flat": params_flat}
    )["u_pred"]
    u_model = u_solver_obs + delta_obs
    data_loss = jnp.mean((u_model - obs.u_obs) ** 2)

    reg_out = apply_tesseract(
        pinn, {"x": x_reg, "t": t_reg, "params_flat": params_flat}
    )
    reg_loss = jnp.mean(reg_out["u_pred"] ** 2)
    smooth_loss = jnp.mean(reg_out["u_x"] ** 2)

    total = w_data * data_loss + w_reg * reg_loss + w_smooth * smooth_loss
    components = {
        "total": total,
        "data": data_loss,
        "reg": reg_loss,
        "smooth": smooth_loss,
    }
    return total, components


class HybridDiscrepancyStrategy(InverseStrategy):
    """Hybrid calibration-with-discrepancy (Stage-1 sidebar): jointly optimize
    ``log_nu`` and a discrepancy network, composing the solver and PINN Tesseract
    VJPs in one reverse-mode pass. A documented confounding *negative result*."""

    loss_component_names = ("total", "data", "reg", "smooth")

    def __init__(self, config, *, solver=None, pinn=None):
        self.config = config
        self.backend = config.backend
        self.problem = config.problem
        self.data_config = config.data
        self.training = config.training
        self.w_data = float(config.loss.data)
        self.w_reg = float(self.training.discrepancy_reg_weight)
        self.w_smooth = float(self.training.discrepancy_smooth_weight)
        self.effective_weights = {
            "data": self.w_data,
            "reg": self.w_reg,
            "smooth": self.w_smooth,
        }
        domain = self.problem.domain
        key = jax.random.PRNGKey(self.data_config.seed)
        key_obs, key_reg_x, key_reg_t = jax.random.split(key, 3)
        self.obs = generate_kdv_observations(
            self.data_config.n_obs,
            self.problem.true_viscosity,
            self.problem.dispersion_beta,
            domain,
            key_obs,
            noise_std=self.data_config.noise_std,
        )
        self.x_reg = jax.random.uniform(
            key_reg_x,
            (self.training.n_col,),
            minval=domain["x"][0],
            maxval=domain["x"][1],
        )
        self.t_reg = jax.random.uniform(
            key_reg_t,
            (self.training.n_col,),
            minval=MIN_OBS_TIME,
            maxval=domain["t"][1],
        )

        self.image_name = image_name_for_backend(self.backend)
        self.owns_pinn = pinn is None
        self.owns_solver = solver is None
        self.pinn = pinn if pinn is not None else Tesseract.from_image(self.image_name)
        self.solver = (
            solver if solver is not None else Tesseract.from_image("burgers_solver")
        )

        self.params_flat = get_initial_params(self.backend, seed=self.data_config.seed)
        self.param_optimizer = optax.adam(self.training.param_learning_rate)
        self.param_opt_state = self.param_optimizer.init(self.params_flat)
        self.loss_and_grads = jax.value_and_grad(
            hybrid_discrepancy_loss, argnums=(0, 1), has_aux=True
        )

    def open_components(self, stack):
        if self.owns_pinn:
            stack.enter_context(self.pinn)
        if self.owns_solver:
            stack.enter_context(self.solver)

    def start_context(self, warmup_epochs):
        return {
            "config": self.config,
            "backend": self.backend,
            "image_name": self.image_name,
            "solver_image": "burgers_solver",
            "warmup_epochs": warmup_epochs,
            "observations": self.obs,
        }

    def run_step(self, log_viscosity, epoch):
        (loss, components), (log_v_grad, p_grad) = self.loss_and_grads(
            log_viscosity,
            self.params_flat,
            self.obs,
            self.x_reg,
            self.t_reg,
            self.solver,
            self.pinn,
            self.w_data,
            self.w_reg,
            self.w_smooth,
        )

        param_updates, self.param_opt_state = self.param_optimizer.update(
            p_grad, self.param_opt_state
        )
        self.params_flat = optax.apply_updates(self.params_flat, param_updates)

        return StepResult(
            loss=float(loss),
            log_v_grad=log_v_grad,
            param_grad_norm=float(jnp.linalg.norm(p_grad)),
            effective_weights=self.effective_weights,
            param_count=int(self.params_flat.size),
            aux={name: float(value) for name, value in components.items()},
        )

    def epoch_loss_components(self, viscosity, log_viscosity, step, record):
        # Components come from the same backward pass every epoch (cheap aux).
        return step.aux

    def finalize_result(self, base):
        return {
            "mode": "hybrid",
            "backend": self.backend,
            "tesseract_image": self.image_name,
            "solver_image": "burgers_solver",
            **base,
            "dispersion_beta": self.problem.dispersion_beta,
            "effective_weights": self.effective_weights,
            "params_flat": self.params_flat,
            "observations": self.obs,
            "pinn": self.pinn,
            "solver": self.solver,
        }


def train_hybrid_inverse(
    config, *, solver=None, pinn=None, callback=None, metrics_every=20
):
    """Run the hybrid calibration-with-discrepancy loop (Stage 1).

    Thin wrapper over the shared engine with a :class:`HybridDiscrepancyStrategy`:
    one reverse-mode `value_and_grad` over both ``log_nu`` and the discrepancy
    network composes the solver and discrepancy Tesseract VJPs each step.
    """
    strategy = HybridDiscrepancyStrategy(config, solver=solver, pinn=pinn)
    return _run_inverse_training(
        config, strategy, callback=callback, metrics_every=metrics_every
    )
