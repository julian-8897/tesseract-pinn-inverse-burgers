"""Inverse problem demo for viscosity inference in Burgers equation.

Demonstrates cross-framework automatic differentiation via Tesseract:
- Same optimization code runs with JAX or PyTorch PINN backends
- JAX gradients computed through PyTorch models via VJP endpoint
- Backend selection controlled by Tesseract image name

Problem: Given noisy observations u(x,t), infer viscosity parameter ν
in Burgers equation: ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
"""

import csv
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from statistics import fmean, pstdev
from typing import NamedTuple

import diffrax
import jax
import jax.numpy as jnp
import optax
import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

from configs import (
    DEFAULT_LOSS_WEIGHTS,
    DEFAULT_NOISE_STD,
    LOSS_WEIGHT_NAMES,
    LossWeights,
    RunConfig,
    loss_weights_from_mapping,
    normalize_loss_weights,
)

REPO_ROOT = Path(__file__).resolve().parent
CONSOLE = Console()

# Solver discretization grid. Shared by the forward solver, the observation
# samplers, and the FMPE sensor layout — named generically rather than after any
# one experiment so cross-module imports read honestly.
SOLVER_NX = 128
SOLVER_NT = 64
# Smallest observation/collocation time. Sensors and collocation points avoid
# t≈0, where the initial condition dominates and the inverse signal is weak.
MIN_OBS_TIME = 0.05


def format_loss_weights(loss_weights):
    """Format loss weights for compact CLI output."""
    return ", ".join(f"{name}={loss_weights[name]:g}" for name in LOSS_WEIGHT_NAMES)


def loss_weights_to_array(loss_weights):
    """Return loss weights as a JAX array in canonical component order."""
    weights = normalize_loss_weights(loss_weights)
    return jnp.asarray([weights[name] for name in LOSS_WEIGHT_NAMES])


def validate_brdr_loss_weights(loss_weights):
    """BRDR uses fixed coefficients as positive component scaling factors."""
    weights = loss_weights_to_array(loss_weights)
    if bool(jnp.any(weights <= 0)):
        raise ValueError("BRDR adaptive loss weights require positive fixed weights")


def initialize_brdr_state(pointwise_losses):
    """Initialize pointwise BRDR moments and weights."""
    return {
        "step": 0,
        "moment": {
            name: jnp.zeros_like(losses) for name, losses in pointwise_losses.items()
        },
        "weights": {
            name: jnp.ones_like(losses) for name, losses in pointwise_losses.items()
        },
    }


def update_brdr_state(state, pointwise_losses, beta_c=0.9999, beta_w=0.999, eps=1e-12):
    """Update BRDR state from pointwise squared residual losses.

    BRDR computes inverse residual decay rates as loss / sqrt(EMA(loss^2)),
    normalizes them to unit global mean, and smooths the resulting pointwise
    weights with an EMA.
    """
    step = state["step"] + 1
    corrected_irdr = {}
    new_moment = {}

    bias_correction = 1.0 - beta_c**step
    for name, losses in pointwise_losses.items():
        losses = jnp.asarray(losses)
        moment = beta_c * state["moment"][name] + (1.0 - beta_c) * losses**2
        new_moment[name] = moment
        corrected_irdr[name] = losses / jnp.sqrt(moment / bias_correction + eps)

    all_irdr = jnp.concatenate(
        [jnp.ravel(corrected_irdr[name]) for name in LOSS_WEIGHT_NAMES]
    )
    mean_irdr = jnp.mean(all_irdr)

    new_weights = {}
    for name in LOSS_WEIGHT_NAMES:
        target_weight = corrected_irdr[name] / (mean_irdr + eps)
        weight = beta_w * state["weights"][name] + (1.0 - beta_w) * target_weight
        new_weights[name] = weight

    return {
        "step": step,
        "moment": new_moment,
        "weights": new_weights,
    }


def summarize_brdr_weights(brdr_weights):
    """Return mean BRDR weight by component for logging and plotting."""
    return {name: float(jnp.mean(brdr_weights[name])) for name in LOSS_WEIGHT_NAMES}


def build_run_config(
    config=None,
    *,
    backend=None,
    true_viscosity=None,
    initial_viscosity=None,
    n_obs=None,
    n_epochs=None,
    learning_rate=None,
    param_learning_rate=None,
    adaptive_loss_weights=None,
    brdr_beta_c=None,
    brdr_beta_w=None,
    brdr_epsilon=None,
    loss_weights=None,
    seed=None,
    noise_std=None,
    n_col=None,
    n_ic=None,
    n_bc=None,
):
    """Build a RunConfig from defaults plus legacy keyword overrides."""
    config = RunConfig() if config is None else config

    if backend is not None:
        config = replace(config, backend=backend)

    problem_updates = {}
    if true_viscosity is not None:
        problem_updates["true_viscosity"] = true_viscosity
    if initial_viscosity is not None:
        problem_updates["initial_viscosity"] = initial_viscosity
    if problem_updates:
        config = replace(config, problem=replace(config.problem, **problem_updates))

    data_updates = {}
    if n_obs is not None:
        data_updates["n_obs"] = n_obs
    if seed is not None:
        data_updates["seed"] = seed
    if noise_std is not None:
        data_updates["noise_std"] = noise_std
    if data_updates:
        config = replace(config, data=replace(config.data, **data_updates))

    training_updates = {}
    if n_epochs is not None:
        training_updates["n_epochs"] = n_epochs
    if learning_rate is not None:
        training_updates["log_nu_learning_rate"] = learning_rate
    if param_learning_rate is not None:
        training_updates["param_learning_rate"] = param_learning_rate
    if adaptive_loss_weights is not None:
        training_updates["adaptive_loss_weights"] = adaptive_loss_weights
    if brdr_beta_c is not None:
        training_updates["brdr_beta_c"] = brdr_beta_c
    if brdr_beta_w is not None:
        training_updates["brdr_beta_w"] = brdr_beta_w
    if brdr_epsilon is not None:
        training_updates["brdr_epsilon"] = brdr_epsilon
    if n_col is not None:
        training_updates["n_col"] = n_col
    if n_ic is not None:
        training_updates["n_ic"] = n_ic
    if n_bc is not None:
        training_updates["n_bc"] = n_bc
    if training_updates:
        config = replace(config, training=replace(config.training, **training_updates))

    if loss_weights is not None:
        config = replace(config, loss=loss_weights_from_mapping(loss_weights))
    elif not isinstance(config.loss, LossWeights):
        config = replace(config, loss=loss_weights_from_mapping(config.loss))

    return config


def log_run_header(config):
    """Log the inverse-problem run configuration."""
    loss_weights = config.loss.as_dict()
    CONSOLE.rule(f"[bold cyan]Inverse Problem: {config.backend.upper()} PINN")
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Parameter", style="bold")
    table.add_column("Value", style="cyan")
    table.add_row("True viscosity", f"ν = {config.problem.true_viscosity:.6f}")
    table.add_row("Initial guess", f"ν = {config.problem.initial_viscosity:.6f}")
    table.add_row("Loss weights", format_loss_weights(loss_weights))
    table.add_row(
        "BRDR weights",
        "yes" if config.training.adaptive_loss_weights else "no",
    )
    table.add_row("Seed", str(config.data.seed))
    CONSOLE.print(table)


def make_training_progress():
    """Create a compact progress display for inverse training."""
    return Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("loss={task.fields[loss]}"),
        TextColumn("ν={task.fields[nu]}"),
        TextColumn("err={task.fields[error]}"),
        TextColumn("epoch={task.fields[epoch_time]}"),
        console=CONSOLE,
    )


def log_final_results(
    final_viscosity,
    true_viscosity,
    relative_error,
    avg_time,
    loss_history,
    loss_weight_history=None,
):
    """Log final scalar results and final loss components."""
    table = Table(title="Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right", style="cyan")
    table.add_row("Inferred ν", f"{final_viscosity:.6f}")
    table.add_row("True ν", f"{true_viscosity:.6f}")
    table.add_row("Relative error", f"{relative_error:.2f}%")
    table.add_row("Avg time/epoch", f"{avg_time:.1f} ms")

    if loss_history["total"]:
        table.add_section()
        table.add_row("total loss", f"{loss_history['total'][-1]:.6e}")
        table.add_row("data loss", f"{loss_history['data'][-1]:.6e}")
        table.add_row("physics loss", f"{loss_history['physics'][-1]:.6e}")
        table.add_row("IC loss", f"{loss_history['ic'][-1]:.6e}")
        table.add_row("BC loss", f"{loss_history['bc'][-1]:.6e}")

    if loss_weight_history:
        table.add_section()
        for name in LOSS_WEIGHT_NAMES:
            table.add_row(f"{name} weight", f"{loss_weight_history[name][-1]:.6g}")

    CONSOLE.print(table)


def log_backend_comparison(results):
    """Log a compact JAX vs PyTorch comparison table."""
    table = Table(title="JAX vs PyTorch PINN")
    table.add_column("Metric", style="bold")
    table.add_column("JAX", justify="right", style="cyan")
    table.add_column("PyTorch", justify="right", style="magenta")
    table.add_row(
        "Inferred viscosity",
        f"{results['jax']['final_viscosity']:.6f}",
        f"{results['pytorch']['final_viscosity']:.6f}",
    )
    table.add_row(
        "Relative error (%)",
        f"{results['jax']['relative_error']:.2f}",
        f"{results['pytorch']['relative_error']:.2f}",
    )
    table.add_row(
        "Avg time/epoch (ms)",
        f"{results['jax']['avg_time_ms']:.1f}",
        f"{results['pytorch']['avg_time_ms']:.1f}",
    )
    CONSOLE.print(table)


def summarize_seed_results(results):
    """Aggregate repeated inverse runs by backend."""
    summary = {}
    backends = sorted({result["backend"] for result in results})

    for backend in backends:
        backend_results = [result for result in results if result["backend"] == backend]
        viscosities = [result["final_viscosity"] for result in backend_results]
        errors = [result["relative_error"] for result in backend_results]
        times = [result["avg_time_ms"] for result in backend_results]
        summary[backend] = {
            "runs": len(backend_results),
            "mean_viscosity": fmean(viscosities),
            "std_viscosity": pstdev(viscosities),
            "mean_relative_error": fmean(errors),
            "std_relative_error": pstdev(errors),
            "mean_time_ms": fmean(times),
        }

    return summary


def log_seed_summary(results):
    """Log mean/std metrics for a seed sweep."""
    summary = summarize_seed_results(results)
    table = Table(title="Seed Sweep Summary")
    table.add_column("Backend", style="bold")
    table.add_column("Runs", justify="right")
    table.add_column("Mean ν", justify="right", style="cyan")
    table.add_column("Std ν", justify="right")
    table.add_column("Mean error (%)", justify="right", style="cyan")
    table.add_column("Std error (%)", justify="right")
    table.add_column("Mean time/epoch (ms)", justify="right")

    for backend, values in summary.items():
        table.add_row(
            backend,
            str(values["runs"]),
            f"{values['mean_viscosity']:.6f}",
            f"{values['std_viscosity']:.6f}",
            f"{values['mean_relative_error']:.2f}",
            f"{values['std_relative_error']:.2f}",
            f"{values['mean_time_ms']:.1f}",
        )

    CONSOLE.print(table)


def get_burgers_solver():
    """Import the solver without leaving a conflicting tesseract_api module loaded."""
    solver_path = str(REPO_ROOT / "tesseracts" / "burgers_solver")
    previous_module = sys.modules.pop("tesseract_api", None)

    sys.path.insert(0, solver_path)
    try:
        from tesseract_api import solve_burgers
    finally:
        sys.path.pop(0)
        if "tesseract_api" in sys.modules:
            del sys.modules["tesseract_api"]
        if previous_module is not None:
            sys.modules["tesseract_api"] = previous_module

    return solve_burgers


def get_initial_params(backend="jax", seed=42):
    """Get initial parameters for the specified backend."""
    backend_path = REPO_ROOT / "tesseracts" / f"pinn_{backend}"
    previous_module = sys.modules.pop("tesseract_api", None)
    sys.path.insert(0, str(backend_path))

    if backend == "jax":
        key = jax.random.PRNGKey(seed)
    else:  # pytorch
        torch.manual_seed(seed)

    try:
        from tesseract_api import PINNNet, flatten_params

        if backend == "jax":
            model = PINNNet(key)
        else:
            # For PyTorch, initialize from actual model for proper initialization.
            model = PINNNet(hidden_sizes=[64, 64, 64], n_fourier_features=32, seed=seed)
        return jnp.array(flatten_params(model))
    finally:
        sys.path.pop(0)
        if "tesseract_api" in sys.modules:
            del sys.modules["tesseract_api"]
        if previous_module is not None:
            sys.modules["tesseract_api"] = previous_module


def generate_observations(
    n_points, true_viscosity, domain, key, noise_std=DEFAULT_NOISE_STD
):
    """
    Generate synthetic observations from the pseudospectral Burgers solver.

    The solver uses the same sinusoidal initial condition assumed by the PINN
    initial-condition loss: u(x, 0) = sin(2πx).
    """
    nx = SOLVER_NX
    nt = SOLVER_NT
    keys = jax.random.split(key, 3)

    x_grid = jnp.linspace(
        domain["x"][0], domain["x"][1], nx, endpoint=False, dtype=jnp.float32
    )
    t_grid = jnp.linspace(domain["t"][0], domain["t"][1], nt, dtype=jnp.float32)

    solve_burgers = get_burgers_solver()
    u_field = solve_burgers(
        jnp.asarray(true_viscosity, dtype=jnp.float32),
        x_grid,
        t_grid,
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
    )

    x_idx = jax.random.randint(keys[0], (n_points,), minval=0, maxval=nx)
    min_t_idx = max(1, int(jnp.searchsorted(t_grid, MIN_OBS_TIME, side="left")))
    t_idx = jax.random.randint(keys[1], (n_points,), minval=min_t_idx, maxval=nt)

    x = x_grid[x_idx]
    t = t_grid[t_idx]
    u_observed = u_field[t_idx, x_idx]

    # Add small noise
    noise = jax.random.normal(keys[2], (n_points,)) * noise_std
    u_observed = u_observed + noise

    return x, t, u_observed


#
# Hybrid Stage 1: KdV-Burgers truth oracle + discrepancy calibration
#
# The in-loop simulator (the `burgers_solver` Tesseract) solves plain viscous
# Burgers. The *truth* is generated here from the structurally richer KdV-Burgers
# equation `u_t + u u_x = nu u_xx - beta u_xxx`. The dispersive `-beta u_xxx` term
# is outside the simulator's reachable family for any `nu`, so the discrepancy the
# PINN learns is irreducible (there is genuinely something to correct). This oracle
# is a local JAX function, not a Tesseract, because it is never differentiated.
#

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


class GridObservations(NamedTuple):
    """Sparse observations plus the grid/indices the in-loop solver field is
    gathered at. Shared by ``--mode solver-inverse`` (clean Burgers truth) and the
    experimental KdV discrepancy sidebar."""

    x_obs: jax.Array
    t_obs: jax.Array
    u_obs: jax.Array
    x_grid: jax.Array
    t_grid: jax.Array
    x_idx: jax.Array
    t_idx: jax.Array


def _sample_grid_indices(key, n_points, x_grid, t_grid):
    """Sample sparse grid-node indices, with a small time floor (t >= 0.05)."""
    keys = jax.random.split(key, 2)
    nx, nt = x_grid.shape[0], t_grid.shape[0]
    x_idx = jax.random.randint(keys[0], (n_points,), minval=0, maxval=nx)
    min_t_idx = max(1, int(jnp.searchsorted(t_grid, MIN_OBS_TIME, side="left")))
    t_idx = jax.random.randint(keys[1], (n_points,), minval=min_t_idx, maxval=nt)
    return x_idx, t_idx


def generate_grid_observations(
    n_points, true_viscosity, domain, key, noise_std=DEFAULT_NOISE_STD
):
    """Sparse noisy observations from the in-loop viscous-Burgers solver itself.

    The truth here is the *same* physics the ``solver-inverse`` mode optimizes
    against (plain viscous Burgers), so the inverse problem is well-posed and the
    solver-adjoint estimate recovers ``nu`` up to noise. Returns the grid and the
    sampled `(t_idx, x_idx)` so the in-loop solver field can be gathered at the same
    nodes inside the differentiated objective.
    """
    nx, nt = SOLVER_NX, SOLVER_NT
    key_idx, key_noise = jax.random.split(key, 2)

    x_grid = jnp.linspace(
        domain["x"][0], domain["x"][1], nx, endpoint=False, dtype=jnp.float32
    )
    t_grid = jnp.linspace(domain["t"][0], domain["t"][1], nt, dtype=jnp.float32)

    solve_burgers = get_burgers_solver()
    u_field = solve_burgers(
        jnp.asarray(true_viscosity, dtype=jnp.float32),
        x_grid,
        t_grid,
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
    )

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


def generate_kdv_observations(
    n_points, true_viscosity, dispersion_beta, domain, key, noise_std=DEFAULT_NOISE_STD
):
    """Sample sparse noisy observations from the KdV-Burgers truth at grid nodes.

    Mirrors :func:`generate_observations`' grid, RNG split, time floor, and noise
    model so the hybrid mode's data is directly comparable to the other modes, but
    also returns the grid and the sampled `(t_idx, x_idx)` so the in-loop solver
    field can be gathered at the same nodes inside the differentiated objective.
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


def compute_pointwise_losses(
    viscosity,
    params_flat,
    x_obs,
    t_obs,
    u_obs,
    x_col,
    t_col,
    x_ic,
    t_bc,
    pinn,
):
    """Compute pointwise squared residual losses for each PINN constraint."""
    result_obs = apply_tesseract(
        pinn,
        {
            "x": x_obs,
            "t": t_obs,
            "params_flat": params_flat,
        },
    )
    u_pred = result_obs["u_pred"]
    data_losses = (u_pred - u_obs) ** 2

    result_col = apply_tesseract(
        pinn, {"x": x_col, "t": t_col, "params_flat": params_flat}
    )

    u_col = result_col["u_pred"]
    u_x = result_col["u_x"]
    u_t = result_col["u_t"]
    u_xx = result_col["u_xx"]

    residual = u_t + u_col * u_x - viscosity * u_xx
    physics_losses = residual**2

    t_ic = jnp.zeros_like(x_ic)
    result_ic = apply_tesseract(
        pinn,
        {
            "x": x_ic,
            "t": t_ic,
            "params_flat": params_flat,
        },
    )
    u_ic = result_ic["u_pred"]
    u_ic_true = jnp.sin(2 * jnp.pi * x_ic)
    ic_losses = (u_ic - u_ic_true) ** 2

    x_left = jnp.zeros_like(t_bc)
    x_right = jnp.ones_like(t_bc)

    result_left = apply_tesseract(
        pinn,
        {
            "x": x_left,
            "t": t_bc,
            "params_flat": params_flat,
        },
    )
    result_right = apply_tesseract(
        pinn,
        {
            "x": x_right,
            "t": t_bc,
            "params_flat": params_flat,
        },
    )
    u_left = result_left["u_pred"]
    u_right = result_right["u_pred"]
    bc_losses = (u_left - u_right) ** 2

    return {
        "data": data_losses,
        "physics": physics_losses,
        "ic": ic_losses,
        "bc": bc_losses,
    }


def compute_loss_components(
    viscosity,
    params_flat,
    x_obs,
    t_obs,
    u_obs,
    x_col,
    t_col,
    x_ic,
    t_bc,
    pinn,
    brdr_weights=None,
    loss_weights=None,
):
    """Compute total and component PINN losses for the inverse problem.

    Components:
    1. Data loss: fit observations
    2. Physics loss: satisfy PDE residual
    3. Initial condition loss: u(x, 0) = sin(2πx)
    4. Boundary condition loss: periodic BCs u(0,t) = u(1,t)

    All terms are differentiable with respect to viscosity.
    """
    weights = normalize_loss_weights(loss_weights)
    pointwise_losses = compute_pointwise_losses(
        viscosity,
        params_flat,
        x_obs,
        t_obs,
        u_obs,
        x_col,
        t_col,
        x_ic,
        t_bc,
        pinn,
    )

    data_loss = jnp.mean(pointwise_losses["data"])
    physics_loss = jnp.mean(pointwise_losses["physics"])
    ic_loss = jnp.mean(pointwise_losses["ic"])
    bc_loss = jnp.mean(pointwise_losses["bc"])

    raw_losses = jnp.asarray([data_loss, physics_loss, ic_loss, bc_loss])
    if brdr_weights is None:
        weight_array = loss_weights_to_array(weights)
        total_loss = jnp.sum(weight_array * raw_losses)
    else:
        total_loss = sum(
            weights[name] * jnp.mean(brdr_weights[name] * pointwise_losses[name])
            for name in LOSS_WEIGHT_NAMES
        )

    return {
        "total": total_loss,
        "data": data_loss,
        "physics": physics_loss,
        "ic": ic_loss,
        "bc": bc_loss,
    }


def compute_loss(
    viscosity,
    params_flat,
    x_obs,
    t_obs,
    u_obs,
    x_col,
    t_col,
    x_ic,
    t_bc,
    pinn,
    brdr_weights=None,
    loss_weights=None,
):
    """Compute scalar total PINN loss for gradient-based optimization."""
    components = compute_loss_components(
        viscosity,
        params_flat,
        x_obs,
        t_obs,
        u_obs,
        x_col,
        t_col,
        x_ic,
        t_bc,
        pinn,
        brdr_weights=brdr_weights,
        loss_weights=loss_weights,
    )

    return components["total"]


def compute_loss_from_log_viscosity(
    log_viscosity,
    params_flat,
    x_obs,
    t_obs,
    u_obs,
    x_col,
    t_col,
    x_ic,
    t_bc,
    pinn,
    brdr_weights=None,
    loss_weights=None,
):
    """Compute loss while optimizing viscosity in log-space."""
    viscosity = jnp.exp(log_viscosity)
    return compute_loss(
        viscosity,
        params_flat,
        x_obs,
        t_obs,
        u_obs,
        x_col,
        t_col,
        x_ic,
        t_bc,
        pinn,
        brdr_weights=brdr_weights,
        loss_weights=loss_weights,
    )


#
# Shared training engine
#
# A single `train_inverse` loop powers both the CLI (`inverse_problem.py`) and
# the Streamlit app (`app.py`). Presentation lives in callbacks so neither
# frontend reimplements the optimization.
#


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


def image_name_for_backend(backend):
    """Map a backend name to its Tesseract image."""
    return "pinn_jax" if backend == "jax" else "pinn_pytorch"


def docker_image_available(image_name):
    """Return whether a local Tesseract Docker image exists."""
    import subprocess

    try:
        result = subprocess.run(
            ["docker", "inspect", image_name, "--type", "image"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0


class TesseractImageNotFoundError(RuntimeError):
    """Raised when a required Tesseract container image is not built locally."""


def ensure_image_available(image_name):
    """Fail early with an actionable message if the container image is missing."""
    if not docker_image_available(image_name):
        raise TesseractImageNotFoundError(
            f"Tesseract image '{image_name}' was not found. "
            "Build the containers first with ./buildall.sh (Docker required)."
        )


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


def _loss_from_log_and_params(
    log_viscosity,
    params_flat,
    x_obs,
    t_obs,
    u_obs,
    x_col,
    t_col,
    x_ic,
    t_bc,
    pinn,
    brdr_weights,
    loss_weights,
):
    """Scalar loss as a function of both optimized variables for value_and_grad."""
    return compute_loss(
        jnp.exp(log_viscosity),
        params_flat,
        x_obs,
        t_obs,
        u_obs,
        x_col,
        t_col,
        x_ic,
        t_bc,
        pinn,
        brdr_weights=brdr_weights,
        loss_weights=loss_weights,
    )


def train_inverse(config, *, pinn=None, callback=None, metrics_every=20):
    """Run the inverse-viscosity optimization loop.

    A single reverse-mode `value_and_grad` over both ``log_nu`` and the PINN
    parameters replaces the previous two separate `jax.grad` passes plus a
    standalone loss evaluation, cutting Tesseract round-trips per step from three
    forward/backward sweeps to one. Presentation is delegated to ``callback``.

    Args:
        config: a validated ``RunConfig``.
        pinn: an already-open Tesseract. If ``None``, one is created from the
            backend image and managed for the duration of the call.
        callback: optional ``TrainingCallback`` for progress/visualization.
        metrics_every: cadence (in epochs) for computing full loss components.
    """
    backend = config.backend
    problem = config.problem
    data_config = config.data
    training = config.training
    loss_weights = config.loss.as_dict()

    if training.adaptive_loss_weights:
        validate_brdr_loss_weights(loss_weights)

    callback = callback or TrainingCallback()
    counter = TesseractCallCounter()

    x_obs, t_obs, u_obs, x_col, t_col, x_ic, t_bc = build_training_inputs(config)

    image_name = image_name_for_backend(backend)
    owns_pinn = pinn is None
    if owns_pinn:
        pinn = Tesseract.from_image(image_name)

    params_flat = get_initial_params(backend, seed=data_config.seed)

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
    param_optimizer = optax.adam(training.param_learning_rate)
    param_opt_state = param_optimizer.init(params_flat)

    loss_and_grads = jax.value_and_grad(_loss_from_log_and_params, argnums=(0, 1))

    viscosity = jnp.exp(log_viscosity)
    times = []
    viscosity_history = [float(viscosity)]
    log_viscosity_history = [float(log_viscosity)]
    loss_history = {name: [] for name in ("total", "data", "physics", "ic", "bc")}
    loss_weight_history = {name: [loss_weights[name]] for name in LOSS_WEIGHT_NAMES}
    brdr_state = None

    def _run():
        nonlocal log_viscosity, params_flat, viscosity
        nonlocal log_visc_opt_state, param_opt_state, brdr_state

        callback.on_start(
            {
                "config": config,
                "backend": backend,
                "image_name": image_name,
                "warmup_epochs": warmup_epochs,
                "x_obs": x_obs,
                "t_obs": t_obs,
                "u_obs": u_obs,
            }
        )

        for epoch in range(training.n_epochs):
            start_time = time.time()
            viscosity = jnp.exp(log_viscosity)

            brdr_weights = None
            if training.adaptive_loss_weights:
                pointwise_losses = compute_pointwise_losses(
                    viscosity,
                    params_flat,
                    x_obs,
                    t_obs,
                    u_obs,
                    x_col,
                    t_col,
                    x_ic,
                    t_bc,
                    pinn,
                )
                if brdr_state is None:
                    brdr_state = initialize_brdr_state(pointwise_losses)
                brdr_state = update_brdr_state(
                    brdr_state,
                    pointwise_losses,
                    beta_c=training.brdr_beta_c,
                    beta_w=training.brdr_beta_w,
                    eps=training.brdr_epsilon,
                )
                brdr_weights = brdr_state["weights"]

            # One reverse-mode sweep yields the loss and both gradients.
            counter.reset()
            with count_tesseract_calls(counter):
                loss, (log_v_grad, p_grad) = loss_and_grads(
                    log_viscosity,
                    params_flat,
                    x_obs,
                    t_obs,
                    u_obs,
                    x_col,
                    t_col,
                    x_ic,
                    t_bc,
                    pinn,
                    brdr_weights,
                    loss_weights,
                )
            apply_calls = counter.apply_calls
            vjp_calls = counter.vjp_calls

            visc_grad_norm = float(jnp.abs(log_v_grad))
            param_grad_norm = float(jnp.linalg.norm(p_grad))

            viscosity_updated = epoch >= warmup_epochs
            if viscosity_updated:
                log_visc_updates, log_visc_opt_state = log_visc_optimizer.update(
                    log_v_grad, log_visc_opt_state
                )
                log_viscosity = optax.apply_updates(log_viscosity, log_visc_updates)
                if log_nu_bounds is not None:
                    log_viscosity = jnp.clip(
                        log_viscosity, log_nu_bounds[0], log_nu_bounds[1]
                    )

            param_updates, param_opt_state = param_optimizer.update(
                p_grad, param_opt_state
            )
            params_flat = optax.apply_updates(params_flat, param_updates)
            viscosity = jnp.exp(log_viscosity)

            epoch_time = time.time() - start_time
            times.append(epoch_time)
            viscosity_history.append(float(viscosity))
            log_viscosity_history.append(float(log_viscosity))

            effective_weights = (
                dict(loss_weights)
                if brdr_weights is None
                else summarize_brdr_weights(brdr_weights)
            )
            for name in LOSS_WEIGHT_NAMES:
                loss_weight_history[name].append(effective_weights[name])

            loss_components = None
            if epoch % metrics_every == 0 or epoch == training.n_epochs - 1:
                loss_components = {
                    name: float(value)
                    for name, value in compute_loss_components(
                        viscosity,
                        params_flat,
                        x_obs,
                        t_obs,
                        u_obs,
                        x_col,
                        t_col,
                        x_ic,
                        t_bc,
                        pinn,
                        brdr_weights=brdr_weights,
                        loss_weights=loss_weights,
                    ).items()
                }
                for name, value in loss_components.items():
                    loss_history[name].append(value)

            callback.on_epoch(
                EpochRecord(
                    epoch=epoch,
                    n_epochs=training.n_epochs,
                    viscosity=float(viscosity),
                    log_viscosity=float(log_viscosity),
                    loss=float(loss),
                    visc_grad_norm=visc_grad_norm,
                    param_grad_norm=param_grad_norm,
                    epoch_time=epoch_time,
                    apply_calls=apply_calls,
                    vjp_calls=vjp_calls,
                    effective_weights=effective_weights,
                    viscosity_updated=viscosity_updated,
                    param_count=int(params_flat.size),
                    brdr_weights=brdr_weights,
                    loss_components=loss_components,
                )
            )

    if owns_pinn:
        with pinn:
            _run()
    else:
        _run()

    final_viscosity = float(viscosity)
    relative_error = (
        abs(final_viscosity - problem.true_viscosity) / problem.true_viscosity * 100
    )
    avg_time = sum(times) / len(times) * 1000 if times else 0.0

    result = {
        "backend": backend,
        "tesseract_image": image_name,
        "final_viscosity": final_viscosity,
        "true_viscosity": problem.true_viscosity,
        "relative_error": relative_error,
        "avg_time_ms": avg_time,
        "viscosity_history": viscosity_history,
        "log_viscosity_history": log_viscosity_history,
        "loss_history": loss_history,
        "loss_weights": loss_weights,
        "loss_weight_history": loss_weight_history,
        "brdr_state": brdr_state,
        "adaptive_loss_weights": training.adaptive_loss_weights,
        "seed": data_config.seed,
        "config": config,
        "params_flat": params_flat,
        "warmup_epochs": warmup_epochs,
        "observations": (x_obs, t_obs, u_obs),
        "pinn": pinn,
    }
    callback.on_finish(result)
    return result


#
# Hybrid Stage 1: composed solver + discrepancy objective and training engine
#


def _solver_field(solver, nu, x_grid, t_grid, ic_amp=1.0, ic_phase=0.0):
    """Run the in-loop viscous-Burgers solver Tesseract at the current parameters.

    Differentiating the loss w.r.t. any of ``nu``/``ic_amp``/``ic_phase`` triggers
    this Tesseract's VJP endpoint, so all three may be passed as traced values (e.g.
    Stage-C multi-parameter refinement). They default to the canonical
    ``u(x, 0) = sin(2πx)`` initial condition used by the single-parameter modes.
    """
    return apply_tesseract(
        solver,
        {
            "nu": nu,
            "x_grid": x_grid,
            "t_grid": t_grid,
            "ic_amp": jnp.asarray(ic_amp, dtype=jnp.float32),
            "ic_phase": jnp.asarray(ic_phase, dtype=jnp.float32),
        },
    )["u_field"]


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


def train_hybrid_inverse(
    config, *, solver=None, pinn=None, callback=None, metrics_every=20
):
    """Run the hybrid calibration-with-discrepancy loop (Stage 1).

    Jointly optimizes `log_nu` and the discrepancy network parameters with one
    reverse-mode `value_and_grad` over both, composing the solver and discrepancy
    Tesseract VJPs each step. The solver (JAX) is opened from `burgers_solver`; the
    discrepancy network from `pinn_{backend}` (JAX or PyTorch). Presentation is
    delegated to `callback`.
    """
    backend = config.backend
    problem = config.problem
    data_config = config.data
    training = config.training
    w_data = float(config.loss.data)
    w_reg = float(training.discrepancy_reg_weight)
    w_smooth = float(training.discrepancy_smooth_weight)

    callback = callback or TrainingCallback()
    counter = TesseractCallCounter()

    domain = problem.domain
    key = jax.random.PRNGKey(data_config.seed)
    key_obs, key_reg_x, key_reg_t = jax.random.split(key, 3)
    obs = generate_kdv_observations(
        data_config.n_obs,
        problem.true_viscosity,
        problem.dispersion_beta,
        domain,
        key_obs,
        noise_std=data_config.noise_std,
    )
    x_reg = jax.random.uniform(
        key_reg_x, (training.n_col,), minval=domain["x"][0], maxval=domain["x"][1]
    )
    t_reg = jax.random.uniform(
        key_reg_t, (training.n_col,), minval=MIN_OBS_TIME, maxval=domain["t"][1]
    )

    pinn_image = image_name_for_backend(backend)
    owns_pinn = pinn is None
    owns_solver = solver is None
    if owns_pinn:
        pinn = Tesseract.from_image(pinn_image)
    if owns_solver:
        solver = Tesseract.from_image("burgers_solver")

    params_flat = get_initial_params(backend, seed=data_config.seed)
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
    param_optimizer = optax.adam(training.param_learning_rate)
    param_opt_state = param_optimizer.init(params_flat)

    loss_and_grads = jax.value_and_grad(
        hybrid_discrepancy_loss, argnums=(0, 1), has_aux=True
    )

    viscosity = jnp.exp(log_viscosity)
    times = []
    viscosity_history = [float(viscosity)]
    log_viscosity_history = [float(log_viscosity)]
    loss_history = {name: [] for name in ("total", "data", "reg", "smooth")}
    effective_weights = {"data": w_data, "reg": w_reg, "smooth": w_smooth}

    def _run():
        nonlocal log_viscosity, params_flat, viscosity
        nonlocal log_visc_opt_state, param_opt_state

        callback.on_start(
            {
                "config": config,
                "backend": backend,
                "image_name": pinn_image,
                "solver_image": "burgers_solver",
                "warmup_epochs": warmup_epochs,
                "observations": obs,
            }
        )

        for epoch in range(training.n_epochs):
            start_time = time.time()

            counter.reset()
            with count_tesseract_calls(counter):
                (loss, components), (log_v_grad, p_grad) = loss_and_grads(
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
                )
            apply_calls = counter.apply_calls
            vjp_calls = counter.vjp_calls

            visc_grad_norm = float(jnp.abs(log_v_grad))
            param_grad_norm = float(jnp.linalg.norm(p_grad))

            viscosity_updated = epoch >= warmup_epochs
            if viscosity_updated:
                log_visc_updates, log_visc_opt_state = log_visc_optimizer.update(
                    log_v_grad, log_visc_opt_state
                )
                log_viscosity = optax.apply_updates(log_viscosity, log_visc_updates)
                if log_nu_bounds is not None:
                    log_viscosity = jnp.clip(
                        log_viscosity, log_nu_bounds[0], log_nu_bounds[1]
                    )

            param_updates, param_opt_state = param_optimizer.update(
                p_grad, param_opt_state
            )
            params_flat = optax.apply_updates(params_flat, param_updates)
            viscosity = jnp.exp(log_viscosity)

            epoch_time = time.time() - start_time
            times.append(epoch_time)
            viscosity_history.append(float(viscosity))
            log_viscosity_history.append(float(log_viscosity))

            loss_components = {name: float(value) for name, value in components.items()}
            if epoch % metrics_every == 0 or epoch == training.n_epochs - 1:
                for name in loss_history:
                    loss_history[name].append(loss_components[name])

            callback.on_epoch(
                EpochRecord(
                    epoch=epoch,
                    n_epochs=training.n_epochs,
                    viscosity=float(viscosity),
                    log_viscosity=float(log_viscosity),
                    loss=float(loss),
                    visc_grad_norm=visc_grad_norm,
                    param_grad_norm=param_grad_norm,
                    epoch_time=epoch_time,
                    apply_calls=apply_calls,
                    vjp_calls=vjp_calls,
                    effective_weights=effective_weights,
                    viscosity_updated=viscosity_updated,
                    param_count=int(params_flat.size),
                    brdr_weights=None,
                    loss_components=loss_components,
                )
            )

    if owns_pinn and owns_solver:
        with pinn, solver:
            _run()
    elif owns_pinn:
        with pinn:
            _run()
    elif owns_solver:
        with solver:
            _run()
    else:
        _run()

    final_viscosity = float(viscosity)
    relative_error = (
        abs(final_viscosity - problem.true_viscosity) / problem.true_viscosity * 100
    )
    avg_time = sum(times) / len(times) * 1000 if times else 0.0

    result = {
        "mode": "hybrid",
        "backend": backend,
        "tesseract_image": pinn_image,
        "solver_image": "burgers_solver",
        "final_viscosity": final_viscosity,
        "true_viscosity": problem.true_viscosity,
        "dispersion_beta": problem.dispersion_beta,
        "relative_error": relative_error,
        "avg_time_ms": avg_time,
        "viscosity_history": viscosity_history,
        "log_viscosity_history": log_viscosity_history,
        "loss_history": loss_history,
        "effective_weights": effective_weights,
        "seed": data_config.seed,
        "config": config,
        "params_flat": params_flat,
        "observations": obs,
        "pinn": pinn,
        "solver": solver,
    }
    callback.on_finish(result)
    return result


#
# Stage A baseline: solver-adjoint inversion (jax.grad through the solver VJP)
#


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


def train_solver_inverse(config, *, solver=None, callback=None, metrics_every=20):
    """Run solver-adjoint inversion: optimize ``log_nu`` against the in-loop solver.

    One reverse-mode pass per step differentiates the data-fit loss through the
    solver Tesseract VJP. Observations come from the same viscous-Burgers physics
    (clean, well-posed inverse), so the estimate recovers ``nu`` up to noise.
    """
    problem = config.problem
    data_config = config.data
    training = config.training

    callback = callback or TrainingCallback()
    counter = TesseractCallCounter()

    key = jax.random.PRNGKey(data_config.seed)
    obs = generate_grid_observations(
        data_config.n_obs,
        problem.true_viscosity,
        problem.domain,
        key,
        noise_std=data_config.noise_std,
    )

    owns_solver = solver is None
    if owns_solver:
        solver = Tesseract.from_image("burgers_solver")

    log_viscosity = jnp.log(jnp.asarray(problem.initial_viscosity))
    log_nu_bounds = None
    if training.clip_log_viscosity:
        log_nu_bounds = (
            jnp.log(jnp.asarray(training.nu_clip_min, dtype=jnp.float32)),
            jnp.log(jnp.asarray(training.nu_clip_max, dtype=jnp.float32)),
        )

    optimizer = optax.adam(training.log_nu_learning_rate)
    opt_state = optimizer.init(log_viscosity)
    loss_and_grad = jax.value_and_grad(solver_inverse_loss, argnums=0)

    viscosity = jnp.exp(log_viscosity)
    times = []
    viscosity_history = [float(viscosity)]
    log_viscosity_history = [float(log_viscosity)]
    loss_history = {"total": [], "data": []}

    def _run():
        nonlocal log_viscosity, viscosity, opt_state

        callback.on_start(
            {
                "config": config,
                "backend": "solver",
                "image_name": "burgers_solver",
                "warmup_epochs": 0,
                "observations": obs,
            }
        )

        for epoch in range(training.n_epochs):
            start_time = time.time()

            counter.reset()
            with count_tesseract_calls(counter):
                loss, log_v_grad = loss_and_grad(log_viscosity, obs, solver)
            apply_calls = counter.apply_calls
            vjp_calls = counter.vjp_calls

            updates, opt_state = optimizer.update(log_v_grad, opt_state)
            log_viscosity = optax.apply_updates(log_viscosity, updates)
            if log_nu_bounds is not None:
                log_viscosity = jnp.clip(
                    log_viscosity, log_nu_bounds[0], log_nu_bounds[1]
                )
            viscosity = jnp.exp(log_viscosity)

            epoch_time = time.time() - start_time
            times.append(epoch_time)
            viscosity_history.append(float(viscosity))
            log_viscosity_history.append(float(log_viscosity))

            loss_value = float(loss)
            loss_components = {"total": loss_value, "data": loss_value}
            if epoch % metrics_every == 0 or epoch == training.n_epochs - 1:
                loss_history["total"].append(loss_value)
                loss_history["data"].append(loss_value)

            callback.on_epoch(
                EpochRecord(
                    epoch=epoch,
                    n_epochs=training.n_epochs,
                    viscosity=float(viscosity),
                    log_viscosity=float(log_viscosity),
                    loss=loss_value,
                    visc_grad_norm=float(jnp.abs(log_v_grad)),
                    param_grad_norm=0.0,
                    epoch_time=epoch_time,
                    apply_calls=apply_calls,
                    vjp_calls=vjp_calls,
                    effective_weights={},
                    viscosity_updated=True,
                    param_count=0,
                    brdr_weights=None,
                    loss_components=loss_components,
                )
            )

    if owns_solver:
        with solver:
            _run()
    else:
        _run()

    final_viscosity = float(viscosity)
    relative_error = (
        abs(final_viscosity - problem.true_viscosity) / problem.true_viscosity * 100
    )
    avg_time = sum(times) / len(times) * 1000 if times else 0.0

    result = {
        "mode": "solver-inverse",
        "backend": "solver",
        "tesseract_image": "burgers_solver",
        "final_viscosity": final_viscosity,
        "true_viscosity": problem.true_viscosity,
        "relative_error": relative_error,
        "avg_time_ms": avg_time,
        "viscosity_history": viscosity_history,
        "log_viscosity_history": log_viscosity_history,
        "loss_history": loss_history,
        "seed": data_config.seed,
        "config": config,
        "observations": obs,
        "solver": solver,
    }
    callback.on_finish(result)
    return result


def log_solver_inverse_results(result):
    """Log the solver-adjoint inversion result table."""
    table = Table(title="Solver-Adjoint Inversion")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right", style="cyan")
    table.add_row("Inferred ν", f"{result['final_viscosity']:.6f}")
    table.add_row("True ν", f"{result['true_viscosity']:.6f}")
    table.add_row("Relative error", f"{result['relative_error']:.2f}%")
    table.add_row("Avg time/epoch", f"{result['avg_time_ms']:.1f} ms")
    if result["loss_history"]["data"]:
        table.add_row("Final data loss", f"{result['loss_history']['data'][-1]:.6e}")
    CONSOLE.print(table)


class SolverInverseCallback(TrainingCallback):
    """CLI presentation for solver-adjoint inversion: progress bar + results, and
    records per-epoch metrics rows for reproducible artifacts."""

    def __init__(self, config):
        self.config = config
        self.true_viscosity = config.problem.true_viscosity
        self.progress = make_training_progress()
        self.task_id = None
        self._last_loss = None
        self.rows = []

    def on_start(self, context):
        initial = float(self.config.problem.initial_viscosity)
        CONSOLE.log("Solver-adjoint inversion (jax.grad through solver Tesseract VJP)")
        self.progress.start()
        self.task_id = self.progress.add_task(
            "solver-inverse",
            total=self.config.training.n_epochs,
            loss="pending",
            nu=f"{initial:.6f}",
            error=f"{abs(initial - self.true_viscosity):.6f}",
            epoch_time="pending",
        )

    def on_epoch(self, record):
        if record.loss_components is not None:
            self._last_loss = record.loss_components["total"]
        loss_text = (
            f"{self._last_loss:.3e}" if self._last_loss is not None else "pending"
        )
        self.progress.update(
            self.task_id,
            advance=1,
            loss=loss_text,
            nu=f"{record.viscosity:.6f}",
            error=f"{abs(record.viscosity - self.true_viscosity):.6f}",
            epoch_time=f"{record.epoch_time * 1000:.0f}ms",
        )
        self.rows.append(
            {
                "epoch": record.epoch,
                "viscosity": record.viscosity,
                "log_viscosity": record.log_viscosity,
                "loss": record.loss,
                "visc_grad_norm": record.visc_grad_norm,
                "param_grad_norm": record.param_grad_norm,
                "epoch_time": record.epoch_time,
                "apply_calls": record.apply_calls,
                "vjp_calls": record.vjp_calls,
                "viscosity_updated": record.viscosity_updated,
            }
        )

    def on_finish(self, result):
        self.progress.stop()
        log_solver_inverse_results(result)


def run_solver_inverse(config):
    """Run solver-adjoint inversion with CLI presentation and image guard."""
    if config.problem.initial_viscosity <= 0:
        raise ValueError("initial_viscosity must be positive when optimizing log_nu")
    ensure_image_available("burgers_solver")
    CONSOLE.rule("[bold cyan]Solver-Adjoint Inversion")
    callback = SolverInverseCallback(config)
    result = train_solver_inverse(config, callback=callback)
    result["metrics_rows"] = callback.rows
    return result


class RichProgressCallback(TrainingCallback):
    """CLI presentation: a live rich progress bar plus a final results table."""

    def __init__(self, config):
        self.config = config
        self.true_viscosity = config.problem.true_viscosity
        self.progress = make_training_progress()
        self.task_id = None
        self._last_loss = None

    def on_start(self, context):
        initial = float(self.config.problem.initial_viscosity)
        CONSOLE.log(f"{context['backend'].upper()} PINN tesseract initialized")
        CONSOLE.log("Optimizing...")
        self.progress.start()
        self.task_id = self.progress.add_task(
            f"{context['backend'].upper()} training",
            total=self.config.training.n_epochs,
            loss="pending",
            nu=f"{initial:.6f}",
            error=f"{abs(initial - self.true_viscosity):.6f}",
            epoch_time="pending",
        )

    def on_epoch(self, record):
        if record.loss_components is not None:
            self._last_loss = record.loss_components["total"]
        loss_text = (
            f"{self._last_loss:.3e}" if self._last_loss is not None else "pending"
        )
        self.progress.update(
            self.task_id,
            advance=1,
            loss=loss_text,
            nu=f"{record.viscosity:.6f}",
            error=f"{abs(record.viscosity - self.true_viscosity):.6f}",
            epoch_time=f"{record.epoch_time * 1000:.0f}ms",
        )

    def on_finish(self, result):
        self.progress.stop()
        log_final_results(
            result["final_viscosity"],
            result["true_viscosity"],
            result["relative_error"],
            result["avg_time_ms"],
            result["loss_history"],
            loss_weight_history=result["loss_weight_history"]
            if result["adaptive_loss_weights"]
            else None,
        )


class MetricsRecorderCallback(RichProgressCallback):
    """CLI callback that records per-epoch metrics while showing progress."""

    def __init__(self, config):
        super().__init__(config)
        self.rows = []

    def on_epoch(self, record):
        self.rows.append(
            {
                "epoch": record.epoch,
                "viscosity": record.viscosity,
                "log_viscosity": record.log_viscosity,
                "loss": record.loss,
                "visc_grad_norm": record.visc_grad_norm,
                "param_grad_norm": record.param_grad_norm,
                "epoch_time": record.epoch_time,
                "apply_calls": record.apply_calls,
                "vjp_calls": record.vjp_calls,
                "viscosity_updated": record.viscosity_updated,
            }
        )
        super().on_epoch(record)


def _to_jsonable(value):
    """Convert dataclass/JAX/NumPy scalars and containers to JSON-safe values."""
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def write_run_artifacts(result, rows, out_dir):
    """Write reproducible CLI run artifacts without serializing live objects."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with (out_path / "config.json").open("w", encoding="utf-8") as file:
        json.dump(_to_jsonable(result["config"]), file, indent=2, sort_keys=True)
        file.write("\n")

    fieldnames = [
        "epoch",
        "viscosity",
        "log_viscosity",
        "loss",
        "visc_grad_norm",
        "param_grad_norm",
        "epoch_time",
        "apply_calls",
        "vjp_calls",
        "viscosity_updated",
    ]
    with (out_path / "metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    last_row = rows[-1] if rows else {}
    summary = {
        "mode": result.get("mode", "pinn"),
        "backend": result["backend"],
        "tesseract_image": result["tesseract_image"],
        "seed": result["seed"],
        "true_viscosity": result["true_viscosity"],
        "final_viscosity": result["final_viscosity"],
        "relative_error": result["relative_error"],
        "avg_time_ms": result["avg_time_ms"],
        "epochs": len(result["viscosity_history"]) - 1,
        "apply_calls_per_step": last_row.get("apply_calls"),
        "vjp_calls_per_step": last_row.get("vjp_calls"),
        "adaptive_loss_weights": result.get("adaptive_loss_weights", False),
    }
    with (out_path / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(_to_jsonable(summary), file, indent=2, sort_keys=True)
        file.write("\n")


def run_inverse_problem(
    config=None,
    backend=None,
    true_viscosity=None,
    initial_viscosity=None,
    n_obs=None,
    n_epochs=None,
    learning_rate=None,
    adaptive_loss_weights=None,
    brdr_beta_c=None,
    brdr_beta_w=None,
    brdr_epsilon=None,
    loss_weights=None,
    seed=None,
):
    """
    Run inverse problem to infer viscosity parameter.

    Args:
        backend: "jax" or "pytorch" - which PINN tesseract to use
    """
    config = build_run_config(
        config,
        backend=backend,
        true_viscosity=true_viscosity,
        initial_viscosity=initial_viscosity,
        n_obs=n_obs,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        adaptive_loss_weights=adaptive_loss_weights,
        brdr_beta_c=brdr_beta_c,
        brdr_beta_w=brdr_beta_w,
        brdr_epsilon=brdr_epsilon,
        loss_weights=loss_weights,
        seed=seed,
    )

    if config.problem.initial_viscosity <= 0:
        raise ValueError("initial_viscosity must be positive when optimizing log_nu")

    ensure_image_available(image_name_for_backend(config.backend))
    log_run_header(config)

    callback = MetricsRecorderCallback(config)
    result = train_inverse(config, callback=callback)
    result["metrics_rows"] = callback.rows
    CONSOLE.log(f"Model parameters: {result['params_flat'].size}")
    return result


def compare_backends(
    config=None,
    n_epochs=None,
    n_obs=None,
    adaptive_loss_weights=None,
    brdr_beta_c=None,
    brdr_beta_w=None,
    brdr_epsilon=None,
    loss_weights=None,
    seed=None,
):
    """Run inverse problem with both backends for comparison."""
    config = build_run_config(
        config,
        n_epochs=n_epochs,
        n_obs=n_obs,
        adaptive_loss_weights=adaptive_loss_weights,
        brdr_beta_c=brdr_beta_c,
        brdr_beta_w=brdr_beta_w,
        brdr_epsilon=brdr_epsilon,
        loss_weights=loss_weights,
        seed=seed,
    )

    CONSOLE.rule("[bold cyan]Cross-Framework Autodiff Comparison")

    results = {}

    # Run JAX PINN
    results["jax"] = run_inverse_problem(
        config=config.with_backend("jax"),
    )

    # Run PyTorch PINN
    results["pytorch"] = run_inverse_problem(
        config=config.with_backend("pytorch"),
    )

    log_backend_comparison(results)

    if results["jax"]["avg_time_ms"] > 0 and results["pytorch"]["avg_time_ms"] > 0:
        speedup = results["pytorch"]["avg_time_ms"] / results["jax"]["avg_time_ms"]
        if speedup > 1:
            CONSOLE.log(f"JAX is {speedup:.1f}x faster than PyTorch")
        else:
            CONSOLE.log(f"PyTorch is {1 / speedup:.1f}x faster than JAX")

    CONSOLE.print(
        "\n[bold]Notes[/bold]\n"
        "The same optimization pipeline executes with both backends.\n"
        "Gradients are computed via Tesseract's VJP endpoint "
        "(jax.grad through PyTorch).\n"
        "Backends can be swapped by changing the Tesseract image name."
    )

    return results


def run_single_backend(
    backend=None,
    n_epochs=None,
    adaptive_loss_weights=None,
    brdr_beta_c=None,
    brdr_beta_w=None,
    brdr_epsilon=None,
    loss_weights=None,
    seed=None,
    config=None,
):
    """Run inverse problem with a single backend only."""
    if config is None and backend is None:
        backend = "jax"
    return run_inverse_problem(
        config=config,
        backend=backend,
        n_epochs=n_epochs,
        adaptive_loss_weights=adaptive_loss_weights,
        brdr_beta_c=brdr_beta_c,
        brdr_beta_w=brdr_beta_w,
        brdr_epsilon=brdr_epsilon,
        loss_weights=loss_weights,
        seed=seed,
    )


def _last_call_counts(result):
    """Return (apply, vjp) Tesseract dispatches per step from the last metrics row."""
    rows = result.get("metrics_rows", [])
    last = rows[-1] if rows else {}
    return last.get("apply_calls"), last.get("vjp_calls")


def log_method_comparison(results):
    """Log a unified table comparing the inverse methods."""
    table = Table(title="Inverse Method Comparison")
    table.add_column("Method", style="bold")
    table.add_column("Tesseract", style="dim")
    table.add_column("Inferred ν", justify="right", style="cyan")
    table.add_column("Rel error (%)", justify="right")
    table.add_column("ms/epoch", justify="right")
    table.add_column("apply/vjp per step", justify="right")

    for label, result in results.items():
        apply_calls, vjp_calls = _last_call_counts(result)
        calls = (
            f"{apply_calls}/{vjp_calls}"
            if apply_calls is not None and vjp_calls is not None
            else "-"
        )
        table.add_row(
            label,
            result["tesseract_image"],
            f"{result['final_viscosity']:.6f}",
            f"{result['relative_error']:.2f}",
            f"{result['avg_time_ms']:.1f}",
            calls,
        )
    CONSOLE.print(table)


def compare_methods(config):
    """Compare solver-adjoint inversion against the PINN method (JAX and PyTorch).

    All three run on the same viscous-Burgers truth and data budget, exercising the
    same uniform Tesseract interface across three swappable, framework-agnostic
    components (the JAX solver and the JAX/PyTorch PINN). Observations are drawn
    independently per method from the same physics and seed.
    """
    CONSOLE.rule("[bold cyan]Inverse Method Comparison")
    results = {}
    results["solver-adjoint"] = run_solver_inverse(config)
    results["pinn (JAX)"] = run_single_backend(backend="jax", config=config)
    results["pinn (PyTorch)"] = run_single_backend(backend="pytorch", config=config)
    log_method_comparison(results)
    return results


def run_seed_sweep(
    backend="jax",
    seeds=(123,),
    n_epochs=None,
    adaptive_loss_weights=None,
    brdr_beta_c=None,
    brdr_beta_w=None,
    brdr_epsilon=None,
    loss_weights=None,
    config=None,
):
    """Run one backend or both backends across multiple random seeds."""
    config = build_run_config(
        config,
        n_epochs=n_epochs,
        adaptive_loss_weights=adaptive_loss_weights,
        brdr_beta_c=brdr_beta_c,
        brdr_beta_w=brdr_beta_w,
        brdr_epsilon=brdr_epsilon,
        loss_weights=loss_weights,
    )
    results = []

    for seed in seeds:
        seed_config = config.with_seed(seed)
        if backend == "both":
            backend_results = compare_backends(
                config=seed_config,
            )
            results.extend(backend_results.values())
        else:
            results.append(
                run_single_backend(
                    backend=backend,
                    config=seed_config,
                )
            )

    log_seed_summary(results)
    return results


def write_cli_artifacts(results, out_dir):
    """Write artifacts for CLI result objects returned by this module."""
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    root = Path(out_dir) / timestamp

    if isinstance(results, dict) and "backend" in results:
        write_run_artifacts(
            results,
            results.get("metrics_rows", []),
            root / results["backend"],
        )
    elif isinstance(results, dict):
        for backend, result in results.items():
            write_run_artifacts(result, result.get("metrics_rows", []), root / backend)
    else:
        for result in results:
            seed_dir = f"seed-{result['seed']}"
            write_run_artifacts(
                result,
                result.get("metrics_rows", []),
                root / seed_dir / result["backend"],
            )

    CONSOLE.log(f"Wrote run artifacts to {root}")
    return root


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inverse Problem Demo")
    parser.add_argument(
        "--mode",
        choices=["pinn", "solver-inverse", "compare"],
        default="pinn",
        help=(
            "Inverse method: 'pinn' (jax.grad through the PINN Tesseract), "
            "'solver-inverse' (solver-adjoint; jax.grad through the solver Tesseract), "
            "or 'compare' (solver-adjoint vs PINN JAX/PyTorch in one table)"
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["jax", "pytorch", "both"],
        default="both",
        help="Which PINN backend to use (pinn mode only)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for observations, collocation points, and model init",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Run a seed sweep with one or more random seeds",
    )
    parser.add_argument(
        "--adaptive-loss-weights",
        action="store_true",
        help="Use BRDR pointwise adaptive weights for data/physics/IC/BC residuals",
    )
    parser.add_argument(
        "--brdr-beta-c",
        type=float,
        default=RunConfig().training.brdr_beta_c,
        help="EMA factor for BRDR residual-history estimates",
    )
    parser.add_argument(
        "--brdr-beta-w",
        type=float,
        default=RunConfig().training.brdr_beta_w,
        help="EMA factor for BRDR pointwise weights",
    )
    parser.add_argument(
        "--brdr-epsilon",
        type=float,
        default=RunConfig().training.brdr_epsilon,
        help="Small positive constant for BRDR numerical stability",
    )
    parser.add_argument(
        "--w-data",
        type=float,
        default=DEFAULT_LOSS_WEIGHTS["data"],
        help="Data loss weight",
    )
    parser.add_argument(
        "--w-physics",
        type=float,
        default=DEFAULT_LOSS_WEIGHTS["physics"],
        help="Physics residual loss weight",
    )
    parser.add_argument(
        "--w-ic",
        type=float,
        default=DEFAULT_LOSS_WEIGHTS["ic"],
        help="Initial-condition loss weight",
    )
    parser.add_argument(
        "--w-bc",
        type=float,
        default=DEFAULT_LOSS_WEIGHTS["bc"],
        help="Boundary-condition loss weight",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Directory for reproducible run artifacts",
    )
    args = parser.parse_args()

    try:
        loss_weights = loss_weights_from_mapping(
            {
                "data": args.w_data,
                "physics": args.w_physics,
                "ic": args.w_ic,
                "bc": args.w_bc,
            }
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    config = build_run_config(
        backend=args.backend,
        n_epochs=args.epochs,
        seed=args.seed,
        adaptive_loss_weights=args.adaptive_loss_weights,
        brdr_beta_c=args.brdr_beta_c,
        brdr_beta_w=args.brdr_beta_w,
        brdr_epsilon=args.brdr_epsilon,
        loss_weights=loss_weights,
    )

    try:
        if args.mode == "solver-inverse":
            results = run_solver_inverse(config=config)
        elif args.mode == "compare":
            results = compare_methods(config=config)
        elif args.seeds:
            results = run_seed_sweep(
                backend=args.backend,
                seeds=args.seeds,
                config=config,
            )
        elif args.backend == "both":
            results = compare_backends(config=config)
        else:
            results = run_single_backend(
                backend=args.backend,
                config=config,
            )
        if args.out is not None:
            write_cli_artifacts(results, args.out)
    except TesseractImageNotFoundError as exc:
        CONSOLE.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)
