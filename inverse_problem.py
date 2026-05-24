"""Inverse problem demo for viscosity inference in Burgers equation.

Demonstrates cross-framework automatic differentiation via Tesseract:
- Same optimization code runs with JAX or PyTorch PINN backends
- JAX gradients computed through PyTorch models via VJP endpoint
- Backend selection controlled by Tesseract image name

Problem: Given noisy observations u(x,t), infer viscosity parameter ν
in Burgers equation: ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
"""

import sys
import time
from dataclasses import replace
from pathlib import Path
from statistics import fmean, pstdev

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
    LOSS_WEIGHT_NAMES,
    LossWeights,
    RunConfig,
    loss_weights_from_mapping,
    normalize_loss_weights,
)


REPO_ROOT = Path(__file__).resolve().parent
CONSOLE = Console()


def format_loss_weights(loss_weights):
    """Format loss weights for compact CLI output."""
    return ", ".join(f"{name}={loss_weights[name]:g}" for name in LOSS_WEIGHT_NAMES)


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
    final_viscosity, true_viscosity, relative_error, avg_time, loss_history
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
        backend_results = [
            result for result in results if result["backend"] == backend
        ]
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
    if backend == "jax":
        sys.path.insert(0, "tesseracts/pinn_jax")
        from tesseract_api import PINNNet, flatten_params

        model = PINNNet(jax.random.PRNGKey(seed))
        params = flatten_params(model)
        sys.path.pop(0)
        # Clear the imported module to avoid conflicts
        if "tesseract_api" in sys.modules:
            del sys.modules["tesseract_api"]
        return jnp.array(params)
    else:  # pytorch
        # For PyTorch, initialize from actual model for proper initialization
        sys.path.insert(0, "tesseracts/pinn_pytorch")
        from tesseract_api import PINNNet, flatten_params

        torch.manual_seed(seed)
        model = PINNNet(hidden_sizes=[64, 64, 64], n_fourier_features=32, seed=seed)
        params = flatten_params(model)
        sys.path.pop(0)
        # Clear the imported module to avoid conflicts
        if "tesseract_api" in sys.modules:
            del sys.modules["tesseract_api"]
        return jnp.array(params)


def generate_observations(n_points, true_viscosity, domain, key, noise_std=0.02):
    """
    Generate synthetic observations from the pseudospectral Burgers solver.

    The solver uses the same sinusoidal initial condition assumed by the PINN
    initial-condition loss: u(x, 0) = sin(2πx).
    """
    nx = 128
    nt = 64
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
    min_t_idx = max(1, int(jnp.searchsorted(t_grid, 0.05, side="left")))
    t_idx = jax.random.randint(keys[1], (n_points,), minval=min_t_idx, maxval=nt)

    x = x_grid[x_idx]
    t = t_grid[t_idx]
    u_observed = u_field[t_idx, x_idx]

    # Add small noise
    noise = jax.random.normal(keys[2], (n_points,)) * noise_std
    u_observed = u_observed + noise

    return x, t, u_observed


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

    # Data loss
    result_obs = apply_tesseract(
        pinn,
        {
            "x": x_obs,
            "t": t_obs,
            "params_flat": params_flat,
        },
    )
    u_pred = result_obs["u_pred"]
    data_loss = jnp.mean((u_pred - u_obs) ** 2)

    # PDE residual loss components
    result_col = apply_tesseract(
        pinn, {"x": x_col, "t": t_col, "params_flat": params_flat}
    )

    u_col = result_col["u_pred"]  # Solution values
    u_x = result_col["u_x"]  # ∂u/∂x
    u_t = result_col["u_t"]  # ∂u/∂t
    u_xx = result_col["u_xx"]  # ∂²u/∂x²

    # Burgers residual
    residual = u_t + u_col * u_x - viscosity * u_xx
    physics_loss = jnp.mean(residual**2)

    # IC loss (sin(2πx) at t=0) and BC loss (periodic)
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
    ic_loss = jnp.mean((u_ic - u_ic_true) ** 2)

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
    bc_loss = jnp.mean((u_left - u_right) ** 2)

    total_loss = (
        weights["data"] * data_loss
        + weights["physics"] * physics_loss
        + weights["ic"] * ic_loss
        + weights["bc"] * bc_loss
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
        loss_weights=loss_weights,
    )


def run_inverse_problem(
    config=None,
    backend=None,
    true_viscosity=None,
    initial_viscosity=None,
    n_obs=None,
    n_epochs=None,
    learning_rate=None,
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
        loss_weights=loss_weights,
        seed=seed,
    )
    backend = config.backend
    problem = config.problem
    data_config = config.data
    training = config.training
    loss_weights = config.loss.as_dict()
    domain = problem.domain

    if problem.initial_viscosity <= 0:
        raise ValueError("initial_viscosity must be positive when optimizing log_nu")

    log_run_header(config)

    # Make observations and collocation points
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
        key_col_t, (training.n_col,), minval=0.05, maxval=domain["t"][1]
    )

    x_ic = jax.random.uniform(
        key_ic, (training.n_ic,), minval=domain["x"][0], maxval=domain["x"][1]
    )

    t_bc = jax.random.uniform(
        key_bc, (training.n_bc,), minval=0.05, maxval=domain["t"][1]
    )

    image_name = "pinn_jax" if backend == "jax" else "pinn_pytorch"
    pinn = Tesseract.from_image(image_name)

    params_flat = get_initial_params(backend, seed=data_config.seed)
    CONSOLE.log(f"Model parameters: {params_flat.size}")

    log_viscosity = jnp.log(jnp.asarray(problem.initial_viscosity))
    viscosity = jnp.exp(log_viscosity)

    log_visc_optimizer = optax.adam(training.log_nu_learning_rate)
    log_visc_opt_state = log_visc_optimizer.init(log_viscosity)

    param_optimizer = optax.adam(training.param_learning_rate)
    param_opt_state = param_optimizer.init(params_flat)

    # Gradient functions
    # jax.grad computes gradients through the tesseract VJP endpoint
    grad_log_visc = jax.grad(compute_loss_from_log_viscosity, argnums=0)
    grad_params = jax.grad(compute_loss, argnums=1)

    with pinn:
        CONSOLE.log(f"{backend.upper()} PINN tesseract initialized")
        CONSOLE.log("Optimizing...")

        times = []
        viscosity_history = [float(viscosity)]
        log_viscosity_history = [float(log_viscosity)]
        loss_history = {name: [] for name in ("total", "data", "physics", "ic", "bc")}

        with make_training_progress() as progress:
            task_id = progress.add_task(
                f"{backend.upper()} training",
                total=training.n_epochs,
                loss="pending",
                nu=f"{float(viscosity):.6f}",
                error=f"{abs(float(viscosity) - problem.true_viscosity):.6f}",
                epoch_time="pending",
            )

            for epoch in range(training.n_epochs):
                start_time = time.time()

                # Compute gradients
                log_v_grad = grad_log_visc(
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
                    loss_weights=loss_weights,
                )
                viscosity = jnp.exp(log_viscosity)
                p_grad = grad_params(
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
                    loss_weights=loss_weights,
                )

                # Both viscosity and parameters are updated
                log_visc_updates, log_visc_opt_state = log_visc_optimizer.update(
                    log_v_grad, log_visc_opt_state
                )
                log_viscosity = optax.apply_updates(log_viscosity, log_visc_updates)
                viscosity = jnp.exp(log_viscosity)

                param_updates, param_opt_state = param_optimizer.update(
                    p_grad, param_opt_state
                )
                params_flat = optax.apply_updates(params_flat, param_updates)

                epoch_time = time.time() - start_time
                times.append(epoch_time)
                viscosity_history.append(float(viscosity))
                log_viscosity_history.append(float(log_viscosity))

                loss_value = loss_history["total"][-1] if loss_history["total"] else None
                if epoch % 20 == 0 or epoch == training.n_epochs - 1:
                    loss_components = compute_loss_components(
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
                        loss_weights=loss_weights,
                    )
                    for name, value in loss_components.items():
                        loss_history[name].append(float(value))
                    loss_value = float(loss_components["total"])

                progress.update(
                    task_id,
                    advance=1,
                    loss=f"{loss_value:.3e}" if loss_value is not None else "pending",
                    nu=f"{float(viscosity):.6f}",
                    error=f"{abs(float(viscosity) - problem.true_viscosity):.6f}",
                    epoch_time=f"{epoch_time * 1000:.0f}ms",
                )

        final_viscosity = float(viscosity)
        relative_error = (
            abs(final_viscosity - problem.true_viscosity)
            / problem.true_viscosity
            * 100
        )
        avg_time = sum(times) / len(times) * 1000 if times else 0.0

        log_final_results(
            final_viscosity,
            problem.true_viscosity,
            relative_error,
            avg_time,
            loss_history,
        )

    return {
        "backend": backend,
        "final_viscosity": final_viscosity,
        "true_viscosity": problem.true_viscosity,
        "relative_error": relative_error,
        "avg_time_ms": avg_time,
        "viscosity_history": viscosity_history,
        "log_viscosity_history": log_viscosity_history,
        "loss_history": loss_history,
        "loss_weights": loss_weights,
        "seed": data_config.seed,
        "config": config,
    }


def compare_backends(
    config=None,
    n_epochs=None,
    n_obs=None,
    loss_weights=None,
    seed=None,
):
    """Run inverse problem with both backends for comparison."""
    config = build_run_config(
        config,
        n_epochs=n_epochs,
        n_obs=n_obs,
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
        loss_weights=loss_weights,
        seed=seed,
    )


def run_seed_sweep(
    backend="jax",
    seeds=(123,),
    n_epochs=None,
    loss_weights=None,
    config=None,
):
    """Run one backend or both backends across multiple random seeds."""
    config = build_run_config(config, n_epochs=n_epochs, loss_weights=loss_weights)
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inverse Problem Demo")
    parser.add_argument(
        "--backend",
        choices=["jax", "pytorch", "both"],
        default="both",
        help="Which backend to use",
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
        loss_weights=loss_weights,
    )

    if args.seeds:
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
