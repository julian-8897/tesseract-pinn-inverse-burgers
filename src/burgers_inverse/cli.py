"""Command-line orchestration for the deterministic inverse methods.

Builds run configs from defaults plus overrides, drives the engine with CLI
presentation callbacks, and serializes reproducible artifacts. Run with
``python -m burgers_inverse.cli`` or the ``burgers-inverse`` console script.
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

from burgers_inverse.components import (
    TesseractImageNotFoundError,
    ensure_image_available,
    image_name_for_backend,
)
from burgers_inverse.configs import (
    DEFAULT_LOSS_WEIGHTS,
    ComponentConfig,
    LossWeights,
    RunConfig,
    loss_weights_from_mapping,
)
from burgers_inverse.engine import train_inverse, train_solver_inverse
from burgers_inverse.reporting import (
    CONSOLE,
    MetricsRecorderCallback,
    SolverInverseCallback,
    log_backend_comparison,
    log_method_comparison,
    log_run_header,
    log_seed_summary,
    write_run_artifacts,
)


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
    solver_image=None,
    pinn_jax_image=None,
    pinn_pytorch_image=None,
    fmpe_image=None,
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

    component_updates = {}
    if solver_image is not None:
        component_updates["solver_image"] = solver_image
    if pinn_jax_image is not None:
        component_updates["pinn_jax_image"] = pinn_jax_image
    if pinn_pytorch_image is not None:
        component_updates["pinn_pytorch_image"] = pinn_pytorch_image
    if fmpe_image is not None:
        component_updates["fmpe_image"] = fmpe_image
    if component_updates:
        components = (
            config.components
            if isinstance(config.components, ComponentConfig)
            else ComponentConfig(**config.components)
        )
        config = replace(
            config,
            components=replace(components, **component_updates),
        )

    return config


def run_solver_inverse(
    config, *, checkpoint_path=None, checkpoint_every=0, resume_from=None
):
    """Run solver-adjoint inversion with CLI presentation and image guard."""
    if config.problem.initial_viscosity <= 0:
        raise ValueError("initial_viscosity must be positive when optimizing log_nu")
    ensure_image_available(config.components.solver_image)
    CONSOLE.rule("[bold cyan]Solver-Adjoint Inversion")
    callback = SolverInverseCallback(config)
    result = train_solver_inverse(
        config,
        callback=callback,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        resume_from=resume_from,
    )
    result["metrics_rows"] = callback.rows
    return result


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
    checkpoint_path=None,
    checkpoint_every=0,
    resume_from=None,
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

    ensure_image_available(image_name_for_backend(config.backend, config.components))
    log_run_header(config)

    callback = MetricsRecorderCallback(config)
    result = train_inverse(
        config,
        callback=callback,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        resume_from=resume_from,
    )
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
    checkpoint_path=None,
    checkpoint_every=0,
    resume_from=None,
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
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        resume_from=resume_from,
    )


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


def main(argv=None):
    """CLI entry point for the deterministic inverse methods."""
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
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Write a resumable deterministic-training checkpoint",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Checkpoint cadence in epochs; 0 writes only the final state",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from a trusted local checkpoint; --epochs is the total target",
    )
    parser.add_argument(
        "--solver-image",
        default=RunConfig().components.solver_image,
        help="Solver Tesseract image reference, optionally registry/tag/digest pinned",
    )
    parser.add_argument(
        "--pinn-jax-image",
        default=RunConfig().components.pinn_jax_image,
        help="JAX PINN Tesseract image reference",
    )
    parser.add_argument(
        "--pinn-pytorch-image",
        default=RunConfig().components.pinn_pytorch_image,
        help="PyTorch PINN Tesseract image reference",
    )
    args = parser.parse_args(argv)

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
        solver_image=args.solver_image,
        pinn_jax_image=args.pinn_jax_image,
        pinn_pytorch_image=args.pinn_pytorch_image,
    )

    checkpoint_target = args.checkpoint or args.resume
    multi_run = args.mode == "compare" or (
        args.mode == "pinn" and (args.backend == "both" or args.seeds)
    )
    if (args.checkpoint is not None or args.resume is not None) and multi_run:
        parser.error(
            "Checkpoint/resume requires one deterministic method and one backend"
        )
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be non-negative")

    try:
        if args.mode == "solver-inverse":
            results = run_solver_inverse(
                config=config,
                checkpoint_path=checkpoint_target,
                checkpoint_every=args.checkpoint_every,
                resume_from=args.resume,
            )
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
                checkpoint_path=checkpoint_target,
                checkpoint_every=args.checkpoint_every,
                resume_from=args.resume,
            )
        if args.out is not None:
            write_cli_artifacts(results, args.out)
    except TesseractImageNotFoundError as exc:
        CONSOLE.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
