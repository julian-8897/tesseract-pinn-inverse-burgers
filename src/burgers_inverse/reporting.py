"""Presentation layer: Rich console tables, progress callbacks, and artifacts.

All terminal output and reproducible run-artifact serialization for the
deterministic inverse methods. Imports the engine's :class:`TrainingCallback`
contract but the engine never imports this module, keeping optimization free of
presentation concerns.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from statistics import fmean, pstdev

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from burgers_inverse.configs import LOSS_WEIGHT_NAMES
from burgers_inverse.engine import TrainingCallback
from burgers_inverse.losses import format_loss_weights

CONSOLE = Console()


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
