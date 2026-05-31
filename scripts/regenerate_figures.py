"""Regenerate professional README figures from the current solver-backed path."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

from configs import DataConfig, RunConfig, TrainingConfig
from inverse_problem import (
    TrainingCallback,
    ensure_image_available,
    get_burgers_solver,
    image_name_for_backend,
    train_inverse,
)

BACKEND_STYLES = {
    "jax": {"label": "JAX PINN", "color": "#2563eb"},
    "pytorch": {"label": "PyTorch PINN", "color": "#f97316"},
}
FIELD_CMAP = "RdBu_r"
ERROR_CMAP = "magma"
TEXT_COLOR = "#111827"
MUTED_COLOR = "#4b5563"
GRID_COLOR = "#d1d5db"


class FigureMetricsCallback(TrainingCallback):
    """Record per-epoch scalar metrics without rendering CLI progress."""

    def __init__(self):
        self.rows = []

    def on_epoch(self, record):
        self.rows.append(
            {
                "epoch": record.epoch,
                "viscosity": record.viscosity,
                "loss": record.loss,
                "apply_calls": record.apply_calls,
                "vjp_calls": record.vjp_calls,
            }
        )


def configure_matplotlib():
    """Use a consistent style suitable for README display."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#374151",
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "font.size": 10,
            "legend.frameon": False,
            "xtick.color": MUTED_COLOR,
            "ytick.color": MUTED_COLOR,
            "grid.color": GRID_COLOR,
            "grid.alpha": 0.35,
            "grid.linewidth": 0.7,
        }
    )


def finish_axis(ax):
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def generate_solution_grid(true_viscosity, params_flat, pinn, nx, nt):
    """Evaluate the trained PINN and solver ground truth on a common grid."""
    x = np.linspace(0.0, 1.0, nx, endpoint=False, dtype=np.float32)
    t = np.linspace(0.0, 1.0, nt, dtype=np.float32)
    x_grid, t_grid = np.meshgrid(x, t)

    result = apply_tesseract(
        pinn,
        {
            "x": jnp.asarray(x_grid.ravel(), dtype=jnp.float32),
            "t": jnp.asarray(t_grid.ravel(), dtype=jnp.float32),
            "params_flat": params_flat,
        },
    )
    u_pred = np.asarray(result["u_pred"]).reshape(nt, nx)

    solve_burgers = get_burgers_solver()
    u_solver = solve_burgers(
        jnp.asarray(true_viscosity, dtype=jnp.float32),
        jnp.asarray(x, dtype=jnp.float32),
        jnp.asarray(t, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )

    return x_grid, t_grid, u_pred, np.asarray(u_solver)


def save_field_figure(result, out_path, nx, nt):
    """Save a three-panel PINN / solver / error field figure."""
    backend = result["backend"]
    style = BACKEND_STYLES[backend]
    config = result["config"]
    x_obs, t_obs, _ = result["observations"]
    true_nu = result["true_viscosity"]
    final_nu = result["final_viscosity"]

    if "solution_grid" in result:
        x_grid, t_grid, u_pred, u_solver = result["solution_grid"]
    else:
        pinn = Tesseract.from_image(result["tesseract_image"])
        with pinn:
            x_grid, t_grid, u_pred, u_solver = generate_solution_grid(
                true_nu,
                result["params_flat"],
                pinn,
                nx,
                nt,
            )

    error = np.abs(u_pred - u_solver)
    field_abs = float(np.nanmax(np.abs([u_pred, u_solver])))
    field_norm = colors.TwoSlopeNorm(vmin=-field_abs, vcenter=0.0, vmax=field_abs)
    field_levels = np.linspace(-field_abs, field_abs, 43)
    error_max = float(np.nanmax(error))
    error_levels = np.linspace(0.0, error_max, 43)

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.055, right=0.98, bottom=0.12, top=0.76, wspace=0.24)
    fig.suptitle(
        f"{style['label']} Reconstruction vs Solver Ground Truth",
        fontsize=15,
        fontweight="bold",
        color=TEXT_COLOR,
        y=0.97,
    )
    fig.text(
        0.5,
        0.89,
        (
            f"seed={config.data.seed} | inferred nu={final_nu:.5f} | "
            f"true nu={true_nu:.5f} | relative error={result['relative_error']:.2f}%"
        ),
        ha="center",
        va="center",
        color=MUTED_COLOR,
        fontsize=10,
    )

    panels = [
        (u_pred, f"PINN field ({backend})", FIELD_CMAP, field_norm, field_levels),
        (u_solver, "Solver ground truth", FIELD_CMAP, field_norm, field_levels),
        (
            error,
            f"Absolute error (max {error_max:.3f})",
            ERROR_CMAP,
            None,
            error_levels,
        ),
    ]
    for ax, (field, title, cmap, norm, levels) in zip(axes, panels, strict=True):
        contour = ax.contourf(
            x_grid,
            t_grid,
            field,
            levels=levels,
            cmap=cmap,
            norm=norm,
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        finish_axis(ax)
        cbar = fig.colorbar(contour, ax=ax, fraction=0.045, pad=0.025)
        cbar.ax.tick_params(labelsize=8, colors=MUTED_COLOR)
        cbar.outline.set_linewidth(0.5)

    axes[0].set_ylabel("t")
    axes[0].scatter(
        np.asarray(x_obs),
        np.asarray(t_obs),
        s=11,
        c="white",
        edgecolors="#111827",
        linewidths=0.35,
        alpha=0.88,
        label="observations",
        zorder=5,
    )
    axes[0].legend(loc="upper right", fontsize=8)

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def metric_card(ax, title, value, subtitle, color):
    ax.set_axis_off()
    ax.text(0.02, 0.82, title, fontsize=10, color=MUTED_COLOR, transform=ax.transAxes)
    ax.text(
        0.02,
        0.42,
        value,
        fontsize=22,
        color=color,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(0.02, 0.15, subtitle, fontsize=9, color=MUTED_COLOR, transform=ax.transAxes)
    ax.axhline(0.02, color="#e5e7eb", linewidth=1.0)


def save_comparison_figure(results, out_path):
    """Save a polished backend comparison dashboard figure."""
    fig = plt.figure(figsize=(15.8, 7.8), constrained_layout=True)
    subfigs = fig.subfigures(3, 1, height_ratios=[0.16, 0.54, 0.30])

    title_ax = subfigs[0].subplots()
    title_ax.set_axis_off()
    title_ax.text(
        0.0,
        0.72,
        "Backend Consistency: JAX vs PyTorch PINN Tesseracts",
        fontsize=17,
        fontweight="bold",
        color=TEXT_COLOR,
        transform=title_ax.transAxes,
    )
    title_ax.text(
        0.0,
        0.26,
        (
            "Same solver-backed observations, same JAX/Optax outer loop, "
            "backend-specific apply/VJP endpoints"
        ),
        fontsize=10.5,
        color=MUTED_COLOR,
        transform=title_ax.transAxes,
    )

    ax_visc, ax_loss = subfigs[1].subplots(1, 2)
    true_nu = next(iter(results.values()))["true_viscosity"]
    for backend, result in results.items():
        style = BACKEND_STYLES[backend]
        epochs = np.arange(len(result["viscosity_history"]))
        ax_visc.plot(
            epochs,
            result["viscosity_history"],
            color=style["color"],
            linewidth=2.4,
            label=style["label"],
        )

        rows = result.get("metrics_rows", [])
        loss_epochs = np.asarray([row["epoch"] + 1 for row in rows])
        losses = np.asarray([row["loss"] for row in rows])
        ax_loss.plot(
            loss_epochs,
            losses,
            color=style["color"],
            linewidth=2.2,
            label=style["label"],
        )

    ax_visc.axhline(
        true_nu,
        color="#dc2626",
        linestyle="--",
        linewidth=1.8,
        label=f"true nu = {true_nu:.3f}",
    )
    ax_visc.set_title("Inferred viscosity trajectory")
    ax_visc.set_xlabel("epoch")
    ax_visc.set_ylabel("nu")
    ax_visc.legend(loc="lower right")
    finish_axis(ax_visc)

    ax_loss.set_yscale("log")
    ax_loss.set_title("Objective loss per epoch")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss (log scale)")
    ax_loss.legend(loc="upper right")
    finish_axis(ax_loss)

    metric_axes = subfigs[2].subplots(1, 4)
    jax_result = results["jax"]
    pytorch_result = results["pytorch"]
    gap = abs(jax_result["final_viscosity"] - pytorch_result["final_viscosity"])
    cards = [
        (
            "JAX final nu",
            f"{jax_result['final_viscosity']:.5f}",
            f"{jax_result['relative_error']:.2f}% relative error",
            BACKEND_STYLES["jax"]["color"],
        ),
        (
            "PyTorch final nu",
            f"{pytorch_result['final_viscosity']:.5f}",
            f"{pytorch_result['relative_error']:.2f}% relative error",
            BACKEND_STYLES["pytorch"]["color"],
        ),
        (
            "Backend spread",
            f"{gap:.5f}",
            "absolute difference in final nu",
            "#7c3aed",
        ),
        (
            "Measured Tesseract calls",
            f"{jax_result['metrics_rows'][-1]['apply_calls']} / "
            f"{jax_result['metrics_rows'][-1]['vjp_calls']}",
            "apply / VJP calls per gradient step",
            "#059669",
        ),
    ]
    for ax, card in zip(metric_axes, cards, strict=True):
        metric_card(ax, *card)

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def train_backend(backend, args):
    image_name = image_name_for_backend(backend)
    ensure_image_available(image_name)
    config = RunConfig(
        backend=backend,
        data=DataConfig(n_obs=args.n_obs, seed=args.seed),
        training=TrainingConfig(
            n_epochs=args.epochs,
            log_nu_learning_rate=args.learning_rate,
            param_learning_rate=args.param_learning_rate,
            n_col=args.n_col,
            n_ic=args.n_ic,
            n_bc=args.n_bc,
            viscosity_warmup_epochs=args.viscosity_warmup_epochs,
            clip_log_viscosity=args.clip_log_viscosity,
        ),
    )
    pinn = Tesseract.from_image(image_name)
    callback = FigureMetricsCallback()
    with pinn:
        result = train_inverse(
            config,
            pinn=pinn,
            callback=callback,
            metrics_every=max(1, args.epochs // 5),
        )
        result["solution_grid"] = generate_solution_grid(
            result["true_viscosity"],
            result["params_flat"],
            pinn,
            args.nx,
            args.nt,
        )
    result["metrics_rows"] = callback.rows
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--nx", type=int, default=160)
    parser.add_argument("--nt", type=int, default=90)
    parser.add_argument("--out-dir", type=Path, default=Path("img"))
    parser.add_argument("--n-obs", type=int, default=100)
    parser.add_argument("--n-col", type=int, default=500)
    parser.add_argument("--n-ic", type=int, default=50)
    parser.add_argument("--n-bc", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--param-learning-rate", type=float, default=0.001)
    parser.add_argument("--viscosity-warmup-epochs", type=int, default=25)
    parser.add_argument(
        "--clip-log-viscosity",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    configure_matplotlib()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for backend in ("jax", "pytorch"):
        print(f"Training {backend} backend for {args.epochs} epochs...")
        results[backend] = train_backend(backend, args)

    save_field_figure(
        results["jax"],
        args.out_dir / "pinn_field_solution_jax.png",
        args.nx,
        args.nt,
    )
    save_field_figure(
        results["pytorch"],
        args.out_dir / "pinn_field_solution_pytorch.png",
        args.nx,
        args.nt,
    )
    save_comparison_figure(results, args.out_dir / "pinn_solution_comparison.png")

    for path in (
        args.out_dir / "pinn_solution_comparison.png",
        args.out_dir / "pinn_field_solution_jax.png",
        args.out_dir / "pinn_field_solution_pytorch.png",
    ):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
