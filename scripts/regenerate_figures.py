"""Regenerate professional README figures from the current solver-backed path."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from tesseract_core import Tesseract

from configs import DataConfig, RunConfig, TrainingConfig
from inverse_problem import (
    TrainingCallback,
    ensure_image_available,
    evaluate_pinn_solution_grid,
    image_name_for_backend,
    train_inverse,
)
from scripts.ml_plot_style import (
    apply_ml_style,
    paper_size,
    save_figure,
    validate_publication_figure,
)

BACKEND_STYLES = {
    "jax": {"label": "JAX PINN", "color": "#2563eb"},
    "pytorch": {"label": "PyTorch PINN", "color": "#f97316"},
}
FIELD_CMAP = "RdBu_r"
ERROR_CMAP = "magma"


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
    """Use SciencePlots with compact ML-paper figure settings."""
    apply_ml_style(font_size=8.0, palette="bright", require_scienceplots=True)


def save_field_figure(result, out_path, nx, nt):
    """Save a three-panel PINN / solver / error field figure."""
    backend = result["backend"]
    style = BACKEND_STYLES[backend]
    x_obs, t_obs, _ = result["observations"]
    true_nu = result["true_viscosity"]

    if "solution_grid" in result:
        x_grid, t_grid, u_pred, u_solver = result["solution_grid"]
    else:
        pinn = Tesseract.from_image(result["tesseract_image"])
        with pinn:
            x_grid, t_grid, u_pred, u_solver = evaluate_pinn_solution_grid(
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

    fig, axes = plt.subplots(
        1,
        3,
        figsize=paper_size("double", ratio=0.38),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    panels = [
        (u_pred, f"(a) {style['label']}", FIELD_CMAP, field_norm, field_levels),
        (u_solver, "(b) Solver", FIELD_CMAP, field_norm, field_levels),
        (
            error,
            "(c) Absolute error",
            ERROR_CMAP,
            None,
            error_levels,
        ),
    ]
    contours = []
    for ax, (field, title, cmap, norm, levels) in zip(axes, panels, strict=True):
        contour = ax.contourf(
            x_grid,
            t_grid,
            field,
            levels=levels,
            cmap=cmap,
            norm=norm,
        )
        contours.append(contour)
        ax.text(
            0.01,
            1.03,
            title,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontweight="bold",
        )
        ax.set_xlabel(r"$x$")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)

    axes[0].set_ylabel(r"$t$")
    axes[0].scatter(
        np.asarray(x_obs),
        np.asarray(t_obs),
        s=11,
        c="white",
        edgecolors="#111827",
        linewidths=0.35,
        alpha=0.88,
        label="Observations",
        zorder=5,
    )

    field_colorbar = fig.colorbar(
        contours[1],
        ax=axes[:2],
        fraction=0.035,
        pad=0.02,
    )
    field_colorbar.set_label(r"$u(x,t)$")
    field_colorbar.outline.set_linewidth(0.5)
    error_colorbar = fig.colorbar(
        contours[2],
        ax=axes[2],
        fraction=0.07,
        pad=0.02,
    )
    error_colorbar.set_label(r"$|u_{\rm PINN}-u_{\rm solver}|$")
    error_colorbar.outline.set_linewidth(0.5)

    issues = validate_publication_figure(
        fig,
        allow_titles=False,
        allow_grids=False,
    )
    if issues:
        raise RuntimeError("; ".join(issues))
    save_figure(fig, Path(out_path).with_suffix(""))
    plt.close(fig)


def save_comparison_figure(results, out_path):
    """Save a two-panel backend comparison: viscosity trajectory and training loss."""
    fig, (ax_visc, ax_loss) = plt.subplots(
        1,
        2,
        figsize=paper_size("double", ratio=0.42),
        constrained_layout=True,
    )

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
        label=rf"Ground truth $\nu={true_nu:.3f}$",
    )
    ax_visc.set_xlabel("Epoch")
    ax_visc.set_ylabel(r"Inferred viscosity $\nu$")
    ax_visc.legend(loc="lower right")
    ax_visc.grid(False)

    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Training objective")
    ax_loss.legend(loc="upper right")
    ax_loss.grid(False)

    for label, ax in zip(("(a)", "(b)"), (ax_visc, ax_loss), strict=True):
        ax.text(
            0.01,
            1.02,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontweight="bold",
        )

    issues = validate_publication_figure(
        fig,
        allow_titles=False,
        allow_grids=False,
    )
    if issues:
        raise RuntimeError("; ".join(issues))
    save_figure(fig, Path(out_path).with_suffix(""))
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
        result["solution_grid"] = evaluate_pinn_solution_grid(
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
