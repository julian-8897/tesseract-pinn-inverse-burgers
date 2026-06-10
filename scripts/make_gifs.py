"""Generate the two README hero GIFs in the repository's publication style.

1. ``img/burgers_evolution.gif`` -- the space-time field u(x, t) building up in time at
   a low and a high viscosity, side by side. The shock shows up as a sharp slanted band
   at low viscosity and a diffuse one at high viscosity.
2. ``img/posterior_tracking.gif`` -- the amortized FMPE posterior over nu tracking a
   ground truth that sweeps across the prior. One trained network, many observations.

Both run in-process (no Docker): the solver via ``get_burgers_solver`` and the
posterior via the trained ``posterior.pkl`` bundle. Styling matches the static figures
through ``scripts.ml_plot_style`` (SciencePlots, no titles, no grids, inward ticks);
all descriptive text lives in the README captions, not in figure titles.

    uv run python -m scripts.make_gifs
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from burgers_inverse import SOLVER_NX, get_burgers_solver
from scripts.ml_plot_style import apply_ml_style, paper_size

# Match the static figures: RdBu_r fields and the Okabe-Ito posterior palette.
FIELD_CMAP = "RdBu_r"
NU_COLOR = "#0072B2"
IMG = pathlib.Path("img")


def optimize_gif(path, *, colors=160):
    """Re-encode a GIF with one shared adaptive palette to cut file size."""
    from PIL import Image, ImageSequence

    source = Image.open(path)
    duration = source.info.get("duration", 50)
    frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(source)]
    palette = frames[-1].quantize(colors=colors, method=Image.FASTOCTREE)
    quantized = [frame.quantize(palette=palette, dither=Image.NONE) for frame in frames]
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        loop=0,
        duration=duration,
        optimize=True,
        disposal=2,
    )


def make_burgers_evolution(
    path, *, nu_low=0.01, nu_high=0.08, nt=110, frames=50, fps=12
):
    """Animate the space-time field u(x, t) filling in for low and high viscosity."""
    solve = get_burgers_solver()
    x = np.linspace(0.0, 1.0, SOLVER_NX, endpoint=False, dtype=np.float32)
    t = np.linspace(0.0, 1.0, nt, dtype=np.float32)
    one, zero = np.float32(1.0), np.float32(0.0)
    fields = [
        np.asarray(solve(np.float32(nu_low), x, t, one, zero)),
        np.asarray(solve(np.float32(nu_high), x, t, one, zero)),
    ]
    vmax = float(max(np.abs(f).max() for f in fields))

    cmap = plt.get_cmap(FIELD_CMAP).copy()
    cmap.set_bad("white", 1.0)

    fig, axes = plt.subplots(1, 2, figsize=paper_size("double", ratio=0.52))
    images, scans = [], []
    for ax, nu, field in zip(axes, (nu_low, nu_high), fields, strict=True):
        blank = np.full_like(field, np.nan)
        im = ax.imshow(
            blank,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            aspect="auto",
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        scan = ax.axhline(0.0, color="black", linewidth=0.9)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$t$")
        ax.text(
            0.05,
            0.92,
            rf"$\nu = {nu:g}$",
            transform=ax.transAxes,
            ha="left",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.5},
        )
        images.append(im)
        scans.append(scan)

    colorbar = fig.colorbar(images[-1], ax=axes, pad=0.02, fraction=0.046)
    colorbar.set_label(r"$u(x, t)$")
    colorbar.outline.set_linewidth(0.5)

    def update(frame):
        k = int(round((frame + 1) / frames * nt))
        k = max(1, min(k, nt))
        for im, field in zip(images, fields, strict=True):
            shown = field.copy()
            shown[k:, :] = np.nan
            im.set_array(shown)
        for scan in scans:
            scan.set_ydata([t[k - 1], t[k - 1]])
        return (*images, *scans)

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    optimize_gif(path)
    print(f"wrote {path}")


def make_posterior_tracking(path, *, n_frames=36, n_samples=1500, fps=9):
    """Animate the nu posterior following a ground truth that sweeps the prior."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from burgers_inverse.fmpe_posterior import load_model, observation_from_theta

    bundle = load_model("tesseracts/fmpe_posterior/posterior.pkl")
    contract = bundle["metadata"]["contract"]
    sensors = bundle["sensors"]
    posterior = bundle["posterior"]
    noise_std = float(contract["noise_std"])
    lo, hi = float(contract["prior_low"][0]), float(contract["prior_high"][0])

    sweep = np.linspace(lo + 0.006, hi - 0.006, n_frames)
    nu_samples = []
    for nu_true in sweep:
        obs = observation_from_theta(
            (float(nu_true), 1.0, 0.0), sensors, noise_std=noise_std, seed=7
        )
        torch.manual_seed(0)
        draws = posterior.sample((n_samples,), x=obs, show_progress_bars=False)
        nu_samples.append(np.asarray(draws[:, 0], dtype=np.float64))

    bins = 32
    ymax = max(
        np.histogram(s, bins=bins, range=(lo, hi), density=True)[0].max()
        for s in nu_samples
    )

    handles = [
        Patch(facecolor=NU_COLOR, alpha=0.30, label="Posterior"),
        Line2D([0], [0], color=NU_COLOR, linewidth=1.5, label="Posterior median"),
        Line2D(
            [0], [0], color="black", linestyle="--", linewidth=1.0, label="Ground truth"
        ),
    ]

    fig, ax = plt.subplots(figsize=paper_size("double", ratio=0.5))

    def update(frame):
        ax.clear()
        s = nu_samples[frame]
        q05, q50, q95 = np.percentile(s, [5, 50, 95])
        ax.hist(
            s,
            bins=bins,
            range=(lo, hi),
            density=True,
            histtype="stepfilled",
            color=NU_COLOR,
            edgecolor=NU_COLOR,
            alpha=0.30,
        )
        ax.axvspan(q05, q95, color=NU_COLOR, alpha=0.12)
        ax.axvline(q50, color=NU_COLOR, linewidth=1.5)
        ax.axvline(sweep[frame], color="black", linestyle="--", linewidth=1.0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, ymax * 1.1)
        ax.set_xlabel(r"Viscosity $\nu$")
        ax.set_ylabel("Posterior density")
        ax.legend(handles=handles, loc="upper right")

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=110)
    plt.close(fig)
    optimize_gif(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    IMG.mkdir(exist_ok=True)
    apply_ml_style(font_size=8.0, palette="bright", require_scienceplots=True)
    make_burgers_evolution(IMG / "burgers_evolution.gif")
    make_posterior_tracking(IMG / "posterior_tracking.gif")
