"""Generate the two README hero GIFs.

1. ``img/burgers_evolution.gif`` -- the Burgers field u(x, t) evolving in time at a
   low and a high viscosity, side by side, so the effect of nu is visible.
2. ``img/posterior_tracking.gif`` -- the amortized FMPE posterior over nu tracking a
   ground truth that sweeps across the prior. One trained network, many observations.

Both run in-process (no Docker): the solver via ``get_burgers_solver`` and the
posterior via the trained ``posterior.pkl`` bundle.

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

from inverse_problem import SOLVER_NX, get_burgers_solver

PINN_COLOR = "#1f77b4"
TRUE_COLOR = "#c44e52"
FILL_COLOR = "#55a868"
IMG = pathlib.Path("img")


def make_burgers_evolution(path, *, nu_low=0.01, nu_high=0.08, nt=70, fps=20):
    """Animate u(x, t) for a low and high viscosity, side by side."""
    solve = get_burgers_solver()
    x = np.linspace(0.0, 1.0, SOLVER_NX, endpoint=False, dtype=np.float32)
    t = np.linspace(0.0, 1.0, nt, dtype=np.float32)
    one, zero = np.float32(1.0), np.float32(0.0)
    u_low = np.asarray(solve(np.float32(nu_low), x, t, one, zero))
    u_high = np.asarray(solve(np.float32(nu_high), x, t, one, zero))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    lines = []
    for ax, nu, color in (
        (axes[0], nu_low, PINN_COLOR),
        (axes[1], nu_high, TRUE_COLOR),
    ):
        (line,) = ax.plot([], [], color=color, linewidth=2.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(-1.15, 1.15)
        ax.set_xlabel("x")
        ax.set_ylabel("u(x, t)")
        ax.set_title(f"ν = {nu:g}")
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        lines.append(line)
    suptitle = fig.suptitle("", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    def update(frame):
        lines[0].set_data(x, u_low[frame])
        lines[1].set_data(x, u_high[frame])
        suptitle.set_text(f"Burgers field at t = {t[frame]:.2f}")
        return (*lines, suptitle)

    anim = FuncAnimation(fig, update, frames=nt, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    print(f"wrote {path}")


def make_posterior_tracking(path, *, n_frames=36, n_samples=1200, fps=14):
    """Animate the nu posterior following a ground truth that sweeps the prior."""
    from fmpe_posterior import load_model, observation_from_theta

    bundle = load_model("tesseracts/fmpe_posterior/posterior.pkl")
    contract = bundle["metadata"]["contract"]
    sensors = bundle["sensors"]
    posterior = bundle["posterior"]
    noise_std = float(contract["noise_std"])
    lo, hi = float(contract["prior_low"][0]), float(contract["prior_high"][0])

    sweep = np.linspace(lo + 0.005, hi - 0.005, n_frames)
    nu_samples = []
    for nu_true in sweep:
        obs = observation_from_theta(
            (float(nu_true), 1.0, 0.0), sensors, noise_std=noise_std, seed=7
        )
        torch.manual_seed(0)
        draws = posterior.sample((n_samples,), x=obs, show_progress_bars=False)
        nu_samples.append(np.asarray(draws[:, 0], dtype=np.float64))

    fig, ax = plt.subplots(figsize=(8, 4.2))

    def update(frame):
        ax.clear()
        s = nu_samples[frame]
        q05, q95 = np.percentile(s, [5, 95])
        ax.hist(s, bins=40, range=(lo, hi), density=True, color=PINN_COLOR, alpha=0.85)
        ax.axvspan(q05, q95, color=FILL_COLOR, alpha=0.18, label="90% credible")
        ax.axvline(
            sweep[frame], color=TRUE_COLOR, linestyle="--", linewidth=2, label="true ν"
        )
        ax.axvline(
            s.mean(),
            color="#333333",
            linestyle=":",
            linewidth=1.5,
            label="posterior mean",
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 170)
        ax.set_xlabel("viscosity ν")
        ax.set_ylabel("posterior density")
        ax.set_title("One trained FMPE network, posterior follows the truth")
        ax.legend(frameon=False, loc="upper right")
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    IMG.mkdir(exist_ok=True)
    make_burgers_evolution(IMG / "burgers_evolution.gif")
    make_posterior_tracking(IMG / "posterior_tracking.gif")
