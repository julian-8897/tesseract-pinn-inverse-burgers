"""Tesseract component access, image guards, and field evaluation helpers.

Thin wrappers that load the local Burgers solver and PINN component APIs, map a
backend name to its container image, and evaluate the solver/PINN fields. Kept
free of any optimization or presentation logic so every higher layer can depend
on it without cycles.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import torch
from tesseract_jax import apply_tesseract

from burgers_inverse.component_loader import load_tesseract_api


def get_burgers_solver():
    """Load the in-process Burgers solver implementation."""
    return load_tesseract_api("burgers_solver").solve_burgers


def get_initial_params(backend="jax", seed=42):
    """Get initial parameters for the specified backend."""
    api = load_tesseract_api(f"pinn_{backend}")

    if backend == "jax":
        key = jax.random.PRNGKey(seed)
    else:  # pytorch
        torch.manual_seed(seed)

    if backend == "jax":
        model = api.PINNNet(key)
    else:
        model = api.PINNNet(
            hidden_sizes=[64, 64, 64],
            n_fourier_features=32,
            seed=seed,
        )
    return jnp.array(api.flatten_params(model))


def evaluate_pinn_solution_grid(
    true_viscosity,
    params_flat,
    pinn,
    nx=128,
    nt=64,
    *,
    ic_amp=1.0,
    ic_phase=0.0,
):
    """Evaluate a PINN and the Burgers solver on one visualization grid."""
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
        jnp.asarray(ic_amp, dtype=jnp.float32),
        jnp.asarray(ic_phase, dtype=jnp.float32),
    )
    return x_grid, t_grid, u_pred, np.asarray(u_solver)


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
