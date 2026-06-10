"""Synthetic observation samplers backed by the viscous-Burgers solver.

Clean (well-posed) observations for the deterministic inverse methods. The
experimental KdV-Burgers truth and its sampler live in
:mod:`burgers_inverse.experimental`; both share the :class:`GridObservations`
container and the time-floored index sampler defined here.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from burgers_inverse.components import get_burgers_solver
from burgers_inverse.configs import DEFAULT_NOISE_STD
from burgers_inverse.constants import MIN_OBS_TIME, SOLVER_NT, SOLVER_NX


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
