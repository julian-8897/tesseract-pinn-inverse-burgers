"""Regression tests for inverse-problem observation generation."""

import pathlib
import sys

import jax
import jax.numpy as jnp

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from inverse_problem import generate_observations


def test_generate_observations_uses_burgers_solver_not_heat_equation():
    n_points = 64
    true_viscosity = 0.05
    domain = {"x": (0.0, 1.0), "t": (0.0, 1.0)}
    key = jax.random.PRNGKey(123)

    x_obs, t_obs, u_obs = generate_observations(
        n_points, true_viscosity, domain, key
    )

    assert x_obs.shape == (n_points,)
    assert t_obs.shape == (n_points,)
    assert u_obs.shape == (n_points,)
    assert jnp.all(jnp.isfinite(u_obs))
    assert float(jnp.min(t_obs)) >= 0.05

    _, _, noise_key = jax.random.split(key, 3)
    noise = jax.random.normal(noise_key, (n_points,)) * 0.02
    denoised_u_obs = u_obs - noise
    heat_equation_u = jnp.sin(2 * jnp.pi * x_obs) * jnp.exp(
        -true_viscosity * (2 * jnp.pi) ** 2 * t_obs
    )

    max_difference = jnp.max(jnp.abs(denoised_u_obs - heat_equation_u))
    assert float(max_difference) > 1e-2
