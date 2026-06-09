"""Regression tests for inverse-problem observation generation."""

import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import inverse_problem as ip


def test_generate_observations_uses_burgers_solver_not_heat_equation():
    n_points = 64
    true_viscosity = 0.05
    domain = {"x": (0.0, 1.0), "t": (0.0, 1.0)}
    key = jax.random.PRNGKey(123)

    x_obs, t_obs, u_obs = ip.generate_observations(
        n_points,
        true_viscosity,
        domain,
        key,
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


def test_evaluate_pinn_solution_grid_shares_one_grid(monkeypatch):
    def fake_apply_tesseract(_pinn, inputs):
        return {"u_pred": inputs["x"] + inputs["t"]}

    def fake_solver(nu, x, t, ic_amp, ic_phase):
        return (
            t[:, None] + x[None, :] + nu + jnp.asarray(ic_amp) + jnp.asarray(ic_phase)
        )

    monkeypatch.setattr(ip, "apply_tesseract", fake_apply_tesseract)
    monkeypatch.setattr(ip, "get_burgers_solver", lambda: fake_solver)

    x_grid, t_grid, u_pred, u_solver = ip.evaluate_pinn_solution_grid(
        0.05,
        np.zeros(2, dtype=np.float32),
        object(),
        nx=8,
        nt=5,
        ic_amp=0.8,
        ic_phase=0.1,
    )

    assert x_grid.shape == t_grid.shape == u_pred.shape == u_solver.shape == (5, 8)
    np.testing.assert_allclose(u_pred, x_grid + t_grid, rtol=1e-6)
    np.testing.assert_allclose(
        u_solver,
        x_grid + t_grid + 0.95,
        rtol=1e-6,
    )
