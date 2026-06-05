"""Tests for --mode solver-inverse (solver-adjoint inversion, Stage A baseline).

Docker-free tests check the clean-Burgers observation generator; a Docker-gated
test exercises the full solver-adjoint loop (jax.grad through the solver VJP) and
confirms it recovers nu on the well-posed inverse.
"""

import pathlib
import sys

import jax
import jax.numpy as jnp
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import inverse_problem as ip  # noqa: E402
from configs import DataConfig, ProblemConfig, RunConfig, TrainingConfig  # noqa: E402

NU_TRUE = 0.05
DOMAIN = {"x": (0.0, 1.0), "t": (0.0, 1.0)}


def test_grid_observations_shapes_and_consistency():
    """Observations sit on grid nodes and match the noiseless field there (no noise)."""
    obs = ip.generate_grid_observations(
        50, NU_TRUE, DOMAIN, jax.random.PRNGKey(0), noise_std=0.0
    )
    assert obs.x_obs.shape == (50,)
    assert obs.t_obs.shape == (50,)
    assert obs.u_obs.shape == (50,)
    assert obs.x_grid.shape == (ip._HYBRID_NX,)
    assert obs.t_grid.shape == (ip._HYBRID_NT,)
    # Indices index into the grid, and the time floor (t >= 0.05) is respected.
    assert jnp.all(obs.x_idx < ip._HYBRID_NX)
    assert jnp.all(obs.t_grid[obs.t_idx] >= 0.05 - 1e-6)
    assert jnp.all(jnp.isfinite(obs.u_obs))
    # Sampled coordinates correspond to their indices.
    assert jnp.allclose(obs.x_obs, obs.x_grid[obs.x_idx])
    assert jnp.allclose(obs.t_obs, obs.t_grid[obs.t_idx])


def test_noise_perturbs_observations():
    clean = ip.generate_grid_observations(
        60, NU_TRUE, DOMAIN, jax.random.PRNGKey(1), noise_std=0.0
    )
    noisy = ip.generate_grid_observations(
        60, NU_TRUE, DOMAIN, jax.random.PRNGKey(1), noise_std=0.05
    )
    # Same indices (same key) but values differ only by the added noise.
    assert jnp.allclose(clean.x_obs, noisy.x_obs)
    assert float(jnp.std(noisy.u_obs - clean.u_obs)) > 0.0


def test_solver_inverse_recovers_nu_full_path():
    if not ip.docker_image_available("burgers_solver"):
        pytest.skip("Tesseract image 'burgers_solver' not built; run ./buildall.sh")

    config = RunConfig(
        problem=ProblemConfig(true_viscosity=NU_TRUE, initial_viscosity=0.01),
        data=DataConfig(n_obs=80, noise_std=0.02, seed=123),
        training=TrainingConfig(n_epochs=60, log_nu_learning_rate=0.1),
    )

    class _Capture(ip.TrainingCallback):
        def on_epoch(self, record):
            self.last = record

    capture = _Capture()
    result = ip.train_solver_inverse(config, callback=capture)

    # Solver VJP dispatched every step (apply + vjp).
    assert capture.last.apply_calls >= 1
    assert capture.last.vjp_calls >= 1
    # Well-posed inverse -> recovers nu up to noise.
    assert result["relative_error"] < 5.0


if __name__ == "__main__":
    test_grid_observations_shapes_and_consistency()
    test_noise_perturbs_observations()
    print("solver-inverse Docker-free tests passed")
