"""Tests for the hybrid calibration-with-discrepancy mode (Stage 1).

Docker-free tests validate the identifiability *mechanism* using the in-process
solvers: that the KdV-Burgers truth is genuinely outside the viscous-Burgers
family (so the discrepancy is irreducible, not circular) and that differentiating
the in-loop solver recovers the viscosity. A Docker-gated integration test then
exercises the full composed Tesseract path (solver + discrepancy VJPs).
"""

import jax
import jax.numpy as jnp
import optax
import pytest

import inverse_problem as ip
from configs import (
    DataConfig,
    ProblemConfig,
    RunConfig,
    TrainingConfig,
)

NU_TRUE = 0.05
BETA = 1e-3


def _small_grid(nx=64, nt=32):
    x = jnp.linspace(0.0, 1.0, nx, endpoint=False, dtype=jnp.float32)
    t = jnp.linspace(0.0, 1.0, nt, dtype=jnp.float32)
    return x, t


def _sparse_obs(n_obs=80, seed=0, nx=64, nt=32):
    """Sparse noisy KdV observations on a small grid, with their gather indices."""
    x, t = _small_grid(nx, nt)
    u_truth = ip.solve_kdv_burgers(NU_TRUE, BETA, x, t)
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    x_idx = jax.random.randint(k1, (n_obs,), 0, nx)
    t_idx = jax.random.randint(k2, (n_obs,), 2, nt)
    u_obs = u_truth[t_idx, x_idx] + jax.random.normal(k3, (n_obs,)) * 0.02
    return x, t, x_idx, t_idx, u_obs


# --- Docker-free: the discrepancy is real (guards against the circularity trap) ---


def test_kdv_beta_zero_matches_in_loop_solver():
    """With beta=0 the truth reduces to viscous Burgers -> discrepancy ~ 0."""
    x, t = _small_grid()
    solve_burgers = ip.get_burgers_solver()
    u_burgers = solve_burgers(
        jnp.float32(NU_TRUE), x, t, jnp.float32(1.0), jnp.float32(0.0)
    )
    u_kdv0 = ip.solve_kdv_burgers(NU_TRUE, 0.0, x, t)
    assert float(jnp.max(jnp.abs(u_kdv0 - u_burgers))) < 1e-3


def test_kdv_dispersion_is_irreducible_model_error():
    """beta>0 produces a clear, finite discrepancy the in-loop solver cannot match."""
    x, t = _small_grid()
    solve_burgers = ip.get_burgers_solver()
    u_burgers = solve_burgers(
        jnp.float32(NU_TRUE), x, t, jnp.float32(1.0), jnp.float32(0.0)
    )
    u_kdv = ip.solve_kdv_burgers(NU_TRUE, BETA, x, t)
    discrepancy_rms = float(jnp.sqrt(jnp.mean((u_kdv - u_burgers) ** 2)))
    field_rms = float(jnp.sqrt(jnp.mean(u_burgers**2)))
    assert jnp.all(jnp.isfinite(u_kdv))
    # Clearly nonzero (something to learn) but not overwhelming (nu still recoverable).
    assert 0.01 < discrepancy_rms < 0.5 * field_rms


# --- Docker-free: differentiating the in-loop solver recovers the viscosity ---


def _solver_only_data_loss(log_nu, x, t, x_idx, t_idx, u_obs):
    solve_burgers = ip.get_burgers_solver()
    field = solve_burgers(jnp.exp(log_nu), x, t, jnp.float32(1.0), jnp.float32(0.0))
    return jnp.mean((field[t_idx, x_idx] - u_obs) ** 2)


def test_data_loss_is_minimized_near_true_nu():
    x, t, x_idx, t_idx, u_obs = _sparse_obs()
    losses = {
        nu: float(
            _solver_only_data_loss(jnp.log(jnp.float32(nu)), x, t, x_idx, t_idx, u_obs)
        )
        for nu in (0.5 * NU_TRUE, NU_TRUE, 2.0 * NU_TRUE)
    }
    assert losses[NU_TRUE] < losses[0.5 * NU_TRUE]
    assert losses[NU_TRUE] < losses[2.0 * NU_TRUE]


def test_inprocess_solver_inversion_recovers_nu():
    """Gradient descent through the in-process solver recovers nu (the mechanism the
    hybrid relies on). A residual bias from the un-modeled dispersion is expected and
    is exactly what the learned discrepancy removes in the full hybrid."""
    x, t, x_idx, t_idx, u_obs = _sparse_obs()
    grad_fn = jax.value_and_grad(_solver_only_data_loss)

    log_nu = jnp.log(jnp.float32(0.01))
    optimizer = optax.adam(0.1)
    opt_state = optimizer.init(log_nu)
    for _ in range(40):
        _, grad = grad_fn(log_nu, x, t, x_idx, t_idx, u_obs)
        updates, opt_state = optimizer.update(grad, opt_state)
        log_nu = optax.apply_updates(log_nu, updates)

    recovered = float(jnp.exp(log_nu))
    rel_error = abs(recovered - NU_TRUE) / NU_TRUE
    assert jnp.isfinite(log_nu)
    assert rel_error < 0.25  # recovers nu, modulo the dispersion-induced bias


# --- Docker-gated: the full composed Tesseract path (both VJPs fire) ---


class _CaptureCallback(ip.TrainingCallback):
    def __init__(self):
        self.last = None

    def on_epoch(self, record):
        self.last = record


def test_hybrid_full_path_composes_both_tesseract_vjps():
    for image in ("burgers_solver", ip.image_name_for_backend("jax")):
        if not ip.docker_image_available(image):
            pytest.skip(f"Tesseract image '{image}' not built; run ./buildall.sh")

    config = RunConfig(
        backend="jax",
        problem=ProblemConfig(
            true_viscosity=NU_TRUE, initial_viscosity=0.02, dispersion_beta=BETA
        ),
        data=DataConfig(n_obs=60, noise_std=0.02, seed=123),
        training=TrainingConfig(
            n_epochs=25,
            log_nu_learning_rate=0.1,
            param_learning_rate=1e-3,
            n_col=128,
            discrepancy_reg_weight=1.0,
        ),
    )

    capture = _CaptureCallback()
    result = ip.train_hybrid_inverse(config, callback=capture)

    # The composition proof: both the solver apply and the discrepancy apply (plus
    # their VJPs) dispatch every step -> >= 3 of each (solver + obs + reg).
    assert capture.last is not None
    assert capture.last.apply_calls >= 3
    assert capture.last.vjp_calls >= 3

    final_nu = result["final_viscosity"]
    assert jnp.isfinite(jnp.asarray(final_nu))
    # Moved toward the truth from the initial guess.
    assert abs(final_nu - NU_TRUE) < abs(0.02 - NU_TRUE)


if __name__ == "__main__":
    test_kdv_beta_zero_matches_in_loop_solver()
    test_kdv_dispersion_is_irreducible_model_error()
    test_data_loss_is_minimized_near_true_nu()
    test_inprocess_solver_inversion_recovers_nu()
    print("hybrid Docker-free mechanism tests passed")
