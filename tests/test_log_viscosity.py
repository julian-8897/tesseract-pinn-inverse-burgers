"""Tests for log-space viscosity optimization helpers."""

import jax.numpy as jnp
import pytest

import burgers_inverse as inverse_problem
from burgers_inverse import losses


def test_compute_loss_from_log_viscosity_exponentiates_nu(monkeypatch):
    observed = {}

    def fake_compute_loss(viscosity, *args, **kwargs):
        observed["viscosity"] = float(viscosity)
        return viscosity * 2.0

    monkeypatch.setattr(losses, "compute_loss", fake_compute_loss)

    loss = inverse_problem.compute_loss_from_log_viscosity(
        jnp.log(jnp.asarray(0.05)),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )

    assert observed["viscosity"] == pytest.approx(0.05)
    assert float(loss) == pytest.approx(0.1)


def test_run_inverse_problem_rejects_non_positive_initial_viscosity():
    with pytest.raises(ValueError, match="initial_viscosity must be positive"):
        inverse_problem.run_inverse_problem(initial_viscosity=0.0)
