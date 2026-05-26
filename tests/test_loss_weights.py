"""Tests for configurable PINN loss weights."""

import pathlib
import sys

import pytest
import jax.numpy as jnp

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from inverse_problem import (
    DEFAULT_LOSS_WEIGHTS,
    initialize_brdr_state,
    normalize_loss_weights,
    summarize_brdr_weights,
    update_brdr_state,
    validate_brdr_loss_weights,
)


def test_normalize_loss_weights_fills_defaults():
    weights = normalize_loss_weights({"physics": 0.25})

    assert weights == {
        **DEFAULT_LOSS_WEIGHTS,
        "physics": 0.25,
    }


def test_normalize_loss_weights_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown loss weight"):
        normalize_loss_weights({"bad": 1.0})


def test_normalize_loss_weights_rejects_negative_values():
    with pytest.raises(ValueError, match="non-negative"):
        normalize_loss_weights({"data": -1.0})


def test_validate_brdr_loss_weights_accepts_positive_weights():
    validate_brdr_loss_weights(DEFAULT_LOSS_WEIGHTS)


def test_validate_brdr_loss_weights_rejects_zero_weights():
    with pytest.raises(ValueError, match="positive fixed weights"):
        validate_brdr_loss_weights({"physics": 0.0})


def test_update_brdr_state_keeps_unit_global_mean_weight():
    pointwise_losses = {
        "data": jnp.array([1.0, 0.25]),
        "physics": jnp.array([0.1, 0.2, 0.3]),
        "ic": jnp.array([0.5]),
        "bc": jnp.array([0.05, 0.1]),
    }
    state = initialize_brdr_state(pointwise_losses)
    updated = update_brdr_state(
        state, pointwise_losses, beta_c=0.9, beta_w=0.0
    )

    all_weights = jnp.concatenate(
        [jnp.ravel(updated["weights"][name]) for name in DEFAULT_LOSS_WEIGHTS]
    )
    summary = summarize_brdr_weights(updated["weights"])

    assert float(jnp.mean(all_weights)) == pytest.approx(1.0)
    assert set(summary) == set(DEFAULT_LOSS_WEIGHTS)
