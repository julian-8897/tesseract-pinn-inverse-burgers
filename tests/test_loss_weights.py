"""Tests for configurable PINN loss weights."""

import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from inverse_problem import DEFAULT_LOSS_WEIGHTS, normalize_loss_weights


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
