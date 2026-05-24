"""Tests for seed-sweep result aggregation."""

import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from inverse_problem import summarize_seed_results


def test_summarize_seed_results_groups_by_backend():
    summary = summarize_seed_results(
        [
            {
                "backend": "jax",
                "final_viscosity": 0.04,
                "relative_error": 20.0,
                "avg_time_ms": 100.0,
            },
            {
                "backend": "jax",
                "final_viscosity": 0.06,
                "relative_error": 20.0,
                "avg_time_ms": 120.0,
            },
            {
                "backend": "pytorch",
                "final_viscosity": 0.05,
                "relative_error": 0.0,
                "avg_time_ms": 200.0,
            },
        ]
    )

    assert summary["jax"]["runs"] == 2
    assert summary["jax"]["mean_viscosity"] == pytest.approx(0.05)
    assert summary["jax"]["std_viscosity"] == pytest.approx(0.01)
    assert summary["jax"]["mean_time_ms"] == pytest.approx(110.0)
    assert summary["pytorch"]["runs"] == 1
    assert summary["pytorch"]["std_relative_error"] == pytest.approx(0.0)
