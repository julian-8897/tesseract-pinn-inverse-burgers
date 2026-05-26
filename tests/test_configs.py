"""Tests for typed run configuration helpers."""

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from configs import LossWeights, RunConfig
from inverse_problem import build_run_config


def test_build_run_config_applies_legacy_overrides():
    config = build_run_config(
        backend="pytorch",
        true_viscosity=0.04,
        initial_viscosity=0.02,
        n_obs=32,
        n_epochs=7,
        learning_rate=0.2,
        param_learning_rate=2e-3,
        adaptive_loss_weights=True,
        brdr_beta_c=0.99,
        brdr_beta_w=0.98,
        brdr_epsilon=1e-10,
        loss_weights={"physics": 0.25},
        seed=9,
        noise_std=0.01,
        n_col=16,
        n_ic=8,
        n_bc=6,
    )

    assert config.backend == "pytorch"
    assert config.problem.true_viscosity == 0.04
    assert config.problem.initial_viscosity == 0.02
    assert config.data.n_obs == 32
    assert config.data.seed == 9
    assert config.data.noise_std == 0.01
    assert config.training.n_epochs == 7
    assert config.training.log_nu_learning_rate == 0.2
    assert config.training.param_learning_rate == 2e-3
    assert config.training.adaptive_loss_weights is True
    assert config.training.brdr_beta_c == 0.99
    assert config.training.brdr_beta_w == 0.98
    assert config.training.brdr_epsilon == 1e-10
    assert config.training.n_col == 16
    assert config.training.n_ic == 8
    assert config.training.n_bc == 6
    assert config.loss == LossWeights(physics=0.25)


def test_run_config_with_seed_preserves_other_fields():
    config = RunConfig(backend="pytorch", loss=LossWeights(data=2.0)).with_seed(4)

    assert config.backend == "pytorch"
    assert config.data.seed == 4
    assert config.loss.data == 2.0
