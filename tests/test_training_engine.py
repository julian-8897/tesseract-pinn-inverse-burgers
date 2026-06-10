"""Tests for shared inverse-training engine telemetry."""

import jax.numpy as jnp
import pytest
from tesseract_jax.tesseract_compat import Jaxeract

import burgers_inverse as ip
from burgers_inverse.configs import DataConfig, RunConfig, TrainingConfig


class _TelemetryStrategy(ip.InverseStrategy):
    loss_component_names = ("total",)

    def open_components(self, stack):
        pass

    def start_context(self, warmup_epochs):
        return {}

    def run_step(self, log_viscosity, epoch):
        Jaxeract.apply(None)
        Jaxeract.vector_jacobian_product(None)
        return ip.StepResult(
            loss=1.0,
            log_v_grad=jnp.asarray(0.0),
            param_grad_norm=0.0,
            effective_weights={},
            param_count=0,
        )

    def epoch_loss_components(self, viscosity, log_viscosity, step, record):
        if not record:
            return None
        Jaxeract.apply(None)
        Jaxeract.apply(None)
        return {"total": step.loss}

    def finalize_result(self, base):
        return base


class _Capture(ip.TrainingCallback):
    def __init__(self):
        self.records = []

    def on_epoch(self, record):
        self.records.append(record)


def test_engine_counts_gradient_and_metric_dispatches(monkeypatch):
    monkeypatch.setattr(Jaxeract, "apply", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(
        Jaxeract,
        "vector_jacobian_product",
        lambda self, *args, **kwargs: None,
    )
    monkeypatch.setattr(
        Jaxeract,
        "jacobian_vector_product",
        lambda self, *args, **kwargs: None,
    )
    config = RunConfig(training=TrainingConfig(n_epochs=3))
    capture = _Capture()

    ip._run_inverse_training(
        config,
        _TelemetryStrategy(),
        callback=capture,
        metrics_every=2,
    )

    assert [record.apply_calls for record in capture.records] == [3, 1, 3]
    assert [record.vjp_calls for record in capture.records] == [1, 1, 1]


class _QuadraticStrategy(ip.InverseStrategy):
    loss_component_names = ("total",)

    def open_components(self, stack):
        pass

    def start_context(self, warmup_epochs):
        return {}

    def run_step(self, log_viscosity, epoch):
        target = jnp.log(jnp.asarray(0.08))
        error = log_viscosity - target
        return ip.StepResult(
            loss=float(error**2),
            log_v_grad=2.0 * error,
            param_grad_norm=0.0,
            effective_weights={},
            param_count=0,
        )

    def epoch_loss_components(self, viscosity, log_viscosity, step, record):
        return {"total": step.loss} if record else None

    def finalize_result(self, base):
        return base


def test_checkpoint_resume_matches_uninterrupted_optimizer_state(tmp_path):
    checkpoint = tmp_path / "training.checkpoint"
    first_config = RunConfig(
        training=TrainingConfig(n_epochs=2, log_nu_learning_rate=0.05)
    )
    resumed_config = RunConfig(
        training=TrainingConfig(n_epochs=5, log_nu_learning_rate=0.05)
    )

    ip._run_inverse_training(
        first_config,
        _QuadraticStrategy(),
        checkpoint_path=checkpoint,
        checkpoint_every=1,
        metrics_every=1,
    )
    resumed = ip._run_inverse_training(
        resumed_config,
        _QuadraticStrategy(),
        resume_from=checkpoint,
        metrics_every=1,
    )
    uninterrupted = ip._run_inverse_training(
        resumed_config,
        _QuadraticStrategy(),
        metrics_every=1,
    )

    assert resumed["start_epoch"] == 2
    assert resumed["resumed_from"] == str(checkpoint)
    assert resumed["viscosity_history"] == pytest.approx(
        uninterrupted["viscosity_history"]
    )
    assert resumed["log_viscosity_history"] == pytest.approx(
        uninterrupted["log_viscosity_history"]
    )
    assert resumed["loss_history"]["total"] == pytest.approx(
        uninterrupted["loss_history"]["total"]
    )


def test_checkpoint_resume_rejects_changed_run_configuration(tmp_path):
    checkpoint = tmp_path / "training.checkpoint"
    config = RunConfig(training=TrainingConfig(n_epochs=1))
    ip._run_inverse_training(
        config,
        _QuadraticStrategy(),
        checkpoint_path=checkpoint,
    )

    changed = RunConfig(
        data=DataConfig(seed=999),
        training=TrainingConfig(n_epochs=2),
    )
    with pytest.raises(ValueError, match="configuration does not match"):
        ip._run_inverse_training(
            changed,
            _QuadraticStrategy(),
            resume_from=checkpoint,
        )
