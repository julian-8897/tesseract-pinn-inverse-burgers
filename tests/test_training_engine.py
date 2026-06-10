"""Tests for shared inverse-training engine telemetry."""

import jax.numpy as jnp
from tesseract_jax.tesseract_compat import Jaxeract

import burgers_inverse as ip
from burgers_inverse.configs import RunConfig, TrainingConfig


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
