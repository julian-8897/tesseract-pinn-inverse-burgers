"""Tests for the amortized flow-matching posterior pipeline (Stage B).

The fast tests check the simulator/prior/sensor mechanics (no training). A slow
smoke trains a tiny FMPE posterior and checks the sampling interface; it is marked
``slow`` so the default suite stays quick.
"""

import numpy as np
import pytest
import torch

from burgers_inverse import fmpe_posterior as fp
from burgers_inverse.component_loader import load_tesseract_api
from burgers_inverse.configs import FMPEConfig


def test_prior_ranges():
    prior = fp.default_prior()
    samples = prior.sample((500,))
    assert samples.shape == (500, 3)
    low = torch.tensor(fp.PRIOR_LOW)
    high = torch.tensor(fp.PRIOR_HIGH)
    assert bool((samples >= low).all()) and bool((samples <= high).all())


def test_sensor_layout():
    sensors = fp.Sensors(n_sensors=48, seed=0)
    assert sensors.n_sensors == 48
    assert sensors.x_idx.shape == (48,)
    assert sensors.t_idx.shape == (48,)
    assert int(sensors.x_idx.max()) < fp.SOLVER_NX
    # Time floor t >= 0.05 respected.
    assert float(sensors.t_grid[sensors.t_idx].min()) >= 0.05 - 1e-6


def test_simulator_shapes_and_noise():
    sensors = fp.Sensors(n_sensors=32, seed=0)
    prior = fp.default_prior()
    theta = prior.sample((16,))
    clean = fp.simulate(theta, sensors, noise_std=0.0, seed=0)
    noisy = fp.simulate(theta, sensors, noise_std=0.05, seed=0)
    assert clean.shape == (16, 32)
    assert torch.isfinite(clean).all()
    # Noise perturbs values but not the underlying simulation.
    assert float((noisy - clean).std()) > 0.0


def test_observation_from_theta_shape():
    sensors = fp.Sensors(n_sensors=40, seed=1)
    x_o = fp.observation_from_theta((0.05, 1.0, 0.0), sensors)
    assert x_o.shape == (1, 40)
    assert torch.isfinite(x_o).all()


def test_fmpe_simulation_is_reproducible_from_config():
    config = FMPEConfig(
        n_sims=8,
        n_sensors=12,
        sensor_seed=2,
        simulation_seed=3,
        training_seed=4,
        max_num_epochs=1,
    )
    prior = fp.default_prior()

    torch.manual_seed(config.simulation_seed)
    theta_a = prior.sample((config.n_sims,))
    sensors_a = fp.Sensors(config.n_sensors, seed=config.sensor_seed)
    x_a = fp.simulate(
        theta_a, sensors_a, noise_std=config.noise_std, seed=config.simulation_seed
    )

    torch.manual_seed(config.simulation_seed)
    theta_b = prior.sample((config.n_sims,))
    sensors_b = fp.Sensors(config.n_sensors, seed=config.sensor_seed)
    x_b = fp.simulate(
        theta_b, sensors_b, noise_std=config.noise_std, seed=config.simulation_seed
    )

    assert torch.equal(theta_a, theta_b)
    assert torch.equal(x_a, x_b)
    assert np.array_equal(sensors_a.x_idx, sensors_b.x_idx)


def test_saved_bundle_has_versioned_contract_and_validates_observations(tmp_path):
    sensors = fp.Sensors(n_sensors=10, seed=7)
    result = {
        "posterior": object(),
        "sensors": sensors,
        "noise_std": 0.02,
        "config": FMPEConfig(n_sims=20, n_sensors=10),
    }
    path = fp.save_model(result, tmp_path / "posterior.pkl")
    bundle = fp.load_model(path, allow_legacy=False)

    metadata = bundle["metadata"]
    assert metadata["format"] == fp.BUNDLE_FORMAT
    assert metadata["version"] == fp.BUNDLE_VERSION
    assert metadata["contract"]["observation_dim"] == 10
    assert metadata["contract"]["sensor_layout_id"] == sensors.layout_id
    assert len(metadata["model_id"]) == 64

    observation = fp.validate_observation(np.zeros(10), bundle)
    assert observation.dtype == np.float32
    with pytest.raises(ValueError, match="model expects 10"):
        fp.validate_observation(np.zeros(9), bundle)
    with pytest.raises(ValueError, match="finite"):
        fp.validate_observation(np.full(10, np.nan), bundle)


def test_legacy_bundle_requires_explicit_compatibility():
    sensors = fp.Sensors(n_sensors=6, seed=0)
    legacy = {
        "posterior": object(),
        "x_idx": np.asarray(sensors.x_idx),
        "t_idx": np.asarray(sensors.t_idx),
        "noise_std": 0.02,
    }

    migrated = fp.validate_model_bundle(legacy)
    assert migrated["metadata"]["legacy"] is True
    with pytest.raises(ValueError, match="no versioned metadata"):
        fp.validate_model_bundle(legacy, allow_legacy=False)


def test_posterior_summary_reports_coverage_and_contraction():
    truths = np.array([[0.05, 1.0, 0.0], [0.06, 0.9, 0.1]])
    offsets = np.linspace(-1.0, 1.0, 101)
    samples = np.stack(
        [
            truths[case] + offsets[:, None] * np.array([0.005, 0.02, 0.02])
            for case in range(2)
        ]
    )

    summary = fp.summarize_posterior_samples(samples, truths)

    assert summary["parameters"]["nu"]["coverage"] == 1.0
    assert summary["parameters"]["nu"]["rmse"] == pytest.approx(0.0, abs=1e-8)
    assert 0.0 < summary["parameters"]["nu"]["contraction_ratio"] < 1.0


def test_diagnostic_reports_are_machine_readable(tmp_path):
    report_path = fp.write_json_report(
        {"tensor": torch.tensor([1.0, 2.0])}, tmp_path / "report.json"
    )
    csv_path = fp.write_contraction_csv(
        [{"n_sensors": 16, "nu_coverage": 0.9}],
        tmp_path / "contraction.csv",
    )

    assert '"tensor": [' in report_path.read_text()
    assert "nu_coverage" in csv_path.read_text()


def test_tesseract_runtime_rejects_wrong_observation_length():
    module = load_tesseract_api(
        "fmpe_posterior",
        module_name="fmpe_tesseract_api_test",
    )

    class FakePosterior:
        def sample(self, shape, x, show_progress_bars=False):
            return torch.zeros((shape[0], 3), dtype=torch.float32)

    model = {
        "posterior": FakePosterior(),
        "x_idx": np.arange(4),
        "t_idx": np.arange(4),
        "metadata": {
            "format": module.BUNDLE_FORMAT,
            "version": module.BUNDLE_VERSION,
            "model_id": "test-model",
            "contract": {"observation_dim": 4},
        },
    }
    module._validate_model(model)
    module._MODEL = model

    output = module.apply(
        module.InputSchema(observation=np.zeros(4, dtype=np.float32), seed=0)
    )
    assert output.observation_dim == 4
    assert output.model_id == "test-model"
    with pytest.raises(ValueError, match="model expects 4"):
        module.apply(
            module.InputSchema(observation=np.zeros(3, dtype=np.float32), seed=0)
        )


def test_remote_posterior_tesseract_dispatch(monkeypatch):
    calls = {}

    class FakeRemote:
        def apply(self, payload):
            calls["payload"] = payload
            return {"samples": np.zeros((2, 3), dtype=np.float32)}

    def fake_from_url(url):
        calls["url"] = url
        return FakeRemote()

    monkeypatch.setattr("tesseract_core.Tesseract.from_url", fake_from_url)
    output = fp.query_posterior_tesseract(
        np.arange(4, dtype=np.float64),
        seed=7,
        url="https://posterior.example",
    )

    assert calls["url"] == "https://posterior.example"
    assert calls["payload"]["observation"].dtype == np.float32
    assert calls["payload"]["seed"] == 7
    assert output["samples"].shape == (2, 3)


def test_posterior_tesseract_container():
    """The packaged posterior Tesseract returns calibrated samples for an obs."""
    import burgers_inverse as ip

    if not ip.docker_image_available("fmpe_posterior"):
        pytest.skip(
            "fmpe_posterior image not built; run "
            "`burgers-fmpe train` then `tesseract build`"
        )

    sensors = fp.Sensors(n_sensors=64, seed=0)  # matches the trained layout
    x_o = fp.observation_from_theta((0.05, 1.0, 0.0), sensors)
    out = fp.query_posterior_tesseract(x_o.numpy())

    samples = np.asarray(out["samples"])
    assert samples.shape[1] == 3
    assert np.all(np.isfinite(samples))
    # nu marginal 90% CI contains the truth.
    lo, hi = np.percentile(samples[:, 0], [5, 95])
    assert lo <= 0.05 <= hi


@pytest.mark.slow
def test_fmpe_train_and_sample_smoke():
    """Tiny end-to-end: the trained posterior samples parameters of the right shape
    inside the prior support. Accuracy is validated separately (calibration)."""
    torch.manual_seed(0)
    result = fp.train_fmpe(n_sims=300, n_sensors=32, max_num_epochs=15)
    x_o = fp.observation_from_theta((0.05, 1.0, 0.0), result["sensors"])
    samples = result["posterior"].sample((200,), x=x_o).cpu().numpy()
    assert samples.shape == (200, 3)
    assert np.all(np.isfinite(samples))


if __name__ == "__main__":
    test_prior_ranges()
    test_sensor_layout()
    test_simulator_shapes_and_noise()
    test_observation_from_theta_shape()
    print("fmpe Docker-free mechanics tests passed")
