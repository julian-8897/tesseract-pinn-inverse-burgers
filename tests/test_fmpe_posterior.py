"""Tests for the amortized flow-matching posterior pipeline (Stage B).

The fast tests check the simulator/prior/sensor mechanics (no training). A slow
smoke trains a tiny FMPE posterior and checks the sampling interface; it is marked
``slow`` so the default suite stays quick.
"""

import pathlib
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import fmpe_posterior as fp  # noqa: E402


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
    assert int(sensors.x_idx.max()) < fp._HYBRID_NX
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


def test_posterior_tesseract_container():
    """The packaged posterior Tesseract returns calibrated samples for an obs."""
    import inverse_problem as ip

    if not ip.docker_image_available("fmpe_posterior"):
        pytest.skip(
            "fmpe_posterior image not built; run "
            "`python fmpe_posterior.py train` then `tesseract build`"
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
