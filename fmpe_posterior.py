"""Amortized flow-matching posterior over Burgers parameters (Stage B).

Simulation-based inference: draw ``theta = (nu, ic_amp, ic_phase)`` from a prior,
simulate sparse noisy observations with the differentiable JAX Burgers solver, and
train an amortized **Flow Matching Posterior Estimation** (FMPE) network with
``sbi`` (flow backend: ``zuko``). The trained posterior maps a fresh observation
vector to samples of ``theta`` -- in particular a marginal posterior over the
viscosity ``nu``, with the initial-condition parameters marginalized.

Training simulations use the in-process JAX solver (fast, vmapped) -- the same
physics the ``burgers_solver`` Tesseract serves. The solver Tesseract remains the
packaged component used at inference / for the optional Stage-C refinement.
"""

from __future__ import annotations

import csv
import json
import pathlib
import pickle
import random
from dataclasses import asdict, replace
from hashlib import sha256

import jax
import jax.numpy as jnp
import numpy as np
import torch
from sbi.inference import FMPE
from sbi.utils import BoxUniform

from configs import (
    DEFAULT_FMPE_PRIOR_HIGH,
    DEFAULT_FMPE_PRIOR_LOW,
    DEFAULT_NOISE_STD,
    FMPEConfig,
)
from inverse_problem import (
    MIN_OBS_TIME,
    SOLVER_NT,
    SOLVER_NX,
    get_burgers_solver,
)

# theta = (nu, ic_amp, ic_phase)
PARAM_NAMES = ("nu", "ic_amp", "ic_phase")
PRIOR_LOW = DEFAULT_FMPE_PRIOR_LOW
PRIOR_HIGH = DEFAULT_FMPE_PRIOR_HIGH
BUNDLE_FORMAT = "tesseract-fmpe-posterior"
BUNDLE_VERSION = 1

_solve_burgers = get_burgers_solver()


def seed_random_generators(seed: int):
    """Seed Python, NumPy, and Torch RNGs used by SBI training and diagnostics."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_prior(device: str = "cpu", low=PRIOR_LOW, high=PRIOR_HIGH) -> BoxUniform:
    """Box-uniform prior over (nu, ic_amp, ic_phase)."""
    return BoxUniform(
        low=torch.tensor(low, dtype=torch.float32, device=device),
        high=torch.tensor(high, dtype=torch.float32, device=device),
    )


class Sensors:
    """A fixed sparse sensor layout on the solver grid (shared across simulations)."""

    def __init__(self, n_sensors: int = 64, seed: int = 0, x_idx=None, t_idx=None):
        domain_x, domain_t = (0.0, 1.0), (0.0, 1.0)
        self.x_grid = jnp.linspace(
            *domain_x, SOLVER_NX, endpoint=False, dtype=jnp.float32
        )
        self.t_grid = jnp.linspace(*domain_t, SOLVER_NT, dtype=jnp.float32)
        if x_idx is not None and t_idx is not None:
            self.x_idx = jnp.asarray(x_idx)
            self.t_idx = jnp.asarray(t_idx)
        else:
            key = jax.random.PRNGKey(seed)
            kx, kt = jax.random.split(key, 2)
            self.x_idx = jax.random.randint(kx, (n_sensors,), 0, SOLVER_NX)
            min_t = max(
                1, int(jnp.searchsorted(self.t_grid, MIN_OBS_TIME, side="left"))
            )
            self.t_idx = jax.random.randint(kt, (n_sensors,), min_t, SOLVER_NT)
        self.n_sensors = int(self.x_idx.shape[0])

    @classmethod
    def from_indices(cls, x_idx, t_idx):
        """Rebuild the sensor layout from saved indices (e.g. a trained model)."""
        return cls(x_idx=x_idx, t_idx=t_idx)

    @property
    def layout_id(self) -> str:
        """Stable identifier for the fixed sensor coordinates and solver grid."""
        digest = sha256()
        digest.update(np.asarray(self.x_idx, dtype=np.int64).tobytes())
        digest.update(np.asarray(self.t_idx, dtype=np.int64).tobytes())
        digest.update(f"{SOLVER_NX}:{SOLVER_NT}:{MIN_OBS_TIME}".encode())
        return digest.hexdigest()


def _simulate_chunk(nus, amps, phases, x_grid, t_grid, x_idx, t_idx):
    """Vmapped solve + gather at sensor nodes for one chunk of parameters."""

    def one(nu, amp, phase):
        field = _solve_burgers(nu, x_grid, t_grid, amp, phase)
        return field[t_idx, x_idx]

    return jax.vmap(one)(nus, amps, phases)


_simulate_chunk_jit = jax.jit(_simulate_chunk)


def simulate(
    theta: torch.Tensor,
    sensors: Sensors,
    noise_std: float = DEFAULT_NOISE_STD,
    seed: int = 0,
    chunk: int = 256,
) -> torch.Tensor:
    """Map parameter rows to sparse noisy observation vectors (the SBI simulator).

    ``theta`` is ``(N, 3)`` torch; returns ``(N, n_sensors)`` torch. The JAX solver
    runs in chunked vmap for speed/memory; Gaussian sensor noise is then added.
    """
    theta_np = theta.detach().cpu().numpy().astype(np.float32)
    nus, amps, phases = theta_np[:, 0], theta_np[:, 1], theta_np[:, 2]

    outputs = []
    for start in range(0, theta_np.shape[0], chunk):
        sl = slice(start, start + chunk)
        out = _simulate_chunk_jit(
            jnp.asarray(nus[sl]),
            jnp.asarray(amps[sl]),
            jnp.asarray(phases[sl]),
            sensors.x_grid,
            sensors.t_grid,
            sensors.x_idx,
            sensors.t_idx,
        )
        outputs.append(np.asarray(out))
    clean = np.concatenate(outputs, axis=0)

    rng = np.random.default_rng(seed)
    noisy = clean + rng.normal(0.0, noise_std, size=clean.shape).astype(np.float32)
    return torch.from_numpy(noisy.astype(np.float32))


def train_fmpe(
    config: FMPEConfig | None = None,
    *,
    n_sims: int = 4000,
    n_sensors: int = 64,
    noise_std: float = DEFAULT_NOISE_STD,
    sim_seed: int | None = None,
    sensor_seed: int | None = None,
    simulation_seed: int | None = None,
    training_seed: int | None = None,
    prior: BoxUniform | None = None,
    device: str = "cpu",
    max_num_epochs: int | None = None,
    show_train_summary: bool = False,
):
    """Simulate and train the amortized FMPE posterior.

    Returns a dict with the trained ``posterior``, the ``sensors``, the ``prior``,
    and the training ``theta``/``x`` tensors (handy for SBC diagnostics).
    """
    if config is None:
        legacy_seed = 0 if sim_seed is None else sim_seed
        config = FMPEConfig(
            n_sims=n_sims,
            n_sensors=n_sensors,
            noise_std=noise_std,
            sensor_seed=legacy_seed if sensor_seed is None else sensor_seed,
            simulation_seed=legacy_seed if simulation_seed is None else simulation_seed,
            training_seed=1 if training_seed is None else training_seed,
            device=device,
            max_num_epochs=max_num_epochs,
            show_train_summary=show_train_summary,
        )
    elif any(
        value is not None
        for value in (sim_seed, sensor_seed, simulation_seed, training_seed)
    ):
        raise ValueError("Pass either config or seed keyword overrides, not both")

    prior = prior or default_prior(
        config.device, low=config.prior_low, high=config.prior_high
    )
    sensors = Sensors(n_sensors=config.n_sensors, seed=config.sensor_seed)

    seed_random_generators(config.simulation_seed)
    theta = prior.sample((config.n_sims,))
    x = simulate(
        theta,
        sensors,
        noise_std=config.noise_std,
        seed=config.simulation_seed,
    )

    seed_random_generators(config.training_seed)
    inference = FMPE(prior, device=config.device)
    train_kwargs = {"show_train_summary": config.show_train_summary}
    if config.max_num_epochs is not None:
        train_kwargs["max_num_epochs"] = config.max_num_epochs
    inference.append_simulations(theta, x).train(**train_kwargs)
    posterior = inference.build_posterior()

    return {
        "posterior": posterior,
        "sensors": sensors,
        "prior": prior,
        "theta_train": theta,
        "x_train": x,
        "noise_std": config.noise_std,
        "config": config,
        "config_dict": asdict(config),
        "inference": inference,
    }


def observation_from_theta(
    theta_true, sensors: Sensors, noise_std: float = DEFAULT_NOISE_STD, seed: int = 1234
) -> torch.Tensor:
    """Build a single noisy observation vector ``x_o`` from a known parameter set."""
    theta = torch.tensor([list(theta_true)], dtype=torch.float32)
    return simulate(theta, sensors, noise_std=noise_std, seed=seed)


def run_calibration(
    result, n_sbc: int = 200, n_posterior_samples: int = 200, seed: int = 99
):
    """Validate posterior calibration with simulation-based calibration + TARP.

    SBC checks that, over many simulated ground truths, the posterior ranks are
    uniform (per-parameter KS p-value > ~0.05 indicates calibration). TARP gives an
    expected-coverage curve; ``atc`` near 0 and KS p-value > ~0.05 indicate the
    credible regions have nominal coverage.
    """
    from sbi.diagnostics import check_sbc, check_tarp, run_sbc, run_tarp

    prior, sensors, posterior = result["prior"], result["sensors"], result["posterior"]
    seed_random_generators(seed)
    thetas = prior.sample((n_sbc,))
    xs = simulate(thetas, sensors, noise_std=result["noise_std"], seed=seed)

    ranks, dap = run_sbc(
        thetas,
        xs,
        posterior,
        num_posterior_samples=n_posterior_samples,
        show_progress_bar=False,
    )
    sbc_stats = check_sbc(ranks, thetas, dap, num_posterior_samples=n_posterior_samples)

    ecp, alpha = run_tarp(
        thetas,
        xs,
        posterior,
        num_posterior_samples=n_posterior_samples,
        show_progress_bar=False,
    )
    tarp_atc, tarp_ks_pval = check_tarp(ecp, alpha)

    return {
        "sbc": sbc_stats,
        "tarp_atc": float(tarp_atc),
        "tarp_ks_pval": float(tarp_ks_pval),
        "n_cases": n_sbc,
        "n_posterior_samples": n_posterior_samples,
        "seed": seed,
    }


def result_from_bundle(bundle):
    """Build the minimal diagnostics input expected by calibration/evaluation."""
    contract = bundle["metadata"]["contract"]
    return {
        "posterior": bundle["posterior"],
        "sensors": bundle["sensors"],
        "prior": default_prior(
            low=contract["prior_low"],
            high=contract["prior_high"],
        ),
        "noise_std": contract["noise_std"],
        "metadata": bundle["metadata"],
    }


def summarize_posterior_samples(
    samples,
    truths,
    prior_low=PRIOR_LOW,
    prior_high=PRIOR_HIGH,
    credible_mass=0.9,
):
    """Summarize held-out posterior accuracy, coverage, and contraction.

    ``samples`` has shape ``(n_cases, n_samples, n_params)`` and ``truths`` has
    shape ``(n_cases, n_params)``.
    """
    samples = np.asarray(samples, dtype=np.float64)
    truths = np.asarray(truths, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[2] != len(PARAM_NAMES):
        raise ValueError("samples must have shape (n_cases, n_samples, 3)")
    if truths.shape != (samples.shape[0], len(PARAM_NAMES)):
        raise ValueError("truths must have shape (n_cases, 3)")
    if not 0.0 < credible_mass < 1.0:
        raise ValueError("credible_mass must lie in (0, 1)")

    tail = (1.0 - credible_mass) / 2.0
    lower = np.quantile(samples, tail, axis=1)
    upper = np.quantile(samples, 1.0 - tail, axis=1)
    means = samples.mean(axis=1)
    stds = samples.std(axis=1)
    prior_stds = (
        np.asarray(prior_high, dtype=np.float64)
        - np.asarray(prior_low, dtype=np.float64)
    ) / np.sqrt(12.0)

    parameters = {}
    for index, name in enumerate(PARAM_NAMES):
        covered = (truths[:, index] >= lower[:, index]) & (
            truths[:, index] <= upper[:, index]
        )
        parameters[name] = {
            "rmse": float(np.sqrt(np.mean((means[:, index] - truths[:, index]) ** 2))),
            "mean_posterior_std": float(stds[:, index].mean()),
            "mean_interval_width": float((upper[:, index] - lower[:, index]).mean()),
            "coverage": float(covered.mean()),
            "contraction_ratio": float(stds[:, index].mean() / prior_stds[index]),
        }
    return {
        "n_cases": int(samples.shape[0]),
        "n_posterior_samples": int(samples.shape[1]),
        "credible_mass": float(credible_mass),
        "parameters": parameters,
    }


def evaluate_posterior(
    result,
    n_cases: int = 100,
    n_posterior_samples: int = 500,
    seed: int = 101,
):
    """Evaluate posterior contraction and coverage on held-out simulations."""
    if n_cases <= 0 or n_posterior_samples <= 0:
        raise ValueError("Evaluation sizes must be positive")
    prior, sensors, posterior = result["prior"], result["sensors"], result["posterior"]
    seed_random_generators(seed)
    truths = prior.sample((n_cases,))
    observations = simulate(
        truths,
        sensors,
        noise_std=result["noise_std"],
        seed=seed,
    )

    posterior_samples = []
    for index in range(n_cases):
        seed_random_generators(seed + index + 1)
        samples = posterior.sample(
            (n_posterior_samples,),
            x=observations[index : index + 1],
            show_progress_bars=False,
        )
        posterior_samples.append(samples.detach().cpu().numpy())

    return summarize_posterior_samples(
        np.stack(posterior_samples),
        truths.detach().cpu().numpy(),
        prior_low=prior.low.detach().cpu().numpy(),
        prior_high=prior.high.detach().cpu().numpy(),
    )


def run_contraction_study(
    base_config: FMPEConfig,
    sensor_counts,
    noise_levels,
    *,
    n_cases: int = 100,
    n_posterior_samples: int = 500,
    evaluation_seed: int = 101,
):
    """Retrain FMPE over a sensor/noise grid and return held-out metrics."""
    rows = []
    for n_sensors in sensor_counts:
        for noise_std in noise_levels:
            config = replace(
                base_config,
                n_sensors=int(n_sensors),
                noise_std=float(noise_std),
            )
            result = train_fmpe(config)
            evaluation = evaluate_posterior(
                result,
                n_cases=n_cases,
                n_posterior_samples=n_posterior_samples,
                seed=evaluation_seed,
            )
            row = {
                "n_sensors": config.n_sensors,
                "noise_std": config.noise_std,
                "n_sims": config.n_sims,
                "sensor_seed": config.sensor_seed,
                "simulation_seed": config.simulation_seed,
                "training_seed": config.training_seed,
            }
            for name, metrics in evaluation["parameters"].items():
                for metric_name, value in metrics.items():
                    row[f"{name}_{metric_name}"] = value
            rows.append(row)
    return rows


def _jsonable(value):
    """Convert diagnostic tensors/arrays recursively to JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.ndarray, torch.Tensor)):
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def write_json_report(report, path) -> pathlib.Path:
    """Write a deterministic JSON diagnostics report."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(report), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def write_contraction_csv(rows, path) -> pathlib.Path:
    """Write contraction-study rows to CSV."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Contraction report requires at least one row")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_model(result, path) -> pathlib.Path:
    """Persist the trained posterior + sensor layout for packaging as a Tesseract."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = result.get("config")
    config_dict = (
        asdict(config)
        if isinstance(config, FMPEConfig)
        else dict(result.get("config_dict", {}))
    )
    sensors = result["sensors"]
    contract = {
        "param_names": list(PARAM_NAMES),
        "observation_dim": sensors.n_sensors,
        "sensor_layout_id": sensors.layout_id,
        "solver_grid": {"nx": SOLVER_NX, "nt": SOLVER_NT},
        "min_observation_time": MIN_OBS_TIME,
        "noise_std": float(result["noise_std"]),
        "prior_low": list(config_dict.get("prior_low", PRIOR_LOW)),
        "prior_high": list(config_dict.get("prior_high", PRIOR_HIGH)),
    }
    identity = sha256()
    identity.update(
        json.dumps(
            {"contract": contract, "training_config": config_dict},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    identity.update(pickle.dumps(result["posterior"], protocol=pickle.HIGHEST_PROTOCOL))
    model_id = identity.hexdigest()
    metadata = {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "model_id": model_id,
        "contract": contract,
        "training_config": config_dict,
    }
    with open(path, "wb") as handle:
        pickle.dump(
            {
                "posterior": result["posterior"],
                "x_idx": np.asarray(sensors.x_idx),
                "t_idx": np.asarray(sensors.t_idx),
                "noise_std": result["noise_std"],
                "param_names": PARAM_NAMES,
                "metadata": metadata,
            },
            handle,
        )
    return path


def _legacy_metadata(bundle, sensors):
    """Build a validated compatibility manifest for pre-versioned local bundles."""
    contract = {
        "param_names": list(bundle.get("param_names", PARAM_NAMES)),
        "observation_dim": sensors.n_sensors,
        "sensor_layout_id": sensors.layout_id,
        "solver_grid": {"nx": SOLVER_NX, "nt": SOLVER_NT},
        "min_observation_time": MIN_OBS_TIME,
        "noise_std": float(bundle.get("noise_std", DEFAULT_NOISE_STD)),
        "prior_low": list(PRIOR_LOW),
        "prior_high": list(PRIOR_HIGH),
    }
    return {
        "format": BUNDLE_FORMAT,
        "version": 0,
        "model_id": f"legacy-{sensors.layout_id}",
        "contract": contract,
        "training_config": {},
        "legacy": True,
    }


def validate_model_bundle(bundle, *, allow_legacy=True):
    """Validate posterior bundle structure and return it with rebuilt sensors."""
    required = {"posterior", "x_idx", "t_idx"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Posterior bundle is missing keys: {sorted(missing)}")

    x_idx = np.asarray(bundle["x_idx"])
    t_idx = np.asarray(bundle["t_idx"])
    if x_idx.ndim != 1 or t_idx.ndim != 1 or x_idx.shape != t_idx.shape:
        raise ValueError("Posterior sensor indices must be equal-length 1D arrays")
    if x_idx.size == 0:
        raise ValueError("Posterior sensor layout must not be empty")
    if np.any((x_idx < 0) | (x_idx >= SOLVER_NX)):
        raise ValueError("Posterior x_idx contains values outside the solver grid")
    if np.any((t_idx < 0) | (t_idx >= SOLVER_NT)):
        raise ValueError("Posterior t_idx contains values outside the solver grid")

    sensors = Sensors.from_indices(x_idx, t_idx)
    metadata = bundle.get("metadata")
    if metadata is None:
        if not allow_legacy:
            raise ValueError("Posterior bundle has no versioned metadata")
        metadata = _legacy_metadata(bundle, sensors)
    if metadata.get("format") != BUNDLE_FORMAT:
        raise ValueError("Unsupported posterior bundle format")
    version = int(metadata.get("version", -1))
    if version not in (0, BUNDLE_VERSION):
        raise ValueError(f"Unsupported posterior bundle version: {version}")

    contract = metadata.get("contract", {})
    if int(contract.get("observation_dim", -1)) != sensors.n_sensors:
        raise ValueError("Bundle observation_dim does not match sensor indices")
    if contract.get("sensor_layout_id") != sensors.layout_id:
        raise ValueError("Bundle sensor_layout_id does not match sensor indices")
    if tuple(contract.get("param_names", ())) != PARAM_NAMES:
        raise ValueError("Bundle parameter order does not match the FMPE contract")

    validated = dict(bundle)
    validated["metadata"] = metadata
    validated["sensors"] = sensors
    return validated


def validate_observation(observation, bundle) -> np.ndarray:
    """Return a flat float32 observation after checking the trained contract."""
    observation = np.asarray(observation, dtype=np.float32).ravel()
    expected = int(bundle["metadata"]["contract"]["observation_dim"])
    if observation.size != expected:
        raise ValueError(
            f"Observation has length {observation.size}; model expects {expected}"
        )
    if not np.all(np.isfinite(observation)):
        raise ValueError("Observation must contain only finite values")
    return observation


def load_model(path, *, allow_legacy=True):
    """Load and validate a posterior bundle, rebuilding its sensor layout."""
    with open(path, "rb") as handle:
        bundle = pickle.load(handle)
    return validate_model_bundle(bundle, allow_legacy=allow_legacy)


def upgrade_model_bundle(path, output=None) -> pathlib.Path:
    """Rewrite a legacy posterior bundle using the current versioned contract."""
    path = pathlib.Path(path)
    bundle = load_model(path, allow_legacy=True)
    output = path if output is None else pathlib.Path(output)
    result = {
        "posterior": bundle["posterior"],
        "sensors": bundle["sensors"],
        "noise_std": bundle["metadata"]["contract"]["noise_std"],
        "config_dict": bundle["metadata"].get("training_config", {}),
    }
    return save_model(result, output)


def query_posterior_tesseract(
    observation, seed: int = 0, image: str = "fmpe_posterior"
):
    """Query the packaged posterior Tesseract for one observation (swappable component)."""
    from tesseract_core import Tesseract

    with Tesseract.from_image(image) as tesseract:
        return tesseract.apply(
            {
                "observation": np.asarray(observation, dtype=np.float32).ravel(),
                "seed": int(seed),
            }
        )


def _report_samples(samples, theta_true=None):
    samples = np.asarray(samples)
    for i, name in enumerate(PARAM_NAMES):
        col = samples[:, i]
        lo, hi = np.percentile(col, [5, 95])
        truth = "" if theta_true is None else f"  (true {theta_true[i]:.4f})"
        print(
            f"  {name:8s}: mean={col.mean():.4f} std={col.std():.4f} "
            f"90% CI=[{lo:.4f}, {hi:.4f}]{truth}"
        )


def _main():
    import argparse

    parser = argparse.ArgumentParser(description="Amortized FMPE posterior (Stage B)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser(
        "train", help="Train and persist the posterior for packaging"
    )
    train.add_argument("--n-sims", type=int, default=10000)
    train.add_argument("--n-sensors", type=int, default=64)
    train.add_argument("--noise-std", type=float, default=DEFAULT_NOISE_STD)
    train.add_argument("--sensor-seed", type=int, default=0)
    train.add_argument("--simulation-seed", type=int, default=0)
    train.add_argument("--training-seed", type=int, default=1)
    train.add_argument("--max-epochs", type=int)
    train.add_argument("--out", default="tesseracts/fmpe_posterior/posterior.pkl")
    train.add_argument("--calibrate", action="store_true", help="Run SBC + TARP")

    demo = sub.add_parser(
        "demo", help="Query the posterior for a known nu (in-process)"
    )
    demo.add_argument("--model", default="tesseracts/fmpe_posterior/posterior.pkl")
    demo.add_argument("--nu", type=float, default=0.05)
    demo.add_argument("--ic-amp", type=float, default=1.0)
    demo.add_argument("--ic-phase", type=float, default=0.0)
    demo.add_argument(
        "--tesseract", action="store_true", help="Query via the built Tesseract image"
    )

    migrate = sub.add_parser(
        "migrate", help="Upgrade a legacy posterior.pkl to the versioned bundle format"
    )
    migrate.add_argument("model")
    migrate.add_argument("--out")

    args = parser.parse_args()

    if args.cmd == "train":
        config = FMPEConfig(
            n_sims=args.n_sims,
            n_sensors=args.n_sensors,
            noise_std=args.noise_std,
            sensor_seed=args.sensor_seed,
            simulation_seed=args.simulation_seed,
            training_seed=args.training_seed,
            max_num_epochs=args.max_epochs,
        )
        result = train_fmpe(config)
        path = save_model(result, args.out)
        theta_true = (0.05, 1.0, 0.0)
        x_o = observation_from_theta(theta_true, result["sensors"], result["noise_std"])
        samples = result["posterior"].sample((3000,), x=x_o).cpu().numpy()
        print("\n================ FMPE POSTERIOR ================")
        _report_samples(samples, theta_true)
        print(f"saved model -> {path}")
        if args.calibrate:
            calib = run_calibration(result)
            print(
                "\nSBC KS p-values:",
                [f"{p:.3f}" for p in calib["sbc"]["ks_pvals"].tolist()],
            )
            print(f"TARP atc={calib['tarp_atc']:.4f} KS p={calib['tarp_ks_pval']:.3f}")

    elif args.cmd == "demo":
        bundle = load_model(args.model)
        theta_true = (args.nu, args.ic_amp, args.ic_phase)
        x_o = observation_from_theta(theta_true, bundle["sensors"], bundle["noise_std"])
        if args.tesseract:
            out = query_posterior_tesseract(x_o.numpy())
            print("\n================ FMPE POSTERIOR (Tesseract) ================")
            _report_samples(np.asarray(out["samples"]), theta_true)
        else:
            seed_random_generators(0)
            samples = (
                bundle["posterior"]
                .sample((3000,), x=x_o.reshape(1, -1), show_progress_bars=False)
                .cpu()
                .numpy()
            )
            print("\n================ FMPE POSTERIOR (in-process) ================")
            _report_samples(samples, theta_true)

    elif args.cmd == "migrate":
        print(f"upgraded model -> {upgrade_model_bundle(args.model, args.out)}")


if __name__ == "__main__":
    _main()
