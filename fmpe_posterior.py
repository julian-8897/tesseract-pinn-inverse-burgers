"""Amortized flow-matching posterior over Burgers parameters (Stage B).

Simulation-based inference: draw ``theta = (nu, ic_amp, ic_phase)`` from a prior,
simulate sparse noisy observations with the differentiable JAX Burgers solver, and
train an amortized **Flow Matching Posterior Estimation** (FMPE) network with
``sbi`` (flow backend: ``zuko``). The trained posterior maps a fresh observation
vector to samples of ``theta`` -- in particular a *calibrated* marginal posterior
over the viscosity ``nu``, with the initial-condition parameters marginalized.

Training simulations use the in-process JAX solver (fast, vmapped) -- the same
physics the ``burgers_solver`` Tesseract serves. The solver Tesseract remains the
packaged component used at inference / for the optional Stage-C refinement.
"""

from __future__ import annotations

import pathlib
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import torch
from sbi.inference import FMPE
from sbi.utils import BoxUniform

from configs import DEFAULT_NOISE_STD
from inverse_problem import (
    MIN_OBS_TIME,
    SOLVER_NT,
    SOLVER_NX,
    get_burgers_solver,
)

# theta = (nu, ic_amp, ic_phase)
PARAM_NAMES = ("nu", "ic_amp", "ic_phase")
PRIOR_LOW = (0.02, 0.8, -0.4)
PRIOR_HIGH = (0.10, 1.2, 0.4)

_solve_burgers = get_burgers_solver()


def default_prior(device: str = "cpu") -> BoxUniform:
    """Box-uniform prior over (nu, ic_amp, ic_phase)."""
    return BoxUniform(
        low=torch.tensor(PRIOR_LOW, dtype=torch.float32, device=device),
        high=torch.tensor(PRIOR_HIGH, dtype=torch.float32, device=device),
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
    n_sims: int = 4000,
    n_sensors: int = 64,
    noise_std: float = DEFAULT_NOISE_STD,
    sim_seed: int = 0,
    prior: BoxUniform | None = None,
    device: str = "cpu",
    max_num_epochs: int | None = None,
    show_train_summary: bool = False,
):
    """Simulate and train the amortized FMPE posterior.

    Returns a dict with the trained ``posterior``, the ``sensors``, the ``prior``,
    and the training ``theta``/``x`` tensors (handy for SBC diagnostics).
    """
    prior = prior or default_prior(device)
    sensors = Sensors(n_sensors=n_sensors, seed=sim_seed)

    theta = prior.sample((n_sims,))
    x = simulate(theta, sensors, noise_std=noise_std, seed=sim_seed)

    inference = FMPE(prior, device=device)
    train_kwargs = {"show_train_summary": show_train_summary}
    if max_num_epochs is not None:
        train_kwargs["max_num_epochs"] = max_num_epochs
    inference.append_simulations(theta, x).train(**train_kwargs)
    posterior = inference.build_posterior()

    return {
        "posterior": posterior,
        "sensors": sensors,
        "prior": prior,
        "theta_train": theta,
        "x_train": x,
        "noise_std": noise_std,
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
    }


def save_model(result, path) -> pathlib.Path:
    """Persist the trained posterior + sensor layout for packaging as a Tesseract."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(
            {
                "posterior": result["posterior"],
                "x_idx": np.asarray(result["sensors"].x_idx),
                "t_idx": np.asarray(result["sensors"].t_idx),
                "noise_std": result["noise_std"],
                "param_names": PARAM_NAMES,
            },
            handle,
        )
    return path


def load_model(path):
    """Load a saved posterior bundle; returns the dict and a rebuilt ``Sensors``."""
    with open(path, "rb") as handle:
        bundle = pickle.load(handle)
    bundle["sensors"] = Sensors.from_indices(bundle["x_idx"], bundle["t_idx"])
    return bundle


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

    args = parser.parse_args()

    if args.cmd == "train":
        torch.manual_seed(0)
        result = train_fmpe(n_sims=args.n_sims, n_sensors=args.n_sensors)
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
            torch.manual_seed(0)
            samples = (
                bundle["posterior"]
                .sample((3000,), x=x_o.reshape(1, -1), show_progress_bars=False)
                .cpu()
                .numpy()
            )
            print("\n================ FMPE POSTERIOR (in-process) ================")
            _report_samples(samples, theta_true)


if __name__ == "__main__":
    _main()
