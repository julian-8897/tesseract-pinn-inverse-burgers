"""Reproducible FMPE calibration and contraction diagnostics."""

from __future__ import annotations

import argparse
import pathlib

from burgers_inverse.configs import FMPEConfig
from burgers_inverse.fmpe_posterior import (
    evaluate_posterior,
    load_model,
    result_from_bundle,
    run_calibration,
    run_contraction_study,
    write_contraction_csv,
    write_json_report,
)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate = subparsers.add_parser(
        "calibrate", help="Evaluate one persisted posterior on held-out simulations"
    )
    calibrate.add_argument("--model", default="tesseracts/fmpe_posterior/posterior.pkl")
    calibrate.add_argument("--n-cases", type=int, default=200)
    calibrate.add_argument("--posterior-samples", type=int, default=200)
    calibrate.add_argument("--seed", type=int, default=99)
    calibrate.add_argument(
        "--out", default="artifacts/fmpe_calibration.json", type=pathlib.Path
    )

    contraction = subparsers.add_parser(
        "contraction", help="Retrain over sensor/noise settings and record contraction"
    )
    contraction.add_argument("--n-sims", type=int, default=10000)
    contraction.add_argument(
        "--sensor-counts", type=int, nargs="+", default=[16, 32, 64]
    )
    contraction.add_argument(
        "--noise-levels", type=float, nargs="+", default=[0.01, 0.02, 0.05]
    )
    contraction.add_argument("--n-cases", type=int, default=100)
    contraction.add_argument("--posterior-samples", type=int, default=500)
    contraction.add_argument("--sensor-seed", type=int, default=0)
    contraction.add_argument("--simulation-seed", type=int, default=0)
    contraction.add_argument("--training-seed", type=int, default=1)
    contraction.add_argument("--evaluation-seed", type=int, default=101)
    contraction.add_argument("--max-epochs", type=int)
    contraction.add_argument(
        "--out", default="artifacts/fmpe_contraction.csv", type=pathlib.Path
    )
    return parser


def main():
    args = _parser().parse_args()
    if args.command == "calibrate":
        bundle = load_model(args.model)
        result = result_from_bundle(bundle)
        calibration = run_calibration(
            result,
            n_sbc=args.n_cases,
            n_posterior_samples=args.posterior_samples,
            seed=args.seed,
        )
        contraction = evaluate_posterior(
            result,
            n_cases=args.n_cases,
            n_posterior_samples=args.posterior_samples,
            seed=args.seed + 1,
        )
        report = {
            "model": bundle["metadata"],
            "calibration": calibration,
            "held_out": contraction,
        }
        print(f"wrote {write_json_report(report, args.out)}")
        return

    config = FMPEConfig(
        n_sims=args.n_sims,
        n_sensors=args.sensor_counts[0],
        noise_std=args.noise_levels[0],
        sensor_seed=args.sensor_seed,
        simulation_seed=args.simulation_seed,
        training_seed=args.training_seed,
        max_num_epochs=args.max_epochs,
    )
    rows = run_contraction_study(
        config,
        args.sensor_counts,
        args.noise_levels,
        n_cases=args.n_cases,
        n_posterior_samples=args.posterior_samples,
        evaluation_seed=args.evaluation_seed,
    )
    print(f"wrote {write_contraction_csv(rows, args.out)}")


if __name__ == "__main__":
    main()
