"""Generate publication-quality figures for the FMPE SBI workflow."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from fmpe_posterior import (
    PARAM_NAMES,
    load_model,
    observation_from_theta,
    query_posterior_tesseract,
    seed_random_generators,
    validate_observation,
)
from scripts.ml_plot_style import (
    apply_ml_style,
    paper_size,
    save_figure,
    validate_publication_figure,
)

LABELS = {
    "nu": r"$\nu$",
    "ic_amp": r"$A_{\rm IC}$",
    "ic_phase": r"$\phi_{\rm IC}$",
}
UNITS = {
    "nu": "",
    "ic_amp": "",
    "ic_phase": r" [rad]",
}
COLORS = ("#0072B2", "#D55E00", "#009E73")


def posterior_samples(bundle, theta_true, seed, use_tesseract):
    """Generate one synthetic observation and posterior samples."""
    observation = observation_from_theta(
        theta_true,
        bundle["sensors"],
        noise_std=bundle["metadata"]["contract"]["noise_std"],
        seed=seed,
    )
    observation_np = validate_observation(observation.numpy(), bundle)
    if use_tesseract:
        output = query_posterior_tesseract(observation_np, seed=seed)
        return observation_np, np.asarray(output["samples"]), dict(output)

    seed_random_generators(seed)
    samples = (
        bundle["posterior"]
        .sample(
            (2000,),
            x=observation.reshape(1, -1),
            show_progress_bars=False,
        )
        .detach()
        .cpu()
        .numpy()
    )
    return observation_np, samples, {"model_id": bundle["metadata"]["model_id"]}


def save_sample_artifact(path, bundle, theta_true, observation, samples, source):
    """Persist the exact posterior draw used by the figures."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        theta_true=np.asarray(theta_true, dtype=np.float32),
        observation=np.asarray(observation, dtype=np.float32),
        samples=np.asarray(samples, dtype=np.float32),
        x_idx=np.asarray(bundle["sensors"].x_idx),
        t_idx=np.asarray(bundle["sensors"].t_idx),
        model_id=np.asarray(bundle["metadata"]["model_id"]),
        source=np.asarray(source),
    )


def load_sample_artifact(path):
    """Load a previously persisted posterior draw."""
    with np.load(path) as artifact:
        return (
            tuple(artifact["theta_true"].tolist()),
            artifact["observation"],
            artifact["samples"],
            str(artifact["source"]),
        )


def plot_posterior(samples, theta_true, output_stem):
    """Plot marginals and lower-triangle joint posterior projections."""
    samples = np.asarray(samples)
    fig, axes = plt.subplots(
        3,
        3,
        figsize=paper_size("double", ratio=0.92),
        constrained_layout=True,
    )

    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            if row < col:
                ax.set_visible(False)
                continue
            if row == col:
                values = samples[:, row]
                ax.hist(
                    values,
                    bins=32,
                    density=True,
                    histtype="stepfilled",
                    color=COLORS[row],
                    alpha=0.30,
                    edgecolor=COLORS[row],
                )
                ax.axvline(theta_true[row], color="black", linestyle="--")
                q05, q50, q95 = np.quantile(values, [0.05, 0.5, 0.95])
                ax.axvspan(q05, q95, color=COLORS[row], alpha=0.12)
                ax.axvline(q50, color=COLORS[row], linewidth=1.5)
                ax.set_ylabel("Density")
            else:
                ax.hexbin(
                    samples[:, col],
                    samples[:, row],
                    gridsize=28,
                    mincnt=1,
                    cmap="Blues",
                    linewidths=0,
                )
                ax.axvline(theta_true[col], color="black", linestyle="--")
                ax.axhline(theta_true[row], color="black", linestyle="--")

            if row == 2:
                ax.set_xlabel(f"{LABELS[PARAM_NAMES[col]]}{UNITS[PARAM_NAMES[col]]}")
            else:
                ax.tick_params(labelbottom=False)
            if col == 0 and row > 0:
                ax.set_ylabel(f"{LABELS[PARAM_NAMES[row]]}{UNITS[PARAM_NAMES[row]]}")
            elif row > col:
                ax.tick_params(labelleft=False)

    legend = [
        Line2D([0], [0], color="black", linestyle="--", label="Ground truth"),
        Line2D([0], [0], color=COLORS[0], label="Posterior median"),
    ]
    axes[0, 0].legend(handles=legend, loc="upper right")
    issues = validate_publication_figure(fig, allow_titles=False)
    if issues:
        raise RuntimeError("; ".join(issues))
    save_figure(fig, output_stem)
    plt.close(fig)


def plot_observation_layout(bundle, observation, output_stem):
    """Plot the fixed sensor layout and corresponding observed values."""
    sensors = bundle["sensors"]
    x = np.asarray(sensors.x_grid)[np.asarray(sensors.x_idx)]
    t = np.asarray(sensors.t_grid)[np.asarray(sensors.t_idx)]

    fig, (ax_layout, ax_values) = plt.subplots(
        1,
        2,
        figsize=paper_size("double", ratio=0.50),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.0, 1.15)},
    )
    scatter = ax_layout.scatter(
        x,
        t,
        c=observation,
        cmap="RdBu_r",
        s=20,
        edgecolor="black",
        linewidth=0.25,
    )
    ax_layout.set_xlabel(r"Sensor position $x_i$")
    ax_layout.set_ylabel(r"Sensor time $t_i$")
    ax_layout.set_aspect("equal", adjustable="box")
    colorbar = fig.colorbar(scatter, ax=ax_layout, pad=0.02)
    colorbar.set_label(r"$u_{\rm obs}$")

    order = np.lexsort((x, t))
    value_scatter = ax_values.scatter(
        t[order],
        observation[order],
        c=x[order],
        cmap="viridis",
        s=17,
        edgecolor="black",
        linewidth=0.2,
    )
    ax_values.axhline(0.0, color="0.5", linewidth=0.7)
    ax_values.set_xlabel(r"Sensor time $t_i$")
    ax_values.set_ylabel(r"Observation $u_{\rm obs}(x_i,t_i)$")
    value_colorbar = fig.colorbar(value_scatter, ax=ax_values, pad=0.02)
    value_colorbar.set_label(r"Sensor position $x_i$")

    for label, ax in zip(("(a)", "(b)"), (ax_layout, ax_values), strict=True):
        ax.text(
            0.01,
            1.02,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontweight="bold",
        )

    issues = validate_publication_figure(fig, allow_titles=False)
    if issues:
        raise RuntimeError("; ".join(issues))
    save_figure(fig, output_stem)
    plt.close(fig)


def plot_calibration(report_path, output_stem):
    """Plot SBC summary, held-out coverage, and contraction metrics."""
    with pathlib.Path(report_path).open(encoding="utf-8") as handle:
        report = json.load(handle)
    calibration = report["calibration"]
    held_out = report["held_out"]["parameters"]

    x = np.arange(len(PARAM_NAMES))
    labels = [LABELS[name] for name in PARAM_NAMES]
    c2st = np.asarray(calibration["sbc"]["c2st_ranks"])
    coverage = np.asarray([held_out[name]["coverage"] for name in PARAM_NAMES])
    contraction = np.asarray(
        [held_out[name]["contraction_ratio"] for name in PARAM_NAMES]
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=paper_size("double", ratio=0.38),
        constrained_layout=True,
    )
    axes[0].bar(x, c2st, color=COLORS, alpha=0.85)
    axes[0].axhline(0.5, color="black", linestyle="--", label="Ideal")
    axes[0].set_ylabel("SBC rank C2ST")
    axes[0].set_xticks(x, labels)
    axes[0].set_xlabel("Parameter")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(loc="upper right")

    axes[1].bar(x, coverage, color=COLORS, alpha=0.85)
    axes[1].axhline(
        report["held_out"]["credible_mass"],
        color="black",
        linestyle="--",
        label="Nominal",
    )
    axes[1].set_ylabel("Empirical coverage")
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("Parameter")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend(loc="lower left")

    axes[2].bar(x, contraction, color=COLORS, alpha=0.85)
    axes[2].axhline(1.0, color="black", linestyle="--", label="Prior width")
    axes[2].set_ylabel(r"Posterior $\sigma$ / prior $\sigma$")
    axes[2].set_xticks(x, labels)
    axes[2].set_xlabel("Parameter")
    axes[2].set_ylim(0.0, max(1.05, contraction.max() * 1.15))
    axes[2].legend(loc="center right")

    issues = validate_publication_figure(fig, allow_titles=False)
    if issues:
        raise RuntimeError("; ".join(issues))
    save_figure(fig, output_stem)
    plt.close(fig)


def plot_contraction(csv_path, output_stem):
    """Plot viscosity contraction against sensors for each noise level."""
    with pathlib.Path(csv_path).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Contraction CSV contains no rows")

    noise_levels = sorted({float(row["noise_std"]) for row in rows})
    fig, (ax_std, ax_coverage) = plt.subplots(
        1,
        2,
        figsize=paper_size("double", ratio=0.42),
        constrained_layout=True,
    )
    markers = ("o", "s", "^", "D")
    for index, noise_std in enumerate(noise_levels):
        subset = sorted(
            (row for row in rows if float(row["noise_std"]) == noise_std),
            key=lambda row: int(row["n_sensors"]),
        )
        sensors = [int(row["n_sensors"]) for row in subset]
        contraction = [float(row["nu_contraction_ratio"]) for row in subset]
        coverage = [float(row["nu_coverage"]) for row in subset]
        label = rf"$\sigma_{{\rm obs}}={noise_std:g}$"
        ax_std.plot(
            sensors,
            contraction,
            marker=markers[index % len(markers)],
            label=label,
        )
        ax_coverage.plot(
            sensors,
            coverage,
            marker=markers[index % len(markers)],
            label=label,
        )

    ax_std.axhline(1.0, color="black", linestyle="--", label="Prior width")
    ax_std.set_xlabel("Number of sensors")
    ax_std.set_ylabel(r"$\nu$ posterior $\sigma$ / prior $\sigma$")
    ax_std.set_xscale("log", base=2)
    ax_std.legend(loc="upper right")

    ax_coverage.axhline(0.9, color="black", linestyle="--", label="Nominal")
    ax_coverage.set_xlabel("Number of sensors")
    ax_coverage.set_ylabel(r"$\nu$ 90\% interval coverage")
    ax_coverage.set_xscale("log", base=2)
    ax_coverage.set_ylim(0.0, 1.05)
    ax_coverage.legend(loc="lower right")

    issues = validate_publication_figure(fig, allow_titles=False)
    if issues:
        raise RuntimeError("; ".join(issues))
    save_figure(fig, output_stem)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="tesseracts/fmpe_posterior/posterior.pkl", type=pathlib.Path
    )
    parser.add_argument("--out-dir", default="img/sbi", type=pathlib.Path)
    parser.add_argument("--nu", type=float, default=0.05)
    parser.add_argument("--ic-amp", type=float, default=1.0)
    parser.add_argument("--ic-phase", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--in-process", action="store_true")
    parser.add_argument(
        "--samples-npz",
        type=pathlib.Path,
        help="Reuse a saved posterior draw instead of querying the model",
    )
    parser.add_argument("--calibration-report", type=pathlib.Path)
    parser.add_argument("--contraction-csv", type=pathlib.Path)
    args = parser.parse_args()

    apply_ml_style(palette="bright", require_scienceplots=True)
    bundle = load_model(args.model, allow_legacy=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.samples_npz:
        theta_true, observation, samples, source = load_sample_artifact(
            args.samples_npz
        )
        output = {"model_id": bundle["metadata"]["model_id"]}
    else:
        theta_true = (args.nu, args.ic_amp, args.ic_phase)
        observation, samples, output = posterior_samples(
            bundle,
            theta_true,
            args.seed,
            use_tesseract=not args.in_process,
        )
        source = "in-process" if args.in_process else "Tesseract"
        save_sample_artifact(
            args.out_dir / "fmpe_posterior_samples.npz",
            bundle,
            theta_true,
            observation,
            samples,
            source,
        )

    plot_posterior(samples, theta_true, args.out_dir / "fmpe_posterior")
    plot_observation_layout(
        bundle,
        observation,
        args.out_dir / "fmpe_sensor_observations",
    )
    if args.calibration_report:
        plot_calibration(
            args.calibration_report,
            args.out_dir / "fmpe_calibration",
        )
    if args.contraction_csv:
        plot_contraction(
            args.contraction_csv,
            args.out_dir / "fmpe_contraction",
        )
    print(f"posterior source: {source}")
    print(f"model id: {output.get('model_id', bundle['metadata']['model_id'])}")
    print(f"wrote figures under {args.out_dir}")


if __name__ == "__main__":
    main()
