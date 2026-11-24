"""Render the README's FMPE Burgers example as a looping portfolio plate."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from matplotlib.ticker import MaxNLocator

from burgers_inverse.components import get_burgers_solver
from burgers_inverse.fmpe_posterior import load_model, observation_from_theta

ROOT = Path(__file__).resolve().parents[2]
WIDTH, HEIGHT, DPI = 1440, 850, 100
FPS, DURATION = 24, 10
DRAW_COUNT, DRAW_SEED = 256, 20261124
OBSERVATION_SEED = 1234
BLUE = "#0a5c99"
BAND_ALPHA = 0.48
INK = "#20272d"
GREY = "#626d76"
RULE = "#b2bac2"


@dataclass(frozen=True)
class Trajectory:
    x: np.ndarray
    t: np.ndarray
    truth: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    sensor_x: np.ndarray
    sensor_t: np.ndarray
    observations: np.ndarray
    nu: np.ndarray
    nu_true: float
    steepest_time: float


def calculate(main: Path) -> Trajectory:
    sample_path = main / "img/sbi/fmpe_posterior_samples.npz"
    bundle = load_model(
        main / "tesseracts/fmpe_posterior/posterior.pkl", allow_legacy=False
    )
    sensors = bundle["sensors"]
    with np.load(sample_path) as saved:
        theta = saved["theta_true"]
        samples = saved["samples"]
        observation = saved["observation"]
        x_idx, t_idx = saved["x_idx"], saved["t_idx"]
        if str(saved["model_id"]) != bundle["metadata"]["model_id"]:
            raise ValueError(
                "The saved posterior draw and checkpoint have different model IDs"
            )
        np.testing.assert_array_equal(x_idx, sensors.x_idx)
        np.testing.assert_array_equal(t_idx, sensors.t_idx)

    reference = observation_from_theta(
        theta, sensors, bundle["metadata"]["contract"]["noise_std"], OBSERVATION_SEED
    )
    np.testing.assert_allclose(observation, reference.numpy()[0], atol=2e-6, rtol=0)
    if samples.shape != (2000, 3) or not np.isfinite(samples).all():
        raise ValueError("Expected the README's 2,000 finite joint posterior samples")

    indices = np.random.default_rng(DRAW_SEED).choice(
        len(samples), DRAW_COUNT, replace=False
    )
    solver = get_burgers_solver()
    x_grid, t_grid = sensors.x_grid, sensors.t_grid

    @jax.jit
    def solve(parameters):
        return jax.vmap(lambda nu, amp, phase: solver(nu, x_grid, t_grid, amp, phase))(
            *parameters.T
        )

    print(f"Forward-solving truth and {DRAW_COUNT} joint posterior draws", flush=True)
    truth = np.asarray(solve(jnp.asarray(theta[None, :])))[0]
    draws = np.concatenate(
        [
            np.asarray(solve(jnp.asarray(chunk)))
            for chunk in np.split(samples[indices], 4)
        ]
    )
    if not np.isfinite(truth).all() or not np.isfinite(draws).all():
        raise ValueError("Non-finite velocity field")

    # Close the periodic seam for display. The solver uses the original grid.
    lower, upper = np.quantile(draws, [0.05, 0.95], axis=0)
    x, t = np.asarray(x_grid), np.asarray(t_grid)
    nu_interval = np.quantile(samples[:, 0], [0.05, 0.50, 0.95])
    print(
        f"{len(observation)} observations, max seed-replay error "
        f"{np.max(np.abs(observation - reference.numpy()[0])):.3g}"
    )
    print(f"True θ {theta}, sample ν quantiles (5/50/95%) {nu_interval}")
    front_slope = np.gradient(truth, x, axis=1).min(axis=1)
    steepest = np.argmin(front_slope)
    print(
        f"Minimum ∂u/∂x {front_slope[0]:.3f} initially, "
        f"{front_slope[steepest]:.3f} at t={t[steepest]:.3f}"
    )
    return Trajectory(
        np.r_[x, 1.0],
        t,
        np.pad(truth, ((0, 0), (0, 1)), mode="wrap"),
        np.pad(lower, ((0, 0), (0, 1)), mode="wrap"),
        np.pad(upper, ((0, 0), (0, 1)), mode="wrap"),
        x[x_idx],
        t[t_idx],
        observation,
        samples[:, 0],
        float(theta[0]),
        float(t[steepest]),
    )


class Plate:
    def __init__(self, data: Trajectory):
        self.data = data
        self.fig = Figure(
            figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI, facecolor="white"
        )
        self.canvas = FigureCanvasAgg(self.fig)
        profile = self.fig.add_axes((0.082, 0.529, 0.555, 0.374))
        field = self.fig.add_axes((0.082, 0.128, 0.555, 0.303), sharex=profile)
        marginal = self.fig.add_axes((0.734, 0.54, 0.23, 0.318))
        scale = self.fig.add_axes((0.76, 0.247, 0.178, 0.029))

        self.profile = profile
        bound = 1.07 * np.max(np.abs([data.truth, data.lower, data.upper]))
        profile.set(
            xlim=(data.x[0], data.x[-1]), ylim=(-bound, bound), ylabel="Velocity u"
        )
        amplitude = np.max(np.abs(data.truth[0]))
        profile.set_yticks(
            [-amplitude, 0, amplitude], [f"{v:g}" for v in (-amplitude, 0, amplitude)]
        )
        profile.tick_params(axis="x", bottom=False, labelbottom=False)
        profile.axhline(0, color=RULE, lw=0.8, alpha=0.6)
        (self.truth,) = profile.plot([], [], color=INK, lw=2.2, zorder=4, label="Truth")
        self.band = Polygon(
            np.zeros((2 * len(data.x), 2)),
            fc=BLUE,
            ec="none",
            alpha=BAND_ALPHA,
            label="Predictive band",
        )
        profile.add_patch(self.band)
        self.profile_legend([self.truth, self.band])

        field.set(ylim=(data.t[-1], data.t[0]), xlabel="Position x", ylabel="Time t")
        field.set_xticks(np.linspace(data.x[0], data.x[-1], 3))
        field.set_yticks(np.linspace(data.t[0], data.t[-1], 3))
        limit = np.max(np.abs(np.r_[data.truth.ravel(), data.observations]))
        norm = mpl.colors.CenteredNorm(vcenter=0, halfrange=limit)
        dx, dt = data.x[1] - data.x[0], data.t[1] - data.t[0]
        self.image = field.imshow(
            data.truth,
            cmap="RdBu_r",
            norm=norm,
            origin="lower",
            aspect="auto",
            interpolation="bilinear",
            extent=(
                data.x[0] - dx / 2,
                data.x[-1] + dx / 2,
                data.t[0] - dt / 2,
                data.t[-1] + dt / 2,
            ),
        )
        self.reveal = Rectangle(
            (data.x[0], data.t[0]), np.ptp(data.x), 0, transform=field.transData
        )
        self.observed = field.scatter(
            [],
            [],
            s=67,
            linewidth=1.25,
            edgecolors=INK,
            zorder=4,
            clip_on=False,
        )
        self.sensor_colours = self.image.cmap(norm(data.observations))
        (self.cursor,) = field.plot([], [], color=INK, lw=1.25, zorder=3)
        field.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    marker="o",
                    color=INK,
                    markerfacecolor="white",
                    lw=0,
                    markersize=7.6,
                    label="Sensor readings",
                )
            ],
            loc="lower left",
            bbox_to_anchor=(0, 1.03),
            borderaxespad=0,
            borderpad=0,
            handlelength=0.8,
            handletextpad=0.44,
        )
        bar = self.fig.colorbar(
            self.image, cax=scale, orientation="horizontal", ticks=[-limit, 0, limit]
        )
        bar.set_ticklabels([f"{v:.2g}" for v in (-limit, 0, limit)])
        bar.outline.set_visible(False)
        scale.set_title("Velocity scale", fontsize=24, loc="left", pad=9)

        marginal.hist(
            data.nu,
            bins=32,
            density=True,
            histtype="stepfilled",
            facecolor=mpl.colors.to_rgba(BLUE, 0.24),
            edgecolor=BLUE,
            lw=2.0,
        )
        marginal.axvline(data.nu_true, color=INK, ls=(0, (4, 3)), lw=2.4)
        span = np.ptp(data.nu)
        marginal.set(
            xlim=(data.nu.min() - 0.055 * span, data.nu.max() + 0.055 * span),
            xlabel="Viscosity ν",
            ylim=(0, marginal.get_ylim()[1] * 1.07),
        )
        marginal.xaxis.set_major_locator(
            MaxNLocator(nbins=4, prune="both", steps=[1, 2, 5, 10])
        )
        marginal.set_yticks([])
        marginal.spines["left"].set_visible(False)
        marginal.text(
            0,
            1.17,
            "FMPE marginal",
            transform=marginal.transAxes,
            ha="left",
            va="bottom",
            fontsize=26,
        )
        marginal.legend(
            handles=[Line2D([], [], color=INK, ls=(0, (4, 3)), lw=2.4, label="True ν")],
            loc="lower left",
            bbox_to_anchor=(0, 1.01),
            borderaxespad=0,
            borderpad=0,
            handlelength=1.03,
            handletextpad=0.47,
        )

    def profile_legend(self, handles):
        self.profile.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0, 1.015),
            borderaxespad=0,
            borderpad=0,
            ncols=len(handles),
            columnspacing=1.05,
            handlelength=1.12,
            handletextpad=0.47,
        )

    def print_view(self):
        self.update(self.data.steepest_time, full_field=True)
        (initial,) = self.profile.plot(
            self.data.x,
            self.data.truth[0],
            ls=(0, (2, 2.5)),
            lw=1.55,
            color=GREY,
            alpha=0.72,
            zorder=1,
            label="Initial profile",
        )
        self.profile_legend([self.truth, self.band, initial])

    def update(self, time: float, opacity: float = 1, *, full_field: bool = False):
        data = self.data
        i = np.clip(np.searchsorted(data.t, time, side="right") - 1, 0, len(data.t) - 2)
        fraction = (time - data.t[i]) / (data.t[i + 1] - data.t[i])

        def slice_at(field):
            return (1 - fraction) * field[i] + fraction * field[i + 1]

        self.truth.set_data(data.x, slice_at(data.truth))
        self.truth.set_alpha(opacity)
        self.band.set_xy(
            np.column_stack(
                (
                    np.r_[data.x, data.x[::-1]],
                    np.r_[slice_at(data.lower), slice_at(data.upper)[::-1]],
                )
            )
        )
        self.band.set_alpha(BAND_ALPHA * opacity)

        end = data.t[-1] if full_field else time
        self.reveal.set_height(end - data.t[0])
        self.image.set_clip_path(self.reveal)
        self.image.set_alpha(opacity)
        visible = data.sensor_t <= end
        self.observed.set_offsets(
            np.column_stack((data.sensor_x[visible], data.sensor_t[visible]))
        )
        self.observed.set_facecolors(self.sensor_colours[visible])
        self.observed.set_alpha(opacity)
        self.cursor.set_data(data.x[[0, -1]], [time, time])
        self.cursor.set_alpha(opacity if data.t[0] < time < data.t[-1] else 0)


def encode(plate: Plate, folder: Path):
    frame_times = np.arange(DURATION * FPS) / FPS
    # Allow the early steepening to play out before the later, diffusive tail.
    physical = np.interp(frame_times, [0, 0.7, 3.8, 6.2, 8.7], [0, 0, 0.24, 0.53, 1])
    # Keep the axes and the already-conditioned marginal fixed across the loop seam.
    opacity = np.interp(frame_times, [0, 0.38, 9.35, 9.9], [0, 1, 1, 0])
    command = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgba",
        "-video_size",
        f"{WIDTH}x{HEIGHT}",
        "-framerate",
        str(FPS),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(folder / "burgers.mp4"),
        "-c:v",
        "libvpx-vp9",
        "-crf",
        "27",
        "-b:v",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-row-mt",
        "1",
        "-deadline",
        "good",
        "-cpu-used",
        "2",
        "-fflags",
        "+bitexact",
        str(folder / "burgers.webm"),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as ffmpeg:
        for time, alpha in zip(physical, opacity, strict=True):
            plate.update(time, alpha)
            plate.canvas.draw()
            ffmpeg.stdin.write(plate.canvas.buffer_rgba().cast("B"))
        ffmpeg.stdin.close()
        if ffmpeg.wait():
            raise subprocess.CalledProcessError(ffmpeg.returncode, command)

    poster = np.argmin(np.abs(physical - 0.34))
    plate.update(physical[poster])
    plate.fig.savefig(folder / "burgers_poster.png", dpi=DPI)
    plate.print_view()
    plate.fig.savefig(folder / "burgers_still.png", dpi=2 * DPI)
    print(
        f"Poster frame {poster}, t={physical[poster]:.4f}; still t={plate.data.steepest_time:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main", type=Path, required=True, help="Read-only source clone"
    )
    parser.add_argument(
        "--font", type=Path, required=True, help="IBM Plex Sans Regular TTF"
    )
    args = parser.parse_args()

    fontManager.addfont(args.font)
    family = FontProperties(fname=args.font).get_name()
    if family != "IBM Plex Sans":
        parser.error("--font must point to IBM Plex Sans")
    mpl.rcParams.update(
        {
            "font.family": family,
            "font.size": 24,
            "axes.labelsize": 26,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": RULE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "xtick.color": GREY,
            "ytick.color": GREY,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.major.width": 1,
            "ytick.major.width": 1,
            "legend.frameon": False,
            "axes.unicode_minus": True,
        }
    )
    output = ROOT / "portfolio_media"
    output.mkdir(exist_ok=True)
    with jax.default_device(jax.devices("cpu")[0]):
        plate = Plate(calculate(args.main))
    with TemporaryDirectory(dir=output) as directory:
        folder = Path(directory)
        encode(plate, folder)
        for file in folder.iterdir():
            if file.suffix in {".mp4", ".webm"} and file.stat().st_size >= 4_000_000:
                raise ValueError(f"{file.name} exceeds 4 MB")
        for file in folder.iterdir():
            file.replace(output / file.name)


if __name__ == "__main__":
    main()
