# Burgers inverse problem

The upper-left panel follows the simulated velocity profile as its front steepens and then diffuses. The blue band is the pointwise 5th to 95th percentile of the latent velocity field after propagating 256 joint FMPE draws `(ν, A, φ)` through the Burgers solver. The band shows parameter uncertainty propagated into the field. The sensor noise enters through the posterior conditioning.

The map below shares the same position axis. Time increases downwards and the horizontal cursor locates the profile above. Circles appear at their recorded times and remain at their recorded positions. The heatmap colour encodes the simulated velocity and the circle fill encodes the noisy measurement, both on the same fixed `RdBu_r` scale centred at zero.

The right panel is the density histogram of all 2,000 saved viscosity samples, using 32 bins as in the project README figure. The dashed line is the saved true viscosity. The marginal stays fixed because the flow is conditioned on the complete observation vector. The 5th, 50th and 95th percentiles for this case are 0.04533, 0.05008 and 0.05411, respectively. The project reports mild marginal overconfidence for ν across its SBC cases. The matching model's saved report records a ν SBC rank C2ST of 0.625 and joint TARP results consistent with nominal coverage. The plotted samples are used as saved.

`burgers_poster.png` is a frame partway through the animation, at physical time ≈ 0.340. The 2× `burgers_still.png` shows the full observation map and the slice with the steepest front on the saved time grid (t ≈ 0.111), selected by the most negative spatial finite difference of the simulated velocity. The dotted initial profile gives a reference for the steepening. In both videos the early dynamics receive more playback time, and the moving traces fade out and back in at the loop seam.

## Data provenance

- **README FMPE example, seed 1234.** This is the fixed synthetic demonstration trajectory from the project's [posterior figure](../README.md#posterior-over-all-three-parameters), with `(ν, A, φ) = (0.05, 1, 0)`. It is the README's held-out illustration. The population calibration and held-out coverage runs use separately generated cases, with seeds 99 and 100 respectively.
- **Observations and draws.** `MAIN/img/sbi/fmpe_posterior_samples.npz` contains the exact 64 noisy readings, sensor grid indices, true parameters and 2,000 joint samples from the packaged FMPE Tesseract. The renderer checks the indices and model ID against `MAIN/tesseracts/fmpe_posterior/posterior.pkl`. Replaying `observation_from_theta` at seed 1234 and the checkpoint's noise standard deviation of 0.02 reproduced every archived reading on this CPU. `scripts/plot_sbi_results.py` passes the same seed to the Tesseract sampler when it makes the archive.
- **Solver and subset.** The renderer uses `solve_burgers` in `tesseracts/burgers_solver/tesseract_api.py` with the checkpoint's original 128-by-64 grid, periodic boundaries, and the existing Diffrax settings. It selects 256 rows uniformly without replacement from the archive with NumPy seed `20261124`. It interpolates the solution and the quantile curves between saved times for playback.
- **Calibration.** `MAIN/artifacts/fmpe_calibration.json`, model ID `081153eb45a402d9f1c6de1411da7ccc5600a3b8809c5c1d6b7923deee721c18`. The renderer uses the original posterior and solver, with the calibration report supplying context for the caption.

The only numbers on the plates are axis and colour-scale ticks. Position and time ticks are spaced across the checkpoint grid's [0, 1] domain, including the periodic position endpoint for display. The velocity profile ticks are zero and ± the initial peak absolute velocity. The colour scale endpoints are ± the maximum absolute value across the truth field and the saved observations. The viscosity ticks are placed within the range spanned by the saved draws, with a small margin at either end. The true ν line is set by `theta_true[0]` in the archive.

## Regenerate

Run from the visualisation worktree with the original main clone beside it. Install FFmpeg with the `libx264` and `libvpx-vp9` encoders. The command fetches IBM Plex Sans Regular at a pinned revision into a temporary directory under `portfolio_media/`. The renderer reads the main clone's checkpoint and sample archive, and writes the four media files in this directory.

```sh
uv sync --locked
(
  set -eu
  viz_scratch=$(mktemp -d portfolio_media/.burgers-font.XXXXXX)
  trap 'rm -rf -- "$viz_scratch"' EXIT
  curl -fsSL https://raw.githubusercontent.com/IBM/plex/bc6a7e4b37634eb79785d59781893db63ee3edcb/packages/plex-sans/fonts/complete/ttf/IBMPlexSans-Regular.ttf \
    -o "$viz_scratch/IBMPlexSans-Regular.ttf"
  MPLCONFIGDIR="$viz_scratch" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    uv run --locked python -m scripts.portfolio_media.burgers \
    --main ../tesseract-pinn-inverse-burgers --font "$viz_scratch/IBMPlexSans-Regular.ttf"
)
```

## Website caption

From sparse, noisy readings, flow matching infers viscosity and the initial condition jointly, and the Burgers solver shows how their uncertainty carries into the steepening front, with a viscosity marginal that simulation-based calibration flags as slightly too narrow.
