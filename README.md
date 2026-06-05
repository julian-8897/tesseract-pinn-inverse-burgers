1# Backend-Agnostic Inverse Burgers with Tesseract

[![tesseract-core v1.2.0](https://img.shields.io/badge/tesseract--core-v1.2.0-blue)](https://github.com/pasteurlabs/tesseract-core)
[![tesseract-jax v0.2.3](https://img.shields.io/badge/tesseract--jax-v0.2.3-green)](https://github.com/pasteurlabs/tesseract-jax)
[![JAX 0.8.2](https://img.shields.io/badge/JAX-0.8.2-red)](https://github.com/google/jax)
[![PyTorch 2.9.1](https://img.shields.io/badge/PyTorch-2.9.1-orange)](https://pytorch.org/)
[![Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](LICENSE)
[![CI](https://github.com/julian-8897/tesseract-pinn-inverse-burgers/actions/workflows/ci.yml/badge.svg)](https://github.com/julian-8897/tesseract-pinn-inverse-burgers/actions/workflows/ci.yml)

**Overview**
This project estimates the viscosity coefficient of the 1D viscous Burgers
equation from sparse noisy observations, and showcases **Tesseract as a registry
of framework-agnostic, swappable, differentiable model components**. The physics
solver, the PINN surrogate, and (next) the posterior sampler are each packaged as
an independent Tesseract behind one uniform typed schema — so you swap a JAX
component for a PyTorch one by changing an image name, with identical calling
code and no shared environment.

The inverse problem is solved two ways, both differentiating through a Tesseract:

- **Solver-adjoint inversion** (`--mode solver-inverse`): `jax.grad` flows through
  the differentiable **solver** Tesseract's VJP — a PDE-constrained baseline.
- **PINN inversion** (`--mode pinn`): `jax.grad` flows through the **PINN**
  Tesseract's VJP, with the PINN backend interchangeable between **JAX and
  PyTorch** (the backend-agnostic showcase).

`--mode compare` runs all three (solver-adjoint, PINN-JAX, PINN-PyTorch) on the
same physics and prints a unified table.

**Key implementations:**
- Three swappable, framework-agnostic Tesseracts: a JAX pseudospectral **solver**, and a **PINN** with interchangeable JAX/PyTorch backends behind one `apply`/`vector_jacobian_product` contract
- Two differentiable inverse methods (solver-adjoint and PINN) compared on identical physics
- Shared callback-driven training engines; measured Tesseract apply/VJP telemetry per gradient step
- Configurable loss weights, optional BRDR adaptive residual weighting, `log_nu` optimization, seeded runs, seed sweeps, reproducible artifacts
- Roadmap: amortized flow-matching posterior over `nu` for calibrated uncertainty (see `INVERSE_FMPE_PLAN.md`)

---

## Contents

- [Problem Statement](#problem-statement)
- [Implementation](#implementation)
- [Configuration](#configuration)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Current Status](#current-status)
- [Limitations and Roadmap](#limitations-and-roadmap)
- [References](#references)

---

## Problem Statement

Given noisy observations of the 1D Burgers equation solution, infer the unknown viscosity parameter $\nu$:

$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}
$$

where:
- $u(x, t)$ is the velocity field on $[0, 1] \times [0, T]$
- $\nu$ is the kinematic viscosity (inferred parameter)
- Initial condition: $u(x, 0) = \sin(2\pi x)$
- Boundary conditions: periodic on $[0, 1]$

Synthetic observations come from a differentiable pseudospectral Burgers solver
with FFT spatial derivatives, 2/3 dealiasing, and adaptive Diffrax time
integration. The solver uses the same sinusoidal initial condition and periodic
boundary conditions as the PINN loss. The inverse pipeline samples noisy
observations from the nonlinear solution field with additive Gaussian noise,
using $\sigma = 0.02$ by default.

A physics-informed neural network (PINN) minimizes a combined loss function:

$$\mathcal{L} = \lambda_{\text{data}} \cdot \mathcal{L}_{\text{data}} + \lambda_{\text{physics}} \cdot \mathcal{L}_{\text{physics}} + \lambda_{\text{IC}} \cdot \mathcal{L}_{\text{IC}} + \lambda_{\text{BC}} \cdot \mathcal{L}_{\text{BC}}$$

where:
- $\mathcal{L}_{\text{data}}$: mean squared error between predictions and observations
- $\mathcal{L}_{\text{physics}}$: PDE residual at collocation points
- $\mathcal{L}_{\text{IC}}$: initial condition violation
- $\mathcal{L}_{\text{BC}}$: boundary condition violation

## Implementation

### Architecture

The PINN uses fixed Fourier feature encoding to mitigate spectral bias:

```
Input (x, t) ∈ ℝ²
    ↓
Fixed Fourier encoding: [x, t, sin(x·B_x), cos(x·B_x), sin(t·B_t), cos(t·B_t)]
    ↓
MLP: 130 → 64 → 64 → 64 → 1 (tanh activations)
    ↓
Output: u(x, t)
```

The Fourier frequencies are deterministic backend-independent constants. The
flattened trainable parameter vector contains only MLP weights and biases,
giving the JAX and PyTorch containers the same trainable-parameter contract.
Derivatives ($\partial u/\partial x$, $\partial u/\partial t$,
$\partial^2 u/\partial x^2$) are computed via automatic differentiation within
each Tesseract using the native framework's autograd: `jax.grad` for the JAX
backend and `torch.autograd.grad` for the PyTorch backend.

### Tesseract Endpoints

The inverse-training showcase exercises these `pinn_jax` and `pinn_pytorch` endpoints:

1. **apply(inputs)**: Forward pass returning u_pred, u_x, u_t, u_xx
2. **vector_jacobian_product(...)**: Reverse-mode AD for gradient computation

JVP endpoints are secondary to the current demo because the inverse trainer uses
reverse-mode gradients through `jax.value_and_grad`. The `burgers_solver`
Tesseract implements the same `apply`, VJP, and JVP endpoint pattern for
differentiable solver runs. In the current inverse-problem demo it is used
offline to generate ground-truth observations and visualization fields;
differentiating through the solver during optimization is a planned extension.

Input/output schemas use Tesseract's `Differentiable[Array[...]]` annotations to declare which fields participate in autodiff.

### Cross-Framework Gradient Flow

The CLI path in `inverse_problem.py` optimizes the viscosity in log space:

```python
# Load backend (JAX or PyTorch)
pinn = Tesseract.from_image("pinn_jax")  # or "pinn_pytorch"

# One reverse-mode pass returns loss and both optimized gradients.
loss_and_grads = jax.value_and_grad(_loss_from_log_and_params, argnums=(0, 1))
loss, (log_nu_grad, params_grad) = loss_and_grads(log_nu, params, ..., pinn)

# The objective evaluates the PDE residual with nu = exp(log_nu).
# When jax.value_and_grad runs, it triggers Tesseract's VJP endpoint.
# For PyTorch backend: Tesseract VJP internally uses torch.autograd.grad
# For JAX backend: Tesseract VJP internally uses jax.grad
```

> **Key point:** The system-level gradients
> ($\partial \mathcal{L}/\partial \log\nu$ and
> $\partial \mathcal{L}/\partial \text{params}$) use Tesseract's
> `vector_jacobian_product` endpoint for both backends. The backend selection
> determines which autograd implementation Tesseract uses inside the VJP
> computation.

The inverse loop optimizes `log_nu` and evaluates the PDE residual with
`nu = exp(log_nu)`. This keeps the inferred viscosity positive. Optional
viscosity clipping and viscosity warmup are available through `TrainingConfig`
and the Streamlit app for more stable interactive runs.

### Shared Training Engine

`inverse_problem.py` exposes `train_inverse(config, *, pinn=None, callback=None,
metrics_every=20)`. Both the CLI and Streamlit UI call this function. Presentation
is delegated to callbacks:

- `RichProgressCallback` renders CLI progress and final tables.
- `MetricsRecorderCallback` records per-epoch metrics for CLI artifacts.
- `StreamlitTrainingCallback` drives the app's live plots and trace panels.

Each optimization step uses one `jax.value_and_grad` over `(log_nu, params)`.
`TesseractCallCounter` wraps the `tesseract_jax` dispatch layer to report real
container calls; the current inverse step measures 5 `apply` calls and 5 VJP
calls per gradient step.

### Configuration

Run settings live in validated typed dataclasses in `configs.py`:

- `ProblemConfig`: true viscosity, initial viscosity, and domain
- `DataConfig`: observation count, noise level, and seed
- `TrainingConfig`: epochs, learning rates, collocation/IC/BC sample counts, BRDR settings, optional viscosity warmup, and optional viscosity clipping
- `LossWeights`: data, physics, initial-condition, and boundary-condition weights
- `RunConfig`: full run configuration consumed by `inverse_problem.py`

The CLI exposes the common knobs directly. Internally, `inverse_problem.py`
converts CLI arguments into a `RunConfig`, so Streamlit, tests, and figure
scripts call the same training path without duplicating defaults.

### Project Structure

```
tesseract-pinn-inverse-burgers/
├── configs.py                 # Dataclass run/problem/data/training configs
├── inverse_problem.py         # CLI demo comparing JAX/PyTorch backends
├── app.py                     # Streamlit interactive interface
├── buildall.sh                # Builds Docker containers for all Tesseracts
├── Makefile                   # Common verification and demo commands
├── pyproject.toml
├── scripts/
│   └── regenerate_figures.py  # Rebuilds README figures from current training path
├── tests/                     # Unit tests plus optional container smoke test
└── tesseracts/
    ├── burgers_solver/
    │   ├── tesseract_api.py        # Differentiable pseudospectral Burgers solver
    │   ├── tesseract_config.yaml
    │   └── tesseract_requirements.txt
    ├── pinn_jax/
    │   ├── tesseract_api.py        # JAX/Equinox PINN with Tesseract endpoints
    │   ├── tesseract_config.yaml
    │   └── tesseract_requirements.txt
    └── pinn_pytorch/
        ├── tesseract_api.py        # PyTorch PINN with Tesseract endpoints
        ├── tesseract_config.yaml
        └── tesseract_requirements.txt
```

---

## Installation

**Requirements:** Python >=3.13, Docker, and the Tesseract CLI.

```bash
# Clone repository
git clone https://github.com/julian-8897/tesseract-pinn-inverse-burgers.git
cd tesseract-pinn-inverse-burgers

# Install Python dependencies from pyproject.toml / uv.lock
uv sync

# Build Tesseract containers (requires Docker running)
./buildall.sh

# Verify built images
docker images | grep -E 'burgers_solver|pinn'
```

`pyproject.toml` and `uv.lock` are the dependency source of truth. No separate
`requirements.txt` is maintained.

---

## Usage

### CLI

```bash
# Compare inverse methods: solver-adjoint vs PINN (JAX & PyTorch), one table
uv run python inverse_problem.py --mode compare --epochs 100 --seed 123

# Solver-adjoint inversion (jax.grad through the solver Tesseract VJP)
uv run python inverse_problem.py --mode solver-inverse --epochs 80

# PINN inversion: compare both backends
uv run python inverse_problem.py --backend both --epochs 100

# PINN inversion: single backend
uv run python inverse_problem.py --backend jax --epochs 50
uv run python inverse_problem.py --backend pytorch --epochs 50

# Reproducible single-seed run
uv run python inverse_problem.py --backend jax --epochs 50 --seed 123

# Override PINN loss weights
uv run python inverse_problem.py --backend jax --epochs 50 \
  --w-data 1.0 --w-physics 0.2 --w-ic 0.5 --w-bc 0.5

# Use BRDR pointwise adaptive loss weights
uv run python inverse_problem.py --backend jax --epochs 100 \
  --adaptive-loss-weights

# Seed sweep with summary statistics
uv run python inverse_problem.py --backend jax --epochs 50 --seeds 0 1 2 3 4

# Write reproducible benchmark artifacts
uv run python inverse_problem.py --backend both --epochs 100 --seed 123 --out runs
```

When `--out` is set, the CLI writes artifacts under
`runs/<timestamp>/<backend>/`:

- `config.json`: JSON-serializable `RunConfig`
- `metrics.csv`: per-epoch viscosity, loss, gradient norms, timings, and measured Tesseract call counts
- `summary.json`: final scalar results and apply/VJP calls per step

### Streamlit

```bash
uv run streamlit run app.py
```

The Streamlit app provides:
- Adjustable hyperparameters, observation counts, collocation counts, loss weights, viscosity warmup, and optional viscosity clipping
- Real-time training visualization
- Optional BRDR adaptive loss weighting and component mean-weight plots
- Tesseract trace panel with measured apply/VJP call counts
- Solution field plots against solver ground truth
- Backend consistency report for JAX vs PyTorch runs

The CLI is the reference path for dataclass configs and seeded runs. The Streamlit app follows the same solver-backed observation generation, `log_nu` optimization, and optional adaptive loss-weighting path for interactive runs.

### Tests

```bash
make compile
make lint
make test
make smoke
```

`make smoke` runs the live container round-trip test and skips cleanly when the
local Tesseract images have not been built.

---

## Results
The repository includes regenerated figures from the current solver-backed
workflow. These were produced with `scripts/regenerate_figures.py` using both
PINN Tesseract backends, 100 training epochs, seed 123, and a 160 x 90
visualization grid.

<table align="center" cellpadding="12">
  <tr>
    <td align="center">
      <img src="img/pinn_solution_comparison.png" alt="Backend consistency dashboard comparing JAX and PyTorch PINN Tesseracts" width="900"/>
      <div><em>Backend consistency dashboard: viscosity trajectory, objective loss, final estimates, backend spread, and measured apply/VJP calls.</em></div>
    </td>
  </tr>
</table>

PINN $u(x,t)$ field reconstructions against the pseudospectral Burgers solver
ground truth:

<table align="center" cellpadding="12">
  <tr>
    <td align="center">
      <img src="img/pinn_field_solution_jax.png" alt="JAX PINN field compared with Burgers solver ground truth and absolute error" width="900"/>
      <div><em>JAX PINN reconstruction vs solver ground truth with shared field color scale and absolute error.</em></div>
    </td>
  </tr>
</table>

<table align="center" cellpadding="12">
  <tr>
    <td align="center">
      <img src="img/pinn_field_solution_pytorch.png" alt="PyTorch PINN field compared with Burgers solver ground truth and absolute error" width="900"/>
      <div><em>PyTorch PINN reconstruction vs solver ground truth through the same JAX/Optax outer loop.</em></div>
    </td>
  </tr>
</table>

To regenerate these figures after training-path changes:

```bash
uv run python scripts/regenerate_figures.py --epochs 100 --seed 123 --nx 160 --nt 90
```

## Current Status

- Two differentiable inverse methods: **solver-adjoint** (`--mode solver-inverse`, `jax.grad` through the solver VJP) and **PINN** (`--mode pinn`, `jax.grad` through the PINN VJP); `--mode compare` tabulates both.
- For scalar viscosity inversion the solver-adjoint method is dramatically more accurate and cheaper per step than the PINN; the PINN is mesh-free and needs no solver but converges more slowly. The two PINN backends (JAX, PyTorch) agree, demonstrating backend-agnostic consistency.
- Each step uses one `jax.value_and_grad`; Tesseract apply/VJP counts are measured through the `tesseract_jax` dispatch layer.
- Loss weights can be fixed or adapted with opt-in BRDR pointwise residual weighting; the CLI and Streamlit app share callback-driven training engines.
- Container smoke coverage verifies `Tesseract.from_image(...)` through `apply` and VJP when local images are available; reproducible artifacts and figures are supported.

## Limitations and Roadmap

- **Frontier UQ (next):** an amortized **flow-matching posterior** over `nu` (a third swappable Tesseract) for calibrated uncertainty, validated with simulation-based calibration / coverage. See `INVERSE_FMPE_PLAN.md`.
- **Single scalar parameter inversion:** the inverse target is viscosity `nu` only; joint inference of initial-condition parameters is a natural extension.
- **Model-misspecification sidebar (experimental):** a KdV-Burgers truth oracle (`solve_kdv_burgers`) plus a learned-discrepancy hybrid (`train_hybrid_inverse`) are included to *demonstrate* a known failure mode — naive calibration-with-discrepancy is confounded with the calibration parameter and biases it (Brynjarsdóttir & O'Hagan, 2014). This is documented as a limitation, not a headline result, and motivates the posterior treatment above.
- **No checkpointing; PINN model reconstructed from `params_flat` per call.**

## References

**Tesseract Documentation:**
- [Tesseract Core](https://github.com/pasteurlabs/tesseract-core) - Main repository and CLI
- [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax) - JAX integration layer
- [Creating Tesseracts](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/creating-tesseracts/create.html) - Implementation guide
- [Differentiable Programming](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/introduction/differentiable-programming.html) - VJP/JVP concepts

**Related Publications to PINNs:**
- Raissi, M., Perdikaris, P., & Karniadakis, G. E., ["Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations"](https://www.sciencedirect.com/science/article/pii/S0021999118307125), *Journal of Computational Physics* 378 (2019): 686-707
- Tancik, M., Srinivasan, P. P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R., & Ng, R., ["Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"](https://arxiv.org/abs/2006.10739), *NeurIPS* 2020

---

## Compatibility

| Component | Version | Notes |
|-----------|---------|-------|
| tesseract-core | 1.2.0 | Runtime and CLI |
| tesseract-jax | 0.2.3 | JAX integration |
| Python | >=3.13 | Project requirement in `pyproject.toml` |
| Docker | latest | Container execution |
| JAX | 0.8.2 | CPU backend |
| PyTorch | 2.9.1 | PyTorch backend |
| Equinox | 0.13.2 | JAX PINN modules |
| Optax | 0.2.6 | JAX optimizer |

**Tested platforms:** macOS (Apple Silicon)

---

## License

Licensed under [Apache License 2.0](LICENSE).
