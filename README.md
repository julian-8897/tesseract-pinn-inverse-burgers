# Backend-Agnostic Inverse Burgers PINN with Tesseract

[![tesseract-core v1.2.0](https://img.shields.io/badge/tesseract--core-v1.2.0-blue)](https://github.com/pasteurlabs/tesseract-core)
[![tesseract-jax v0.2.3](https://img.shields.io/badge/tesseract--jax-v0.2.3-green)](https://github.com/pasteurlabs/tesseract-jax)
[![JAX 0.8.2](https://img.shields.io/badge/JAX-0.8.2-red)](https://github.com/google/jax)
[![PyTorch 2.9.1](https://img.shields.io/badge/PyTorch-2.9.1-orange)](https://pytorch.org/)
[![Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](LICENSE)

**Overview**
This project estimates the viscosity coefficient of the 1D viscous Burgers equation from noisy observations. The inverse solver uses a physics-informed neural network (PINN) exposed as a Tesseract, with interchangeable JAX and PyTorch backends. The outer optimization loop stays in JAX; when the PyTorch backend runs, Tesseract routes gradients through the PyTorch VJP endpoint.

**Key implementations:**
- JAX and PyTorch PINN Tesseracts with `apply`, `vector_jacobian_product`, and `jacobian_vector_product` endpoints
- Differentiable pseudospectral Burgers solver Tesseract for solver-generated observations
- One JAX inverse-training loop that can optimize through either PINN backend
- Configurable loss weights, `log_nu` optimization, seeded runs, and seed-sweep summaries

---

## Contents

- [Problem Statement](#problem-statement)
- [Implementation](#implementation)
- [Configuration](#configuration)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Current Status](#current-status)
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

Synthetic observations come from a differentiable pseudospectral Burgers solver with FFT spatial derivatives, 2/3 dealiasing, and adaptive Diffrax time integration. The solver uses the same sinusoidal initial condition and periodic boundary conditions as the PINN loss. The inverse pipeline samples noisy observations from the nonlinear solution field with additive Gaussian noise, using $\sigma = 0.02$ by default.

A physics-informed neural network (PINN) minimizes a combined loss function:

$$\mathcal{L} = \lambda_{\text{data}} \cdot \mathcal{L}_{\text{data}} + \lambda_{\text{physics}} \cdot \mathcal{L}_{\text{physics}} + \lambda_{\text{IC}} \cdot \mathcal{L}_{\text{IC}} + \lambda_{\text{BC}} \cdot \mathcal{L}_{\text{BC}}$$

where:
- $\mathcal{L}_{\text{data}}$: mean squared error between predictions and observations
- $\mathcal{L}_{\text{physics}}$: PDE residual at collocation points
- $\mathcal{L}_{\text{IC}}$: initial condition violation
- $\mathcal{L}_{\text{BC}}$: boundary condition violation

## Implementation

### Architecture

The PINN uses Fourier feature encoding to mitigate spectral bias:

```
Input (x, t) ∈ ℝ²
    ↓
Fourier encoding: [x, t, sin(x·B_x), cos(x·B_x), sin(t·B_t), cos(t·B_t)]
    ↓
MLP: 130 → 64 → 64 → 64 → 1 (tanh activations)
    ↓
Output: u(x, t)
```

Derivatives ($\partial u/\partial x$, $\partial u/\partial t$, $\partial^2 u/\partial x^2$) are computed via automatic differentiation within each Tesseract using the native framework's autograd: `jax.grad` for the JAX backend and `torch.autograd.grad` for the PyTorch backend.

### Tesseract Endpoints

The `pinn_jax` and `pinn_pytorch` Tesseracts implement:

1. **apply(inputs)**: Forward pass returning u_pred, u_x, u_t, u_xx
2. **vector_jacobian_product(...)**: Reverse-mode AD for gradient computation
3. **jacobian_vector_product(...)**: Forward-mode AD for sensitivity analysis (not used by the inverse trainer)

The `burgers_solver` Tesseract implements the same `apply`, VJP, and JVP endpoint pattern for differentiable solver runs. In the current inverse-problem demo it is used offline to generate ground-truth observations; differentiating through the solver during optimization is a planned extension.

Input/output schemas use Tesseract's `Differentiable[Array[...]]` annotations to declare which fields participate in autodiff.

### Cross-Framework Gradient Flow

The CLI path in `inverse_problem.py` optimizes the viscosity in log space:

```python
# Load backend (JAX or PyTorch)
pinn = Tesseract.from_image("pinn_jax")  # or "pinn_pytorch"

# Keep viscosity positive without clipping the optimized variable
nu = jnp.exp(log_nu)
loss = compute_loss(nu, params, x_obs, t_obs, u_obs, ..., pinn)

# System-level gradients computed via Tesseract VJP (regardless of backend)
grad_log_nu = jax.grad(compute_loss_from_log_viscosity, argnums=0)
grad_params = jax.grad(compute_loss, argnums=1)

# When jax.grad is called, it triggers Tesseract's VJP endpoint
# For PyTorch backend: Tesseract VJP internally uses torch.autograd.grad
# For JAX backend: Tesseract VJP internally uses jax.grad
log_nu_grad = grad_log_nu(log_nu, params, ...)
p_grad = grad_params(nu, params, ...)
```

> **Key point:** The system-level gradients ($\partial \mathcal{L}/\partial \log\nu$ and $\partial \mathcal{L}/\partial \text{params}$) use Tesseract's `vector_jacobian_product` endpoint for both backends. The backend selection determines which autograd implementation Tesseract uses inside the VJP computation.

The inverse loop optimizes `log_nu` and evaluates the PDE residual with `nu = exp(log_nu)`. This keeps the inferred viscosity positive without clipping the optimization variable.

### Configuration

Run settings live in typed dataclasses in `configs.py`:

- `ProblemConfig`: true viscosity, initial viscosity, and domain
- `DataConfig`: observation count, noise level, and seed
- `TrainingConfig`: epochs, learning rates, and collocation/IC/BC sample counts
- `LossWeights`: data, physics, initial-condition, and boundary-condition weights
- `RunConfig`: full run configuration consumed by `inverse_problem.py`

The CLI still exposes the common knobs directly. Internally, `inverse_problem.py` converts CLI arguments into a `RunConfig`, so Streamlit or future scripts can call the same training path without duplicating defaults.

### Project Structure

```
tesseract-pinn-inverse-burgers/
├── configs.py                 # Dataclass run/problem/data/training configs
├── inverse_problem.py         # CLI demo comparing JAX/PyTorch backends
├── app.py                     # Streamlit interactive interface
├── buildall.sh                # Builds Docker containers for all Tesseracts
├── pyproject.toml
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

# Option A: using uv
# Install uv if missing: pip install uv
uv venv
source .venv/bin/activate
uv pip install -e .

# Option B: using python venv
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Build Tesseract containers (requires Docker running)
./buildall.sh

# Verify built images
docker images | grep -E 'burgers_solver|pinn'
```

---

## Usage

### CLI

```bash
# Compare both backends
uv run python inverse_problem.py --backend both --epochs 100

# Single backend
uv run python inverse_problem.py --backend jax --epochs 50
uv run python inverse_problem.py --backend pytorch --epochs 50

# Reproducible single-seed run
uv run python inverse_problem.py --backend jax --epochs 50 --seed 123

# Override PINN loss weights
uv run python inverse_problem.py --backend jax --epochs 50 \
  --w-data 1.0 --w-physics 0.2 --w-ic 0.5 --w-bc 0.5

# Seed sweep with summary statistics
uv run python inverse_problem.py --backend jax --epochs 50 --seeds 0 1 2 3 4
```

### Streamlit

```bash
uv run streamlit run app.py
```

The Streamlit app provides:
- Adjustable hyperparameters (viscosity, noise, learning rate)
- Real-time training visualization
- Gradient flow inspector (Tesseract API call statistics)
- Solution comparison plots

The CLI is the reference path for the current solver-backed observations, dataclass configs, seeded runs, and `log_nu` optimization. The Streamlit app still needs a cleanup pass to remove older analytical-reference labels and fully match the CLI pipeline.

### Tests

```bash
uv run --with pytest python -m pytest tests -q
```

---

## Results
The repository includes example figures from the PINN comparison workflow:
<table align="center" cellpadding="12">
  <tr>
    <td align="center">
      <img src="img/pinn_solution_comparison.png" alt="PINN solution comparison" width="900"/>
      <div><em>PINN solution comparison (JAX vs PyTorch)</em></div>
    </td>
  </tr>
</table>

PINN $u(x,t)$ solution plots for both backends:
<table align="center" cellpadding="12">
  <tr>
    <td align="center">
      <img src="img/pinn_field_solution_jax.png" alt="PINN solution comparison" width="900"/>
      <div><em>PINN vs Solver Ground Truth (JAX)</em></div>
    </td>
  </tr>
</table>

<table align="center" cellpadding="12">
  <tr>
    <td align="center">
      <img src="img/pinn_field_solution_pytorch.png" alt="PINN solution comparison" width="900"/>
      <div><em>PINN vs Solver Ground Truth (PyTorch)</em></div>
    </td>
  </tr>
</table>

Regenerate these figures before treating them as benchmark results for the latest CLI path. Recent changes moved observation generation to the Burgers solver, added `log_nu` optimization, and added seed sweeps.

## Current Status

- The CLI inverse pipeline uses solver-generated noisy Burgers observations.
- The solver generates observations offline for training. End-to-end differentiation through the solver during inverse training remains future work.
- The PINN backend can be JAX or PyTorch; Tesseract exposes both through the same `apply`/VJP/JVP interface.
- The Streamlit app is useful for interactive inspection, but it has not yet been fully aligned with the latest CLI refactor.

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
