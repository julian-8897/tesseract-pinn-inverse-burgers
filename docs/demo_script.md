# Showcase Demo Script

## Setup

```bash
uv sync
./buildall.sh
make test
```

The build step creates the `burgers_solver`, `pinn_jax`, and `pinn_pytorch`
Tesseract images. The standard test suite is Docker-free; the container smoke
test self-skips until the images exist.

## CLI Walk-Through

Run both PINN backends through the same JAX/Optax outer loop:

```bash
uv run burgers-inverse --backend both --epochs 100 --seed 123
```

Point out that both runs use solver-generated noisy Burgers observations and
optimize viscosity in log space. The JAX backend differentiates through a JAX
PINN container; the PyTorch backend uses the same host-side objective while
Tesseract routes `jax.grad` through the PyTorch VJP endpoint. The final table is
a backend consistency check, not a claim of bitwise equivalence because the
models have independent initializations and native-framework kernels.

## Streamlit Walk-Through

Start the app:

```bash
uv run streamlit run app.py
```

Run Summary: confirm the selected backend, true viscosity, initial guess, seed,
and loss-weight mode before starting.

Inverse Solve: watch the inferred viscosity, total loss, and gradient norms
update from the shared `train_inverse` engine.

Solution Field: compare the trained PINN field against the solver ground truth
and inspect the error panel.

BRDR Weights: enable adaptive loss weights and show how the mean component
weights evolve during training.

Tesseract Trace: call out that complete-epoch apply and VJP counts are measured
by wrapping the Tesseract JAX dispatch layer, not estimated. Metric epochs and
BRDR runs legitimately show additional `apply` calls.

Backend Consistency: compare JAX and PyTorch runs with independent initial
parameters under the same observation seed.

## Benchmark Artifacts

```bash
uv run burgers-inverse --backend both --epochs 100 --seed 123 --out runs
```

Artifacts land under `runs/<timestamp>/<backend>/` for single-seed runs:

- `config.json`: flattened run configuration
- `metrics.csv`: per-epoch scalar metrics and measured Tesseract call counts
- `summary.json`: final scalar results and apply/VJP calls per step
