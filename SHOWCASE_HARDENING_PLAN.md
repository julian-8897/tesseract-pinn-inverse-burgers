# Showcase Hardening Plan (Phases 2–3 + deferred feature)

> Handoff doc for an implementer (Codex) working on the `tesseract-pinn-inverse-burgers`
> repo. Self-contained: references symbols and file paths rather than raw line numbers,
> because Phase 1 changed the files substantially. Scope is **production polish for a
> Tesseract / Pasteur Labs showcase demo**, not new science. Read "Current State" first,
> then implement phases in order. Each task lists rationale, exact anchors, and verification.

---

## Current State (Phase 1 is DONE, in the working tree, uncommitted)

The repo demonstrates Tesseract cross-framework autodiff: a single JAX/Optax outer loop
infers the viscosity `nu` of 1D viscous Burgers by differentiating through either a
JAX/Equinox PINN or a PyTorch PINN exposed as Tesseract containers (VJP endpoints).
Observations come from a differentiable pseudospectral Burgers solver
(`tesseracts/burgers_solver/`).

**Phase 1 already landed these changes — do NOT redo them; build on them:**

- **Shared training engine** in `inverse_problem.py`:
  `train_inverse(config, *, pinn=None, callback=None, metrics_every=20)`.
  Both the CLI and the Streamlit app call it; presentation is delegated to callbacks.
  - One `jax.value_and_grad` over `(log_nu, params)` (was: two separate `jax.grad` +
    standalone loss). `_loss_from_log_and_params` is the differentiated function.
  - Real Tesseract call telemetry via `TesseractCallCounter` + `count_tesseract_calls`
    (wraps `tesseract_jax.tesseract_compat.Jaxeract.apply` / `.vector_jacobian_product` /
    `.jacobian_vector_product`). Measured cost is **5 apply / 5 VJP per gradient step**.
  - Callback contract: `TrainingCallback` base with `on_start(context)`, `on_epoch(record)`,
    `on_finish(result)`. `RichProgressCallback` is the CLI implementation;
    `StreamlitTrainingCallback` (in `app.py`) is the UI implementation.
- **`EpochRecord`** (dataclass in `inverse_problem.py`) is passed to `on_epoch`. Fields:
  `epoch, n_epochs, viscosity, log_viscosity, loss, visc_grad_norm, param_grad_norm,
  epoch_time, apply_calls, vjp_calls, effective_weights, viscosity_updated, param_count,
  brdr_weights, loss_components`. `loss_components` is `None` except on the `metrics_every`
  cadence (and the final epoch), when it is a dict
  `{"total","data","physics","ic","bc"}`.
- **`train_inverse` returns a result dict** with keys:
  `backend, tesseract_image, final_viscosity, true_viscosity, relative_error, avg_time_ms,
  viscosity_history, log_viscosity_history, loss_history, loss_weights, loss_weight_history,
  brdr_state, adaptive_loss_weights, seed, config, params_flat, warmup_epochs, observations,
  pinn`.
  Note: `params_flat` (jnp array), `pinn` (Tesseract), `config` (RunConfig dataclass),
  `observations` (tuple of jnp arrays), `brdr_state` (dict of jnp arrays) are **not
  JSON-serializable** — a serializer must select fields.
- **Config validation** added to every dataclass `__post_init__` in `configs.py`.
  `TrainingConfig` gained fields: `viscosity_warmup_epochs: int = 0`,
  `clip_log_viscosity: bool = False`, `nu_clip_min: float = 1e-4`, `nu_clip_max: float = 0.5`.
- **CLI image-failure handling**: `docker_image_available(image_name)`,
  `TesseractImageNotFoundError`, `ensure_image_available(image_name)` in `inverse_problem.py`;
  `run_inverse_problem` calls `ensure_image_available(...)` before training; the `__main__`
  block catches `TesseractImageNotFoundError` and `sys.exit(1)` with a clean message.
- **PyTorch JVP honesty**: `jacobian_vector_product` in
  `tesseracts/pinn_pytorch/tesseract_api.py` now raises `NotImplementedError` for any output
  other than `u_pred` and for `params_flat` tangents (previously returned a silent partial).
- **"Backend Equivalence" → "Backend Consistency"** reframe throughout `app.py`.
- All three container images were rebuilt to match the (uncommitted) fixed-Fourier API.

**Baseline facts an implementer needs:**
- Python `>=3.13`; environment via `uv` + `pyproject.toml`. Trainable PINN param vector is
  16769 floats; both backends share that contract.
- The Burgers solver, the JAX PINN, and the PyTorch PINN modules are **all named
  `tesseract_api`**. Any host-side import must manage `sys.path` and `sys.modules` to avoid
  clashes — see the existing helpers `get_initial_params` and `get_burgers_solver` in
  `inverse_problem.py` for the established pattern. Reuse them; do not reimplement.
- Verification commands today:
  ```bash
  uv run python -m py_compile app.py inverse_problem.py configs.py
  uv run --with pytest python -m pytest tests -q          # 20 tests, ~7s, no Docker
  uv run --with ruff ruff check app.py inverse_problem.py configs.py
  uv run python inverse_problem.py --backend both --epochs 40 --seed 123   # needs Docker
  uv run streamlit run app.py                            # needs Docker
  ./buildall.sh                                          # rebuild images, needs Docker
  ```
- Tests directory contents: `test_configs.py`, `test_log_viscosity.py`, `test_loss_weights.py`,
  `test_seed_sweep.py`, `test_generate_observations.py`, `test_backend_equivalence.py`,
  `test_burgers_solver.py`. None currently exercise a live container round-trip.

---

## Phase 2 — Test & reproducibility infrastructure

Goal: prove the *product* (the container round-trip) is exercised, make runs reproducible
and inspectable, and give the repo the CI/tooling signals a showcase repo is expected to have.

### 2.1 Container smoke test — `tests/test_container_smoke.py` (HIGHEST VALUE)

**Why:** all 20 existing tests bypass the container (they import the `tesseract_api` modules
directly or call `solve_burgers`). Nothing verifies `Tesseract.from_image(...)` → `apply` →
VJP actually works. This is the gap that let stale images ship a broken param contract.

**Implement:** a pytest module that **skips cleanly when Docker images are absent** so it is
CI-safe, and runs a minimal real round-trip when they exist.

```python
# tests/test_container_smoke.py
import pathlib, sys
import jax, jax.numpy as jnp
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import inverse_problem as ip
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

BACKENDS = ["jax", "pytorch"]

@pytest.mark.parametrize("backend", BACKENDS)
def test_container_apply_and_vjp(backend):
    image = ip.image_name_for_backend(backend)
    if not ip.docker_image_available(image):
        pytest.skip(f"Tesseract image '{image}' not built; run ./buildall.sh")

    params = ip.get_initial_params(backend, seed=0)
    x = jnp.linspace(0.1, 0.9, 5, dtype=jnp.float32)
    t = jnp.linspace(0.05, 0.5, 5, dtype=jnp.float32)

    pinn = Tesseract.from_image(image)
    with pinn:
        # forward: apply returns the four fields
        out = apply_tesseract(pinn, {"x": x, "t": t, "params_flat": params})
        for key in ("u_pred", "u_x", "u_t", "u_xx"):
            assert out[key].shape == x.shape

        # backward: jax.grad through the container triggers the VJP endpoint
        def loss(p):
            r = apply_tesseract(pinn, {"x": x, "t": t, "params_flat": p})
            return jnp.sum(r["u_pred"] ** 2)

        g = jax.grad(loss)(params)
        assert g.shape == params.shape
        assert jnp.all(jnp.isfinite(g))
        assert float(jnp.linalg.norm(g)) > 0.0
```

**Optionally** add a telemetry assertion using `ip.count_tesseract_calls` to lock in the
"5 apply / 5 VJP per step" contract for a full `train_inverse` epoch (guards against silent
regressions in how many container calls a step costs). Keep it a separate test so the basic
smoke test stays fast.

**Verify:** `uv run --with pytest python -m pytest tests/test_container_smoke.py -q` (passes
with Docker, skips without).

### 2.2 Reproducible benchmark artifacts — `inverse_problem.py`

**Why:** the CLI currently writes nothing. A showcase needs a one-command reproducible run
that drops a small, inspectable artifact.

**Implement:**
1. A `MetricsRecorderCallback(RichProgressCallback)` that, in `on_epoch`, appends a per-epoch
   row dict (`epoch, viscosity, log_viscosity, loss, visc_grad_norm, param_grad_norm,
   epoch_time, apply_calls, vjp_calls, viscosity_updated`) to `self.rows`, then calls
   `super().on_epoch(record)`. This composes recording with the existing CLI progress bar.
2. A `write_run_artifacts(result, rows, out_dir)` helper that writes, under
   `out_dir` (create with `pathlib.Path(...).mkdir(parents=True, exist_ok=True)`):
   - `config.json` — the run config flattened to plain types. Source it from `result["config"]`
     (a `RunConfig`); use `dataclasses.asdict` then coerce tuples to lists.
   - `metrics.csv` — the per-epoch `rows` (use `csv.DictWriter`; no pandas dependency needed in
     the CLI path).
   - `summary.json` — selected scalar fields from `result`:
     `backend, tesseract_image, seed, true_viscosity, final_viscosity, relative_error,
     avg_time_ms, epochs (= len(viscosity_history)-1), apply_calls_per_step,
     vjp_calls_per_step` (read the last from any recorded row), `adaptive_loss_weights`.
     **Do not** dump `params_flat`, `pinn`, `observations`, or `brdr_state`.
3. A CLI flag `--out DIR` (argparse). When set, write artifacts to
   `DIR/<timestamp>/<backend>/` for each backend that ran. `compare_backends` returns a dict
   keyed by backend; iterate it. Use `time.strftime("%Y%m%dT%H%M%S")` for the timestamp.
4. Wire it: `run_inverse_problem` / `run_single_backend` / `compare_backends` should accept an
   optional `out_dir` (or have `__main__` orchestrate writing from the returned result(s) +
   the recorder rows). Keep the public signatures the existing tests rely on intact
   (`run_inverse_problem`, `summarize_seed_results`, `build_run_config` must not change shape).
5. Add `runs/` to `.gitignore`.

**Verify:**
```bash
uv run python inverse_problem.py --backend jax --epochs 30 --seed 123 --out runs
test -f runs/*/jax/summary.json && test -f runs/*/jax/metrics.csv && test -f runs/*/jax/config.json
```
`summary.json` should show `final_viscosity` near the inferred value and
`vjp_calls_per_step == 5`.

### 2.3 GitHub Actions CI — `.github/workflows/ci.yml`

**Why:** 20 tests run in ~7s with no Docker; a green CI badge is table stakes and currently
absent (`.github/` does not exist).

**Implement:** a workflow on `push` and `pull_request` that:
- Installs `uv` (use `astral-sh/setup-uv@v6` or `pipx install uv`).
- `uv sync` (or `uv pip install -e .`) to resolve deps (jax, torch, diffrax, etc.).
- Runs `uv run ruff check .` and `uv run ruff format --check .`.
- Runs `uv run pytest tests -q`. The container smoke test (2.1) self-skips with no Docker, so
  CI stays Docker-free. Do **not** attempt to build Tesseract images in CI.
- Pin `python-version: "3.13"`.

Add a CI status badge to the top of `README.md` next to the existing badges.

**Verify:** push a branch; the workflow is green; locally
`uv run ruff format --check . && uv run pytest tests -q` passes.

### 2.4 Lint config + task runner — `pyproject.toml`, `Makefile`

**Why:** ruff is in `.pre-commit-config.yaml` but there is no `[tool.ruff]` block (defaults
only) and no documented task runner. The verification commands are tribal knowledge.

**Implement:**
- Add to `pyproject.toml`:
  ```toml
  [tool.ruff]
  line-length = 88
  target-version = "py313"

  [tool.ruff.lint]
  select = ["E", "F", "I", "UP", "B"]
  ```
  Run `uv run ruff check --fix .` and `uv run ruff format .` once and commit the (small)
  normalization so CI's `--check` passes. Confirm no behavioral diffs (review the format diff).
- Add a `Makefile` (or `justfile`) with targets:
  `compile`, `lint`, `format`, `test`, `build-tesseracts` (`./buildall.sh`),
  `smoke` (`pytest tests/test_container_smoke.py -q`), `benchmark`
  (`python inverse_problem.py --backend both --epochs 100 --seed 123 --out runs`),
  `run-cli`, `run-app` (`streamlit run app.py`). Each target should use `uv run`.
- Update `AGENTS.md` "Verification Commands" to point at the make targets.

**Verify:** `make lint && make test` (or `just lint && just test`) is green.

---

## Phase 3 — Docs, figures, and metadata hygiene

Goal: remove the "unfinished repo" tells. Mostly non-code; a couple of items need Docker.

### 3.1 Regenerate result figures — `img/` (needs Docker)

**Why:** `img/pinn_*.png` predate the fixed-Fourier + solver-backed path, and both `README.md`
and `AGENTS.md` carry a public "regenerate before treating as benchmark" caveat. Shipping
results with an asterisk reads as unfinished.

**Implement:**
- Regenerate the three figures (`pinn_solution_comparison.png`, `pinn_field_solution_jax.png`,
  `pinn_field_solution_pytorch.png`) from a current run against the real solver (the Streamlit
  "Solution Field" tab produces exactly these three-panel PINN / solver-ground-truth / error
  plots — `generate_solution_grid` in `app.py` is the reference). A small standalone script
  under `scripts/` that calls `train_inverse` then `generate_solution_grid` and `savefig`s is
  acceptable and reproducible; prefer that over manual screenshots.
- Update captions in `README.md` to "PINN vs solver ground truth" and **remove the "regenerate
  before treating as benchmark" caveats** from both `README.md` and `AGENTS.md` once the
  figures are current.

**Verify:** figures show nonlinear Burgers fields; no stale caveat remains in `README.md`/`AGENTS.md`.

### 3.2 Demo script — `docs/demo_script.md`

**Why:** a customer showcase needs a "what to run / what to look at" narrative.

**Implement:** create `docs/demo_script.md` with: (1) one-time setup (`uv sync`, `./buildall.sh`);
(2) the CLI walk-through (`--backend both`, point out the same loop, the VJP-through-PyTorch
note, the consistency table); (3) the Streamlit walk-through tab by tab (Run Summary, Inverse
Solve, Solution Field, BRDR Weights, **Tesseract Trace** — call out that apply/VJP counts are
*measured*, not estimated, and Backend Consistency — call out independent inits); (4) the
benchmark command and where artifacts land. Keep it tight (one screen per section).

### 3.3 Project metadata — `pyproject.toml`

**Why:** `name = "tesseract-hackathon"` (and the generated `tesseract_hackathon.egg-info/`)
directly contradict "productionised."

**Implement:**
- Rename the project (e.g. `tesseract-pinn-inverse-burgers`, matching the repo). Update the
  `name` field; reinstall (`uv pip install -e .`) so a fresh `*.egg-info` is generated; delete
  the stale `tesseract_hackathon.egg-info/` directory.
- Confirm `[tool.setuptools.packages.find]` does not try to package the `tesseracts/` container
  dirs or the top-level scripts in a way that breaks the editable install. The importable
  surface is the top-level modules `app.py`, `inverse_problem.py`, `configs.py`; keep that
  working.

**Verify:** `uv run python -c "import inverse_problem, configs"` and `uv run pytest tests -q`
both pass after rename; no `hackathon` string remains in `pyproject.toml` or tracked metadata.

### 3.4 Reconcile roadmaps — `PLAN.md`, `TIER0_PLAN.md`, `AGENTS.md`

**Why:** `PLAN.md` still describes heat-equation observations as the *current* state (its
Context section is stale — that was fixed in the Tier-0 pass), and `TIER0_PLAN.md` is a
completed plan. A reviewer browsing the repo reads contradictory roadmaps.

**Implement:**
- `AGENTS.md` is the single source of truth (`CLAUDE.md` already points there). Update its
  "Current State" / "Current Limitations" to reflect Phase 1: shared engine, `value_and_grad`,
  measured telemetry, config validation, consistency framing. Remove the now-false "Model
  reconstructed from flat params on every call" only if Phase 1 changed it (it did **not** —
  the container still reconstructs per call; leave that limitation).
- Archive `TIER0_PLAN.md` (it's done) — move under `docs/archive/` or delete. Fix `PLAN.md`'s
  Context so it does not claim heat-equation data is current; keep its forward-looking
  flow-matching roadmap.

### 3.5 Dependency source of truth — `requirements.txt` vs `pyproject.toml`

**Why:** both exist and can drift.

**Implement:** make `pyproject.toml` the single source. Either delete `requirements.txt` or
generate it from the lock (`uv export`) and note in the README that it is generated. Ensure the
README install instructions match whichever you keep.

**Verify:** a clean `uv sync` from `pyproject.toml` alone produces a working environment.

---

## Deferred feature (separate effort — do NOT bundle into Phase 2/3)

### Compositional two-Tesseract `jax.grad` demo mode

**Why it matters:** the genuinely Tesseract-unique story is `jax.grad` composing VJPs through
**two** containers in one objective: `nu → burgers_solver_tesseract(nu) → u_obs(nu)`, and
`nu, params → PINN loss`, so `dL/dnu` includes a solver contribution. The solver
(`tesseracts/burgers_solver/`) already exposes full `apply`/VJP/JVP and is currently used only
as an offline data generator. This is a feature, not polish — schedule it on its own.

**Sketch (for a later effort, not now):**
- Add a demo mode where observations are produced *inside* the differentiated objective via
  `apply_tesseract(burgers_solver, {...nu...})` rather than precomputed, so a single
  `value_and_grad` flows through both the solver and the PINN containers.
- Expose it as a separate Streamlit mode / CLI flag, not the default inverse problem (the
  self-generated-observations setup is conceptually different from fitting fixed measurements —
  see `AGENTS.md` "Planned Extensions" §2 for the open design questions).
- Keep the composition cheap (small `nx/nt`) for interactive use.

This is the headline that turns the repo from "a PINN inversion demo" into "differentiating
through a containerized solver *and* a containerized model in one JAX gradient" — the thing
worth showing Pasteur.

---

## Suggested order & PR boundaries

1. PR: 2.1 smoke test + 2.4 ruff config/Makefile (small, safe, immediately green CI inputs).
2. PR: 2.3 CI workflow (depends on 2.4's ruff config and the self-skipping smoke test).
3. PR: 2.2 benchmark artifacts.
4. PR: Phase 3 docs/figures/metadata (3.1 needs Docker; the rest don't).
5. Separate, later: the deferred compositional mode.
</content>
</invoke>
